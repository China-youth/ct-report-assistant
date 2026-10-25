"""LangGraph StateGraph for the brain-CT report pipeline.

Graph structure (explicit nodes + edges + conditional edges):

    vision ──(valid)────► retrieval ──► generate ──► check
       │                                    ▲          │
       │                                    └──────────┘
       │                                         │ (pass / exhausted)
    vision ──(invalid/undetermined)► refuse ─────┴──► END

State fields:
  - original_findings: raw user input text (the "影像所见")
  - image_status:      "valid" | "invalid" | "undetermined"
  - refuse_reason:     set by the refuse node when the image is unusable
  - evidence:          retrieved knowledge entries
  - drafts:            list of report drafts, one per iteration
  - selfcheck:         { passed: bool, issues: [..] } of the last run
  - iteration:         current iteration counter (starts at 0)
  - report:            the final structured report dict (empty on refuse)
"""
from __future__ import annotations

import json
import re
import time
from typing import Annotated, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ._llm import KNOWLEDGE_BASE, OUT_OF_SCOPE_MARKERS, get_model, preprocess_image, retrieve

MAX_ITER = 2  # generate -> check loop ceiling; 0,1 allowed, on 2nd fail -> END

# Build stamp for online verification: lets operators confirm which revision
# is actually running (compared against the value printed after each deploy).
BUILD = "2026-10-23T21-38-formal"

# Ablation modes for the /report request body `mode` field:
#   A = baseline: VLM read + one REPORT call straight to the report;
#       no retrieval, no check, no fallback loop
#   B = half: VLM read + knowledge retrieval + REPORT, but no check / no loop
#   C = full: the complete graph (check + fallback loop), i.e. previous default
ABLATION_MODES = ("A", "B", "C")

DISCLAIMER = (
    "本报告由 AI 辅助生成，仅用于辅助阅片与信息整理，"
    "不能替代执业医师的判断与面诊。任何临床决策请结合完整影像资料与患者实际情况，"
    "由具备资质的放射科医师或相关专科医师作出。"
)


class Report(BaseModel):
    """Structured report schema the model must fill."""
    findings: str = Field(description="影像所见：描述所见的解剖结构与异常。")
    impression: List[str] = Field(description="印象：分条列出主要诊断方向与可能性。")
    recommendation: List[str] = Field(description="建议：复查、进一步检查或临床随访建议。")


class State(TypedDict, total=False):
    original_findings: str
    image: str | None
    image_status: str  # "valid" | "invalid" | "undetermined"
    refuse_reason: str
    evidence: List[dict]
    drafts: List[dict]
    selfcheck: dict
    iteration: int
    report: dict
    node_timings: List[dict]
    token_usage: List[dict]
    error: str
    env: dict  # per-run gateway env, declared so LangGraph keeps it
    _debug: dict  # X-Debug: 1 时由 handler 预填，节点往里写中间量
    mode: str  # ablation mode: "A" | "B" | "C" (default "C")


# ---------------------------------------------------------------------------
# Deterministic validity gate. Status is decided by CODE RULES, not by what
# the model says. Rationale: the VLM sometimes returns a refusal sentence
# wrapped in explanatory prose, which a length-based "valid" default would
# wrongly let through into report generation (the 1x1-pixel regression).
# ---------------------------------------------------------------------------
_INVALID_PATTERNS = (
    r"无图像", r"未识别", r"无法识别", r"未提供",
    r"不是.{0,6}CT", r"非.{0,6}影像", r"无有效", r"无法判断",
    r"non-\s*CT", r"not a ?CT", r"no (?:usable|valid|readable) image",
    r"图像无效", r"没有(有效|可用)图像", r"无效图像",
)

# A genuine head-CT findings report must reference at least one anatomical /
# radiological term. Absence of all of these => undetermined.
_ANATOMY_TERMS = (
    "颅骨", "脑实质", "脑室", "中线", "脑沟", "脑裂", "大脑", "小脑",
    "脑干", "基底节", "白质", "灰质", "硬膜", "蛛网膜", "垂体", "鞍区",
    "脑疝", "血肿", "梗死", "出血", "占位", "水肿", "骨折", "脑积水",
    "侧脑室", "第三脑室", "透明隔", "大脑镰", "半卵圆", "尾状核", "丘脑",
    "额叶", "顶叶", "颞叶", "枕叶", "幕上", "幕下", "脑回", "皮层",
    "皮质", "脑叶", "脑池", "纵裂", "朕眠", "脑组织", "轴位", "层面",
    "高密度", "低密度", "CT",
)

_RX_INVALID = [re.compile(p, re.IGNORECASE) for p in _INVALID_PATTERNS]


# VLM prompt forces a machine-readable verdict (rule B); the JSON parser the
# vision node uses lives next to this definition.
_VLM_SYS = (
    "你是神经放射科影像分析助手。用户会给你一张图像。"
    "先判断该图像是否为一张可读的头颅 CT 轴位影像（is_brain_ct），"
    "再按 颅骨/脑实质/脑室系统/中线结构 四部分输出结构化影像所见（findings）。"
    "输出且只输出一个 JSON 对象：\n"
    '{"is_brain_ct": true 或 false, "findings": "四段所见文本 或 空字符串"}\n'
    "若图像不可读、不是头颅 CT（1x1、损坏、非医学图片等），is_brain_ct=false，"
    "findings 置空字符串。不要输出 JSON 以外的任何文字。"
)

# The literal text of the thinking-closing tag, built from parts so the
# raw characters never appear inside a string literal (they corrupt
# editors / the source). Length is 8:  / { t h i n k }  .
THINK_CLOSE_TAG = '/{think}'


def parse_vlm_reply(reply: str) -> dict:
    """Parse the VLM's forced-JSON reply. Missing / malformed -> undetermined
    (caller maps that to a refuse, never to a generated report).

    minimax-m3 inlines chain-of-thought in the same ``content`` field: the
    final JSON object (if any) comes AFTER the closing tag. Strategy:
      1. strip code fences, try to parse the whole text;
      2. on failure, take the tail after the last closing tag and re-parse;
      3. last resort: regex-extract the outermost ``{...}`` block containing
         an ``is_brain_ct`` key.
    """
    text = (reply or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    closing = text.rfind(THINK_CLOSE_TAG)

    if closing != -1:
        candidates.append(text[closing + len(THINK_CLOSE_TAG):].strip())
    # outermost JSON object that mentions is_brain_ct
    m = re.search(r"\{.*?\"is_brain_ct\".*?\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))

    # Qwen3-VL may wrap JSON in a markdown code block or add a prose preamble.
    # As a last resort, extract all {...} blocks and try each one.
    all_blocks = re.findall(r"\{[^{}]*\}", text)
    for block in all_blocks:
        candidates.append(block)

    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and "is_brain_ct" in obj:
                findings = obj.get("findings", "")
                if not isinstance(findings, str):
                    findings = json.dumps(findings, ensure_ascii=False)
                return {"parsed": True, "is_brain_ct": bool(obj["is_brain_ct"]),
                        "findings": findings}
        except (json.JSONDecodeError, TypeError):
            continue
    return {"parsed": False, "is_brain_ct": None, "findings": ""}


def _classify_vlm_reply(findings: str) -> str:
    """Deterministic valid / invalid / undetermined — never model self-report.

    Rules (in priority order):
      1. findings matches any refusal pattern            -> "invalid"
      2. findings empty / len < 10 / no anatomy keyword   -> "undetermined"
      3. otherwise (no refusal pattern AND has an anatomy
         keyword)                                        -> "valid"
    """
    text = (findings or "").strip()
    # 1) explicit refusal — hard invalid regardless of length
    if any(rx.search(text) for rx in _RX_INVALID):
        return "invalid"
    # 2) substance checks
    if not text or len(text) < 10:
        return "undetermined"
    if not any(t in text for t in _ANATOMY_TERMS):
        return "undetermined"
    # 3) valid
    return "valid"


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of an LLM response that may contain
    markdown code fences or surrounding prose."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _usage_of(resp) -> dict:
    """Extract per-call token usage from a LangChain AIMessage.

    The Makers gateway returns OpenAI-style usage blocks; anything missing
    is reported as 0 so the ablation accounting never fails on shape.
    """
    usage = getattr(resp, "usage_metadata", None) or {}
    if not usage:
        info = getattr(resp, "response_metadata", {}) or {}
        usage = info.get("token_usage") or info.get("usage") or {}
    prompt = usage.get("input_tokens")
    completion = usage.get("output_tokens")
    total = usage.get("total_tokens")
    if prompt is None:
        prompt = usage.get("prompt_tokens", 0)
    if completion is None:
        completion = usage.get("completion_tokens", 0)
    if total is None:
        total = (prompt or 0) + (completion or 0)
    return {"prompt_tokens": prompt or 0,
            "completion_tokens": completion or 0,
            "total_tokens": total or 0}


def _record_usage(state: State, node: str, resp, model_id: str) -> dict:
    """Return the updated token_usage list with one entry appended."""
    usage_list = list(state.get("token_usage") or [])
    entry = {"node": node, "model": model_id}
    entry.update(_usage_of(resp))
    usage_list.append(entry)
    return {"token_usage": usage_list}


def timed(node: str):
    """Decorator that wraps an async node function and records its
    wall-clock duration into ``state['node_timings']``."""
    def deco(fn):
        async def _wrapped(state: State) -> dict:
            t0 = time.perf_counter()
            out = await fn(state)
            ms = int((time.perf_counter() - t0) * 1000)
            timings = list(state.get("node_timings", []))
            timings.append({"node": node, "ms": ms})
            out["node_timings"] = timings
            return out
        return _wrapped
    return deco


# ---------------------------------------------------------------------------
# Node: vision — image understanding + validity gate.
# Produces structured findings from the VLM AND a validity status so the
# graph can short-circuit when the input is not a usable head-CT image.
# ---------------------------------------------------------------------------
@timed("vision")
async def vision_node(state: State) -> dict:
    """Image understanding + three-layer validity gate.

    Layers (first non-valid layer wins; a single refusal routes to refuse):
      B. VLM forced-JSON output ``{"is_brain_ct": bool, "findings": str}`` —
         the code reads the boolean, never trusts free-text self-assessment.
      C. deterministic text rules on the findings (refusal patterns + anatomy
         keywords) as a last-layer safety net.
      A. request-level image check (dims/bytes) is precomputed by the handler
         and short-circuits the whole graph before vision runs; if it is not
         valid, the pipeline never reaches this node.
    """
    env = state.get("env") or {}
    image = state.get("image")
    user_text = (state.get("original_findings") or "").strip()
    dbg = dict(state.get("_debug") or {})
    layers = {}

    findings = ""
    image_status = "undetermined"
    model_id = ""
    out: dict = {}
    if image:
        model = get_model(env, role="vlm")
        model_id = getattr(model, "model_name", "")
        parsed = {"parsed": False, "is_brain_ct": None, "findings": ""}
        try:
            # Shrink + re-encode before the provider sees the bytes, so the
            # request stays within quota regardless of the upload size.
            image = preprocess_image(image)
            resp = await model.ainvoke([
                {"role": "system", "content": _VLM_SYS},
                {"role": "user", "content": [
                    {"type": "text", "text":
                        "判断这张图像是否为可读的头颅 CT 轴位影像（is_brain_ct），"
                        "并输出结构化影像所见。只输出一个 JSON 对象：\n"
                        '{"is_brain_ct": true或false, "findings": "四段所见文本或空字符串"}\n'
                        "不要输出 JSON 以外的任何文字。"},
                    {"type": "image_url", "image_url": {"url": image}},
                ]},
            ])
            vlm_reply = (resp.content or "").strip()
            parsed = parse_vlm_reply(vlm_reply)
            # minimax-m3 inlines chain-of-thought; the final JSON (if any)
            # follows the closing think tag. Retry the parser on the tail.
            if not parsed["parsed"]:
                closing = vlm_reply.rfind(THINK_CLOSE_TAG)
                if closing != -1:
                    parsed = parse_vlm_reply(vlm_reply[closing + 8:])
            layers["vlm_json"] = {
                "raw_len": len(vlm_reply),
                "parsed": parsed["parsed"],
                "is_brain_ct": parsed["is_brain_ct"],
                "findings_len": len(parsed.get("findings") or ""),
            }
            out.update(_record_usage(state, "vision", resp, model_id))
        except Exception as e:
            layers["vlm_error"] = str(e)[:200]

        # Rule B verdict (reads the boolean, not free text).
        if not parsed["parsed"] or parsed["is_brain_ct"] is None:
            image_status = "undetermined"
            findings = ""
        elif parsed["is_brain_ct"] is False:
            image_status = "invalid"
            findings = ""
        else:
            findings = (parsed.get("findings") or "").strip()
            # Rule C: text rules on the VLM's own findings.
            rule_c = _classify_vlm_reply(findings)
            layers["text_rule_on_findings"] = {
                "input": findings,
                "result": rule_c,
            }
            image_status = "valid" if rule_c == "valid" else rule_c
            if image_status != "valid":
                findings = ""
    else:
        # No image: run text rules on user-supplied findings.
        rule_c = _classify_vlm_reply(user_text)
        layers["text_rule_on_findings"] = {
            "input": user_text,
            "result": rule_c,
        }
        image_status = rule_c
        findings = user_text if rule_c == "valid" else ""

    dbg.update({
        "rule_a": state.get("img_check"),
        "rule_b": (layers.get("vlm_json")
                   or ({"vlm_error": layers["vlm_error"]}
                       if "vlm_error" in layers else None)),
        "rule_c": layers.get("text_rule_on_findings"),
        "classify_result": image_status,
        "state_keys": ["original_findings", "image_status", "iteration", "drafts",
                        "report", "selfcheck", "evidence"],
        "findings_at_vision": (findings or "")[:500],
    })

    return {
        "original_findings": findings,
        "image_status": image_status,
        "iteration": 0,
        "drafts": [],
        "report": {},
        "selfcheck": {"passed": None, "issues": []},
        "evidence": [],
        "token_usage": out.get("token_usage", []),
        "_debug": dbg,
    }


# ---------------------------------------------------------------------------
# Node: refuse — short-circuit when the image is not a usable head-CT.
# Returns the same top-level shape as a normal run (report=None + reason).
# ---------------------------------------------------------------------------
@timed("refuse")
async def refuse_node(state: State) -> dict:
    reason = "未识别到有效头颅CT影像，请重新上传清晰、完整的头颅 CT 轴位图像。"
    return {
        "report": None,
        "refuse_reason": reason,
        "selfcheck": {"passed": None, "issues": ["图像不可用，未生成报告"]},
        "iteration": 0,
    }


# ---------------------------------------------------------------------------
# Node: retrieval — local knowledge base lookup on the findings text.
# ---------------------------------------------------------------------------
@timed("retrieval")
async def retrieval_node(state: State) -> dict:
    text = state.get("original_findings", "")
    terms = [k for k in KNOWLEDGE_BASE.keys() if k in text]
    # Also try the user's own tokens against the keys for loose matching.
    extra_terms = re.findall(r"(?:出血|梗死|肿瘤|脑室|中线|脑沟|颅骨)", text)
    all_terms = list(dict.fromkeys(terms + extra_terms))
    evidence = retrieve(all_terms)
    return {"evidence": evidence}


# ---------------------------------------------------------------------------
# Node: generate — synthesize findings/impression/recommendation.
# ---------------------------------------------------------------------------
@timed("generate")
async def generate_node(state: State) -> dict:
    env = state.get("env") or {}
    model = get_model(env, role="report")
    model_id = getattr(model, "model_name", "")
    findings = state.get("original_findings", "")
    evidence = state.get("evidence", [])
    prev_drafts = state.get("drafts", [])
    prev_check = state.get("selfcheck", {})

    sys_prompt = (
        "你是一名神经放射科影像报告撰写助手。基于给定的影像所见文本与检索到的"
        "鉴别要点，输出一份结构化报告（JSON）。"
        "要求：所见要忠实于用户输入，不臆造未提及的解剖结构；"
        "印象段【不得留空】：至少给出 1 条，用'不能排除…/考虑…/需进一步评估…'"
        "'未见明确…但不能排除…'等克制措辞，避免直接下'确诊'或治疗承诺；"
        "建议段【不得留空】：至少给出 1 条，以复查、进一步检查或结合临床随访为主。"
        "只输出 JSON，不要任何解释。"
    )
    user_parts = [f"影像所见：\n{findings or '（未提供）'}"]
    if evidence:
        user_parts.append(
            "检索到的鉴别要点：\n"
            + json.dumps(evidence, ensure_ascii=False, indent=2)
        )
    if prev_drafts:
        user_parts.append("上一次草稿：\n" + json.dumps(prev_drafts[-1], ensure_ascii=False))
    if prev_check and not prev_check.get("passed", True):
        user_parts.append(
            "上一次自检发现的问题（请在本次修改中解决）：\n"
            + json.dumps(prev_check.get("issues", []), ensure_ascii=False)
        )
    user_parts.append(
        "请输出符合 schema 的 JSON 对象，键为 findings(str), impression(list[str]), recommendation(list[str])。"
    )

    resp = await model.ainvoke(
        [SystemMessage(content=sys_prompt),
         HumanMessage(content="\n".join(user_parts))]
    )
    out = _record_usage(state, "generate", resp, model_id)
    draft = _extract_json(resp.content or "")
    # Track whether the fallback filler had to be used, so the ablation CSV
    # can mark which cells have structurally-complete reports vs. model-filled.
    fallback_used = False
    # Coerce fields into the Report schema shape; fall back to sensible
    # defaults so downstream nodes never see a half-filled dict.
    if not isinstance(draft, dict):
        draft = {}
    draft.setdefault("findings", findings)
    if not isinstance(draft.get("impression"), list) or not draft.get("impression"):
        # Fallback: a hedged impression so the report never ships with an
        # empty 印象 segment (the prompt asks the model to fill it, but a
        # blank from the model is recovered here rather than surfacing).
        draft["impression"] = [
            "结合影像所见，具体诊断方向需结合临床病史及进一步影像评估。"
        ]
        fallback_used = True
    if not isinstance(draft.get("recommendation"), list) or not draft.get("recommendation"):
        draft["recommendation"] = [
            "建议结合临床，必要时复查或行进一步影像检查。"
        ]
        fallback_used = True
    try:
        validated = Report(**{
            "findings": draft.get("findings", ""),
            "impression": [str(x) for x in draft.get("impression", [])],
            "recommendation": [str(x) for x in draft.get("recommendation", [])],
        })
        draft = validated.model_dump()
    except Exception:
        pass  # keep the coerced dict
    drafts = list(state.get("drafts", [])) + [draft]
    ret = {"drafts": drafts, "report": draft,
           "iteration": state.get("iteration", 0) + 1,
           "fallback_used": fallback_used}
    ret.update(out)
    return ret


# ---------------------------------------------------------------------------
# Node: check — consistency self-check.
# ---------------------------------------------------------------------------
@timed("check")
async def check_node(state: State) -> dict:
    env = state.get("env") or {}
    # Independent model family for the cross-check (never the same model that
    # wrote the report).
    model = get_model(env, role="check")
    model_id = getattr(model, "model_name", "")
    report = state.get("report", {})
    evidence = state.get("evidence", [])
    findings_text = state.get("original_findings", "")

    # 1) Out-of-scope markers — hard block.
    blob = json.dumps(report, ensure_ascii=False)
    oos = [m for m in OUT_OF_SCOPE_MARKERS if m in blob]

    # 1b) Vacuous-findings guard — hard block. If the underlying findings are
    # empty or purely negative ("未见/无异常") yet the report asserts a
    # diagnostic impression, that is a hallucination by construction.
    findings_norm = (findings_text or "").strip()
    impression_blob = json.dumps(report.get("impression", []), ensure_ascii=False)
    diagnostic_terms = ("出血", "梗死", "肿瘤", "占位", "血肿", "骨折", "积水")
    assert_dx = any(t in impression_blob for t in diagnostic_terms)
    findings_is_vacuous = (
        not findings_norm
        or _looks_purely_negative(findings_norm)
    )
    vacuous_issue = None
    if assert_dx and findings_is_vacuous:
        vacuous_issue = (
            "findings 为空或仅含否定性描述，但印象段给出了诊断性判断"
            "（出血/梗死/占位等），属无依据推断，需改为不确定措辞或留空。"
        )

    # 2) Hallucination check via LLM.
    sys_prompt = (
        "你是影像报告一致性自检助手。判断下方报告是否存在：(a) 幻觉（"
        "提到用户输入中未出现的解剖结构或异常），(b) 超出影像科报告范畴"
        "（直接确诊、治疗承诺），(c) 术语错误。"
        "只输出 JSON：{\"passed\": bool, \"issues\": [string]}。"
        "若无问题，passed 为 true，issues 为空数组。"
    )
    user = (
        f"原始影像所见：\n{findings_text}\n\n"
        f"待检报告（JSON）：\n{json.dumps(report, ensure_ascii=False)}\n\n"
        f"参考鉴别要点：\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"【硬性阻断】检测到超范围用语：{oos or '无'}"
    )
    try:
        resp = await model.ainvoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=user)]
        )
        parsed = _extract_json(resp.content or "")
        out = _record_usage(state, "check", resp, model_id)
    except Exception as e:
        parsed = {"passed": False, "issues": [f"自检调用失败：{e}"]}
        out = {}

    issues = list(parsed.get("issues", []))
    if oos:
        issues.append("超范围用语：" + "、".join(oos))
    if vacuous_issue:
        issues.append(vacuous_issue)

    passed = bool(parsed.get("passed", False)) and not oos and not vacuous_issue
    ret = {"selfcheck": {"passed": passed, "issues": issues}}
    ret.update(out)
    return ret


def _looks_purely_negative(text: str) -> bool:
    """True when findings contain only negative/absence statements and no
    positive lesion description. Used by the check node to block a report that
    asserts a diagnosis on top of an empty or purely-'no abnormality' base."""
    if not text:
        return True
    # Any affirmative lesion descriptor flips the verdict to 'not vacuous'.
    positive_markers = (
        "高密度", "低密度", "影", "灶", "缺损", "肿胀", "移位",
        "增宽", "扩大", "不对称", "骨折", "占位", "血肿", "渗出",
    )
    has_positive = any(m in text for m in positive_markers)
    # Strip the negation scaffolding; if what remains is essentially empty or
    # still negated, treat as purely negative.
    residual = re.sub(
        r"未见|未见明显|未显示|无明显|无|未识别|未明确|未确定|欠清|尚可|正常|对称|居中",
        "", text,
    )
    residual = re.sub(r"[，。、；：,.;:\s]", "", residual)
    return not has_positive and len(residual) <= 12


def _route_after_vision(state: State) -> str:
    """Route on the vision validity gate.

    Only an explicit "valid" continues; "invalid", "undetermined", missing,
    or any unrecognized value routes to refuse. Mode A (baseline) skips
    retrieval entirely and goes straight to generate.
    """
    status = state.get("image_status")
    if status in ("invalid", "undetermined") or status != "valid":
        return "refuse"
    if state.get("mode") == "A":
        return "generate"
    return "retrieval"


def _route_after_check(state: State) -> str:
    sc = state.get("selfcheck", {})
    it = state.get("iteration", 0)
    if sc.get("passed") is True:
        return END
    if it < MAX_ITER:
        return "generate"
    return END


def _route_after_generate(state: State) -> str:
    """Ablation routing after generate.

    - Mode A: no self-check at all; the single REPORT call IS the report.
      Mark selfcheck as skipped and go straight to END.
    - Mode B: half-pipeline — no check, no fallback loop. End here.
    - Mode C (default / unknown): full pipeline — run the consistency check.
    """
    mode = state.get("mode") or "C"
    if mode in ("A", "B"):
        sc = state.get("selfcheck") or {}
        sc["skipped"] = True
        sc.setdefault("passed", None)
        return "END"
    return "check"


def build_graph():
    g = StateGraph(State)
    g.add_node("vision", vision_node)
    g.add_node("refuse", refuse_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("generate", generate_node)
    g.add_node("check", check_node)
    g.add_edge(START, "vision")
    # Validity gate: only "valid" proceeds. Mode A (baseline) skips retrieval
    # and goes straight to generate; the map keys must match the literals
    # returned by _route_after_vision.
    g.add_conditional_edges(
        "vision", _route_after_vision,
        {
            "retrieval": "retrieval",
            "generate": "generate",
            "invalid": "refuse",
            "undetermined": "refuse",
            "refuse": "refuse",
        },
    )
    g.add_edge("refuse", END)
    g.add_edge("retrieval", "generate")
    # Ablation split after generate: modes A/B stop here, mode C continues
    # to the consistency check (+ fallback loop).
    g.add_conditional_edges("generate", _route_after_generate,
                            {"check": "check", "END": END})
    g.add_conditional_edges("check", _route_after_check,
                            {END: END, "generate": "generate"})
    return g.compile()


async def run_pipeline(ctx, findings_text: str, image: str | None,
                       img_check: dict | None = None, debug: bool = False,
                       mode: str = "C"):
    """Run the pipeline in the given ablation mode, returning a dict with
    report, selfcheck, timings, per-node token usage, and the mode used.

    ``mode``:
      - "A" baseline: VLM read + one REPORT call, no retrieval/check/loop
      - "B" half:     VLM read + retrieval + REPORT, no check/loop
      - "C" full:     the complete graph with check + fallback loop
    """
    mode = (mode or "C").upper()
    if mode not in ABLATION_MODES:
        mode = "C"
    graph = build_graph()
    initial: dict = {
        "original_findings": findings_text,
        "image": image,
        "env": ctx.env,
        "img_check": img_check,
        "mode": mode,
    }
    if debug:
        initial["_debug"] = {"enabled": True}
    result = await graph.ainvoke(initial)
    # refuse 分支（vision 状态非 valid）时 timings 只含 vision + refuse，
    # 结构上保证不会出现 generate/check（refuse 是独立节点，直连 END）。
    refused = result.get("image_status") != "valid"
    usage = result.get("token_usage") or []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usage:
        for k in totals:
            totals[k] += u.get(k, 0)
    # fallback_used is True when the generate node had to fill in a
    # template default for an empty impression / recommendation segment.
    fallback_used = bool(result.get("fallback_used"))
    out = {
        "build": BUILD,
        "mode": mode,
        "report": None if refused else result.get("report"),
        "refuse_reason": result.get("refuse_reason", "") if refused else "",
        "image_status": result.get("image_status", "undetermined"),
        "selfcheck": result.get("selfcheck", {}),
        "node_timings": result.get("node_timings", []),
        "usage": {
            "per_node": usage,
            "totals": totals,
            "model_calls": len(usage),
        },
        "iteration": result.get("iteration", 0),
        "fallback_used": fallback_used,
        "evidence": result.get("evidence", []),
        "disclaimer": DISCLAIMER,
    }
    if debug:
        out["_debug"] = result.get("_debug", {})
    return out

