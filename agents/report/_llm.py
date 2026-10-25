"""Shared LLM + local knowledge base for the CT report pipeline.

- The LLM is configured from the platform-injected environment
  (``ctx.env``); it is never read from ``os.environ``.
- The knowledge base is a small, self-contained dictionary of
  differential-diagnosis anchors used by the retrieval node. It is
  deliberately local so the pipeline runs without an external search key.
"""
from __future__ import annotations

from typing import Dict, List

from langchain_openai import ChatOpenAI

# Shared image preprocessor (service + test scripts). Decoded uploads are
# shrunk to 1024px longest edge and re-encoded as JPEG q85 before they enter
# the vision provider — the only place image bytes become data URLs.
from .preprocess_image import preprocess_image

# ---------------------------------------------------------------------------
# Model access — text LLM only. Vision is not available on the free pool, so
# the vision node consumes user-provided text findings, not pixels.
# ---------------------------------------------------------------------------
# Three-model split, each with its own gateway env var (never shared):
#   VLM_MODEL      -> vision node, reads the uploaded CT image
#   REPORT_MODEL   -> report generation node (structured synthesis)
#   CHECK_MODEL    -> consistency self-check node (independent family,
#                     avoids "the model grading its own homework")
#
# Selection basis (see docs/VLM_SELECTION.md, raw probes in
# ~/ct_vlm_compare/real_results.json + compare_results.json):
#   - @makers/minimax-m3 was the only free-pool model that both (a) read real
#     brain-CT images correctly and (b) honestly refused non-CT / synthetic
#     images instead of hallucinating findings. 5/5 real CTs passed; 5/5
#     synthetic test set behaved as expected.
#   - @makers/kimi-k2.6 is EXCLUDED: it returns HTTP 400 (upstream moonshot
#     "invalid request") when the vision prompt is long and multiple images are
#     involved — not viable for the multi-slice + differential pipeline.
#   - @makers/deepseek-v4-pro is the strongest text model in the free pool;
#     used for report synthesis.
#   - @makers/deepseek-v4-flash (lightweight, different model family from
#     REPORT) is the cross-checker; it grades the draft, not the author.
VLM_MODEL = "@makers/minimax-m3"
REPORT_MODEL = "@makers/deepseek-v4-pro"
CHECK_MODEL = "@makers/deepseek-v4-flash"

# Per-node env-var name. The node reads its own model id from this key so the
# three models never collide on one shared AI_GATEWAY_MODEL.
ENV_KEYS = {
    "vlm": "VLM_MODEL",
    "report": "REPORT_MODEL",
    "check": "CHECK_MODEL",
}
_DEFAULTS = {"vlm": VLM_MODEL, "report": REPORT_MODEL, "check": CHECK_MODEL}

# When the Makers free pool is quota-exhausted (402), the vision node can be
# pointed at a third-party OpenAI-compatible provider with free vision quota
# (e.g. SiliconFlow). These two env vars are OPTIONAL: when both are set, the
# "vlm" role uses the third-party base_url + key + model instead of the Makers
# gateway. REPORT / CHECK keep using the Makers gateway regardless.
VLM_PROVIDER_ENV = {
    "base_url": "SILICONFLOW_BASE_URL",
    "api_key": "SILICONFLOW_API_KEY",
    "model": "SILICONFLOW_VLM_MODEL",
}
# Default third-party vision model (SiliconFlow). Overridable via env.
# Qwen2.5-VL family no longer listed at /v1/models (2026-10-23 probe).
# 30B-A3B validated 3/3 on real CTs 2026-10-23; 8B failed on 4 of 6.
_SILICONFLOW_VLM_DEFAULT = "Qwen/Qwen3-VL-30B-A3B-Instruct"

_model_cache: Dict[str, ChatOpenAI] = {}


def _vlm_provider(env: dict):
    """Return (base_url, api_key, model_id) for the vlm role.

    If the third-party provider is fully configured, use it; otherwise fall
    back to the Makers gateway so the default deployment keeps working.
    """
    base = (env.get(VLM_PROVIDER_ENV["base_url"]) or "").strip()
    key = (env.get(VLM_PROVIDER_ENV["api_key"]) or "").strip()
    model = (env.get(VLM_PROVIDER_ENV["model"]) or "").strip()
    if base and key:
        return base, key, model or _SILICONFLOW_VLM_DEFAULT
    return None


def get_model(env: dict, role: str) -> ChatOpenAI:
    """Return a cached ``ChatOpenAI``.

    ``role`` in {"vlm", "report", "check"} selects:
      1. the per-role env key from ``env`` (e.g. ``VLM_MODEL``) if present,
      2. otherwise the built-in default for that role.

    Model / base URL / key all come from ``env`` (``ctx.env`` on the platform),
    never hard-coded, never read from ``os.environ``.

    The ``vlm`` role may be redirected to a third-party provider when the
    ``SILICONFLOW_*`` vars are set (see :func:`_vlm_provider`); ``report`` and
    ``check`` always use the Makers gateway.
    """
    if role not in ENV_KEYS:
        raise ValueError(f"unknown model role: {role!r}")

    third = _vlm_provider(env) if role == "vlm" else None
    if third is not None:
        base_url, api_key, model_id = third
    else:
        model_id = (env.get(ENV_KEYS[role]) or "").strip() or _DEFAULTS[role]
        base_url = (env.get("AI_GATEWAY_BASE_URL") or "").strip()
        api_key = (env.get("AI_GATEWAY_API_KEY") or "").strip()

    if not base_url or not api_key:
        raise RuntimeError(
            "AI_GATEWAY_BASE_URL / AI_GATEWAY_API_KEY missing from context.env"
        )
    cache_key = f"{role}:{model_id}:{base_url}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    kwargs = dict(
        model=model_id,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=180,
        streaming=False,
        # Vision responses are structured JSON / short findings text.
        max_tokens=1024,
    )
    if role in ("report", "check"):
        # Text-report models (Makers gateway) need more headroom for the
        # impression / recommendation segments — 1024 was truncating them.
        kwargs["max_tokens"] = 4096
    llm = ChatOpenAI(**kwargs)
    _model_cache[cache_key] = llm
    return llm


# ---------------------------------------------------------------------------
# Local knowledge base — brain CT differential-diagnosis anchors.
# Keys are short Chinese terms likely to appear in a CT finding.
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE: Dict[str, List[str]] = {
    "出血": [
        "脑出血密度偏高，CT 上呈高密度灶",
        "需鉴别硬膜下血肿、脑室内出血、蛛网膜下腔出血",
        "注意血肿部位与中线是否移位，评估占位效应",
    ],
    "梗死": [
        "急性脑梗死早期 CT 可无异常，24h 后出现低密度灶",
        "需结合发病时间窗与血管分布（前循环/后循环）",
        "鉴别分水岭梗死、腔隙性梗死",
    ],
    "肿瘤": [
        "脑内占位需关注强化方式、水肿范围、中线移位",
        "鉴别原发（胶质瘤、脑膜瘤）与转移瘤",
        "增强 CT 或 MRI 有助于进一步定性",
    ],
    "脑室": [
        "脑室扩大提示脑积水或脑萎缩，需结合年龄",
        "交通性 vs 梗阻性脑积水，注意导水管是否通畅",
    ],
    "中线移位": [
        "提示一侧占位/出血导致推挤，需评估脑疝风险",
        "记录移位程度（mm），与临床沟通病情",
    ],
    "脑沟": [
        "脑沟增宽提示脑萎缩，老年人常见",
        "结合皮层厚度判断是否存在皮质下低密度",
    ],
    "颅骨": [
        "颅骨骨折需与血管损伤、硬膜外血肿相关联",
        "注意矢状窦附近骨折提示硬膜外血肿风险",
    ],
}

# Terms that suggest the model is overstepping / hallucinating diagnosis.
OUT_OF_SCOPE_MARKERS = [
    "确诊",
    "确诊为",
    "恶性",
    "癌",
    "肿瘤性质",
    "立即手术",
    "治愈",
    "无风险",
]


def retrieve(terms: List[str], top_k: int = 5) -> List[dict]:
    """Return knowledge entries matching any of the given terms.

    Each returned entry is ``{"term": ..., "points": [...]}``.
    """
    results: List[dict] = []
    seen = set()
    for t in terms:
        t = t.strip()
        if not t:
            continue
        for key, points in KNOWLEDGE_BASE.items():
            if key in t or t in key:
                if key not in seen:
                    seen.add(key)
                    results.append({"term": key, "points": points})
    return results[:top_k]
