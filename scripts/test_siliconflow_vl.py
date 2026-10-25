"""Single-image test against SiliconFlow Qwen3-VL Instruct models.

Run with SILICONFLOW_API_KEY in the shell (not printed in output).
Tries models in order; prints raw response fragment for each attempt.
Stop at the first model that returns parseable structured JSON with
is_brain_ct + findings fields.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

BASE_URL = "https://api.siliconflow.cn/v1"
TEST_IMAGE = "test_ct_real.png"

# Instruct only — never Thinking (breaks JSON output).
# Free tier first (8B), then larger free/paid fallbacks.
MODELS_TO_TRY = [
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
]

VLM_SYS = (
    "你是神经放射科影像分析助手。用户会给你一张图像。"
    "先判断该图像是否为一张可读的头颅 CT 轴位影像（is_brain_ct），"
    "再按 颅骨/脑实质/脑室系统/中线结构 四部分输出结构化影像所见（findings）。"
    "输出且只输出一个 JSON 对象：\n"
    '{"is_brain_ct": true 或 false, "findings": "四段所见文本 或 空字符串"}\n'
    "若图像不可读、不是头颅 CT（1x1、损坏、非医学图片等），is_brain_ct=false，"
    "findings 置空字符串。不要输出 JSON 以外的任何文字。"
)

VLM_USER_TEXT = (
    "判断这张图像是否为可读的头颅 CT 轴位影像（is_brain_ct），"
    "并输出结构化影像所见。只输出一个 JSON 对象：\n"
    '{"is_brain_ct": true或false, "findings": "四段所见文本或空字符串"}\n'
    "不要输出 JSON 以外的任何文字。"
)


def parse_vlm_reply(reply: str) -> dict:
    """Parse the VLM's forced-JSON reply. Strips markdown fences,
    Thinking closing tag, and extraneous wrappers.
    Returns {parsed: bool, is_brain_ct: bool|None, findings: str}
    """
    text = (reply or "").strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    closing = text.rfind("/{think}")
    if closing != -1:
        candidates.append(text[closing + 8:].strip())
    m = re.search(r"\{.*?\"is_brain_ct\".*?\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))

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


def try_model(model_id: str, key: str, image_data_url: str) -> dict:
    """Call one SiliconFlow model and return structured result."""
    import httpx

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    try:
        r = httpx.post(
            BASE_URL + "/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": VLM_SYS},
                    {"role": "user", "content": [
                        {"type": "text", "text": VLM_USER_TEXT},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ]},
                ],
                "max_tokens": 1024,
                "temperature": 0,
            },
            timeout=120,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        body = r.json() if r.headers.get("content-type","").startswith("application/json") else {"raw": r.text}
        return {
            "ts": ts,
            "model": model_id,
            "status_code": r.status_code,
            "latency_ms": latency_ms,
            "usage": body.get("usage", {}),
            "raw_fragment": json.dumps(body, ensure_ascii=False)[:600],
            "reply": body.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if isinstance(body.get("choices"), list) and body.get("choices")
                    else "",
        }
    except Exception as e:
        return {"ts": ts, "model": model_id, "status_code": None,
                "error": str(e), "raw_fragment": "", "reply": ""}


def main() -> None:
    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        print("ERROR: SILICONFLOW_API_KEY not in shell env")
        sys.exit(1)

    # Preprocess the test image: 1024px longest edge, JPEG q85
    import sys
    sys.path.insert(0, "agents/report")
    import types
    _pkg = types.ModuleType("ctpkg"); _pkg.__path__ = ["agents/report"]
    sys.modules["ctpkg"] = _pkg
    from ctpkg.preprocess_image import preprocess_image

    raw = open(TEST_IMAGE, "rb").read()
    data_url = preprocess_image("data:image/png;base64," + base64.b64encode(raw).decode())
    print(f"Test image: {TEST_IMAGE}  ({len(raw)} bytes original)")
    print(f"Preprocessed data-URL payload: {len(data_url)} chars")
    print()

    for i, model_id in enumerate(MODELS_TO_TRY, 1):
        print(f"=== Attempt {i}/{len(MODELS_TO_TRY)}: {model_id} ===")
        res = try_model(model_id, key, data_url)
        print(f"  [{res['ts']}] status={res['status_code']}  latency={res.get('latency_ms','?')}ms")
        print(f"  raw fragment: {res['raw_fragment'][:300]}")
        if res.get("error"):
            print(f"  ERROR: {res['error']}")
            print()
            continue
        if res["status_code"] != 200:
            print(f"  Provider rejected request (HTTP {res['status_code']}); trying next model")
            print()
            continue
        reply = res.get("reply", "")
        parsed = parse_vlm_reply(reply)
        print(f"  usage: {res.get('usage', {})}")
        print(f"  parsed JSON: {json.dumps(parsed, ensure_ascii=False)[:300]}")
        if parsed["parsed"]:
            print(f"\n[OK] {model_id} returned valid structured JSON.")
            print(f"  is_brain_ct={parsed['is_brain_ct']}  findings_len={len(parsed.get('findings',''))}")
            print("  FINAL: raw reply (first 400 chars):")
            print("  " + reply[:400].replace("\n", "\n  "))
            print(f"\nSELECTED_MODEL={model_id}")
            return
        else:
            print("  [X] No parseable JSON with is_brain_ct; trying next model")
            print()
    print("ALL ATTEMPTS FAILED — no valid structured JSON returned.")
    sys.exit(1)


if __name__ == "__main__":
    main()
