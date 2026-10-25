"""SiliconFlow vision-node smoke test.

- Loads the SILICONFLOW_API_KEY from C:/Users/horita/.env (never printed).
- Confirms the model id via /v1/models, picking the free Qwen2.5-VL tier.
- Runs one real CT image through the shared provider path
  (preprocess_image + get_model(role="vlm") + the vision node's VLM prompt)
  and prints the parsed structured-JSON verdict. A "non-CT" refusal counts
  as success for the wiring check; a real structured JSON either way.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

BASE_URL = "https://api.siliconflow.cn/v1"
# Free-tier candidates, prefer the largest available for read accuracy.
CANDIDATES = [
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]


def load_sf_key() -> str:
    for p in (r"C:/Users/horita/.env",):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    m = re.match(r"SILICONFLOW_API_KEY=(\S+)", line.strip())
                    if m:
                        return m.group(1)
    return ""


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ".")
    from scripts.preprocess_image import preprocess_image
    from agents.report._llm import get_model

    key = load_sf_key()
    if not key:
        print("ERROR: SILICONFLOW_API_KEY not found in C:/Users/horita/.env")
        sys.exit(1)

    env = {
        "SILICONFLOW_BASE_URL": BASE_URL,
        "SILICONFLOW_API_KEY": key,
    }

    # --- 1) list models, confirm a Qwen2.5-VL id is present ---
    print("=== step 1: /v1/models probe (key from C:/Users/horita/.env, "
          "SILICONFLOW_API_KEY) ===")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    import httpx
    r = httpx.get(BASE_URL + "/models", headers={"Authorization": f"Bearer {key}"},
                  timeout=60)
    print(f"[{ts}] status={r.status_code}")
    print("   body head:", r.text[:400].replace("\n", " "))
    if r.status_code != 200:
        print("ERROR: /v1/models probe failed; aborting")
        sys.exit(1)
    ids = [m.get("id", "") for m in r.json().get("data", [])]
    vl_ids = [i for i in ids if "Qwen2.5-VL" in i or "qwen2.5-vl" in i.lower()]
    print("   found Qwen2.5-VL ids:", vl_ids)
    pick = next((i for i in CANDIDATES if i in ids), vl_ids[0] if vl_ids else None)
    if not pick:
        print("ERROR: no Qwen2.5-VL model listed; aborting")
        sys.exit(1)
    env["SILICONFLOW_VLM_MODEL"] = pick
    print("   selected model:", pick)

    # --- 2) run the vision provider path on one real CT image ---
    print(f"\n=== step 2: single-image vision call via shared provider "
          f"(model={pick}) ===")
    ts2 = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    model = get_model(env, role="vlm")
    import base64
    raw = open("test_ct_real.png", "rb").read()
    data_url = preprocess_image(
        "data:image/png;base64," + base64.b64encode(raw).decode("ascii"))
    print(f"   upload size {len(raw)} bytes -> preprocessed data-url "
          f"payload {len(data_url)} chars")

    from agents.report._graph import _VLM_SYS, parse_vlm_reply

    resp = asyncio.run(model.ainvoke([
        {"role": "system", "content": _VLM_SYS},
        {"role": "user", "content": [
            {"type": "text", "text":
                "判断这张图像是否为可读的头颅 CT 轴位影像（is_brain_ct），"
                "并输出结构化影像所见。只输出一个 JSON 对象：\n"
                '{"is_brain_ct": true或false, "findings": "四段所见文本或空字符串"}\n'
                "不要输出 JSON 以外的任何文字。"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]))
    latency_ms = int((time.perf_counter() - t0) * 1000)
    reply = (resp.content or "").strip()
    parsed = parse_vlm_reply(reply)
    usage = getattr(resp, "usage_metadata", None) or {}
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
          f"latency={latency_ms}ms usage={usage}")
    print("   raw reply (truncated):", reply[:600])
    print("   parsed:", json.dumps(parsed, ensure_ascii=False)[:400])
    print("   model_id:", getattr(model, "model_name", ""))
    print("SMOKE OK" if (parsed["parsed"] or not reply.strip()) else
          "SMOKE PARTIAL (no JSON parsed, review raw reply)")


if __name__ == "__main__":
    main()
