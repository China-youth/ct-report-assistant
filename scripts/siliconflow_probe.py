"""SiliconFlow /v1/models probe.

Reads SILICONFLOW_API_KEY from C:/Users/horita/.env (never printed),
GETs https://api.siliconflow.cn/v1/models, and returns the parsed JSON
along with the full list of model ids. Designed for repeated runs so
evidence can be captured with a timestamp on each call.
"""
from __future__ import annotations

import json
import os
import re
import time

BASE_URL = "https://api.siliconflow.cn/v1"


def _load_key() -> str:
    path = r"C:/Users/horita/.env"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r"SILICONFLOW_API_KEY=(\S+)", line.strip())
                if m:
                    return m.group(1)
    return ""


def probe(timeout: int = 30) -> dict:
    import httpx

    key = _load_key()
    if not key:
        return {"status_code": None, "error": "SILICONFLOW_API_KEY not found in C:/Users/horita/.env"}

    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        r = httpx.get(
            BASE_URL + "/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
        try:
            body = r.json()
        except Exception:
            body = {"_raw": r.text[:500]}
        ids = []
        data = body.get("data") if isinstance(body, dict) else None
        if isinstance(data, list):
            ids = [m.get("id", "") for m in data if isinstance(m, dict)]
        return {
            "ts": ts,
            "status_code": r.status_code,
            "ids": ids,
            "raw": r.text[:600],
            "body": body,
        }
    except Exception as e:
        return {"ts": ts, "status_code": None, "error": str(e), "ids": []}


if __name__ == "__main__":
    res = probe()
    print(json.dumps(res, ensure_ascii=False, indent=2)[:1200])
