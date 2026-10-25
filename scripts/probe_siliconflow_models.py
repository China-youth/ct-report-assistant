"""One-shot probe of https://api.siliconflow.cn/v1/models.

Prints the full raw JSON response (timestamped) so the agent can cite
which Qwen2.5-VL model IDs the SiliconFlow endpoint actually exposes
before choosing SILICONFLOW_VLM_MODEL.
"""
import datetime
import json
import sys
import urllib.request

URL = "https://api.siliconflow.cn/v1/models"
API_KEY = sys.argv[1] if len(sys.argv) > 1 else ""

if not API_KEY:
    print("usage: python scripts/probe_siliconflow_models.py <API_KEY>")
    sys.exit(1)

ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
print(f"[probe] {ts}  GET {URL}")
print(f"[probe] key: {API_KEY[:4]}...{API_KEY[-4:]}  (length {len(API_KEY)})")

req = urllib.request.Request(
    URL, headers={"Authorization": f"Bearer {API_KEY}"}
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read()
        print(f"[probe] status={resp.status}")
        print(f"[probe] body length={len(body)}")
        data = json.loads(body)
        print(f"[probe] data keys: {list(data.keys())}")
        if "models" in data:
            models = data["models"]
            print(f"[probe] model count: {len(models)}")
            for m in models:
                mid = m.get("id") if isinstance(m, dict) else m
                print(f"  - {mid}")
        else:
            print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])
except urllib.error.HTTPError as e:
    print(f"[probe] HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:500]}")
except Exception as e:
    print(f"[probe] error: {e}")
