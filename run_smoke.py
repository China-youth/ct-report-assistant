"""Run the CT pipeline offline (reads gateway creds from the old project .env).

Usage:
    python -I run_smoke.py            # text-only, all 3 text-model roles
    python -I run_smoke.py --image <file.png>   # adds VLM read of that image
"""
import asyncio
import base64
import importlib
import json
import os
import sys
import types

sys.path.insert(0, "agents/report")
_pkg = types.ModuleType("ctpkg")
_pkg.__path__ = ["agents/report"]
sys.modules["ctpkg"] = _pkg
gm = importlib.import_module("ctpkg._graph")


class FakeCtx:
    def __init__(self, env):
        self.env = env


def load_env(path):
    vals = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("AI_GATEWAY_") and "=" in line:
                vals[line.split("=", 1)[0]] = line.split("=", 1)[1].strip()
    return vals


async def main():
    env = load_env(r"C:\Users\horita\ai-investor-edgeone\.env")
    # pin the three-model split for the offline run
    env["VLM_MODEL"] = "@makers/minimax-m3"
    env["REPORT_MODEL"] = "@makers/deepseek-v4-pro"
    env["CHECK_MODEL"] = "@makers/deepseek-v4-flash"

    image = None
    if "--image" in sys.argv:
        p = sys.argv[sys.argv.index("--image") + 1]
        with open(p, "rb") as f:
            image = "data:image/png;base64," + base64.b64encode(f.read()).decode()

    text = ("双侧大脑半球对称，脑沟脑裂未见明显增宽，脑室系统未见扩大，"
            "中线结构居中，左侧基底节区可见小片状低密度灶，未见明显出血。")
    if image:
        text = "（图像模式：所见由 VLM 读图生成，下方文本仅为检索锚点）"

    res = await gm.run_pipeline(FakeCtx(env), text, image)
    out = {
        "iteration": res["iteration"],
        "selfcheck": res["selfcheck"],
        "node_timings": res["node_timings"],
        "findings": res["report"].get("findings", ""),
        "impression": res["report"].get("impression"),
        "recommendation": res["report"].get("recommendation"),
        "evidence_terms": [e["term"] for e in res["evidence"]],
        "disclaimer_present": bool(res["disclaimer"]),
    }
    sys.stdout.buffer.write(
        json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8")
    )
    print()


asyncio.run(main())
