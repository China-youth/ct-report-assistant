"""Deterministic + VLM-JSON verification for the three-layer gate.

Usage:  python -I test_shortcircuit.py [--no-network]
"""
import argparse
import base64
import importlib
import json
import os
import shutil
import sys
import types

sys.path.insert(0, "agents/report")
_pkg = types.ModuleType("ctpkg")
_pkg.__path__ = ["agents/report"]
sys.modules["ctpkg"] = _pkg
gm = importlib.import_module("ctpkg._graph")
ix = importlib.import_module("ctpkg.index")


def load_env(p):
    v = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("AI_GATEWAY_") and "=" in line:
                v[line.split("=", 1)[0]] = line.split("=", 1)[1].strip()
    return v


PIXEL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAABAAEAQj96KQAAAABJRU5ErkJggg=="

env = load_env(r"C:\Users\horita\ai-investor-edgeone\.env")
env.update({"VLM_MODEL": "@makers/minimax-m3",
            "REPORT_MODEL": "@makers/deepseek-v4-pro",
            "CHECK_MODEL": "@makers/deepseek-v4-flash"})

GOOD = "颅骨骨质连续，脑实质密度均匀，脑室系统未见扩大，中线结构居中，未见明显出血及梗死灶。"


def unit():
    """No-network checks: rule A + rule B parser + rule C classifier."""
    out = {}
    out["A_1x1"] = ix.check_image(PIXEL)
    out["A_empty"] = ix.check_image("")
    out["B_good"] = gm.parse_vlm_reply('{"is_brain_ct": true, "findings": "%s"}' % GOOD)
    out["B_false"] = gm.parse_vlm_reply('{"is_brain_ct": false, "findings": ""}')
    out["B_bad"] = gm.parse_vlm_reply("无图像可读：未识别到有效头颅CT影像")
    _cbr = chr(123) + "think" + chr(125)        # opening tag text
    _close = "/" + _cbr                          # closing tag text
    _json = '{"is_brain_ct": true, "findings": "%s"}' % GOOD
    out["B_coem"] = gm.parse_vlm_reply(_close + "\n" + _json)
    out["C_good"] = gm._classify_vlm_reply(GOOD)
    out["C_refuse"] = gm._classify_vlm_reply("未提供头颅CT图像，无法评估")
    out["C_empty"] = gm._classify_vlm_reply("")

    checks = [
        ("A_1x1 hard-blocks", not out["A_1x1"]["valid"]),
        ("B_good parses", out["B_good"]["parsed"] and out["B_good"]["is_brain_ct"] is True),
        ("B_false parses", out["B_false"]["parsed"] and out["B_false"]["is_brain_ct"] is False),
        ("B_bad unparsable", not out["B_bad"]["parsed"]),
        ("B_coem parsed via tag-strip", out["B_coem"]["parsed"]),
        ("C_good -> valid", out["C_good"] == "valid"),
        ("C_refuse -> invalid", out["C_refuse"] == "invalid"),
        ("C_empty -> undetermined", out["C_empty"] == "undetermined"),
    ]
    ok = True
    for name, passed in checks:
        print(("PASS " if passed else "FAIL "), name)
        ok = ok and passed
    print("UNIT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return out


def e2e(run_real_ct: bool):
    """Handler-level checks. The 1x1 case short-circuits at rule A (no network);
    only the real-CT case (run_real_ct=True) calls the VLM."""
    import asyncio

    class FakeCtx:
        def __init__(self, e):
            self.env = e
            self.request = None

    class FakeReq:
        def __init__(self, body, headers=None):
            self.body = body
            self.headers = headers or {}

    class FullCtx(FakeCtx):
        def __init__(self, e, body, headers=None):
            FakeCtx.__init__(self, e)
            self.request = FakeReq(body, headers)

    real_url = None
    if run_real_ct:
        # Prefer a locally-copied CT to avoid slow E-drive reads; fall back to E:.
        local_real = os.path.join(os.getcwd(), "test_ct_real.png")
        if os.path.exists(local_real):
            with open(local_real, "rb") as f:
                real_url = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        else:
            real_src = os.path.join("E:", "项目3 多模态数据集工程", "images_output",
                                    "8031efe0-1b5c-11ef-8929-000066532cad.png")
            if not os.path.exists(real_src):
                print("E2E real-CT skipped: no local copy and source not found")
                run_real_ct = False
            else:
                local = os.path.join(os.getcwd(), "test_ct.png")
                shutil.copyfile(real_src, local)
                with open(local, "rb") as f:
                    real_url = "data:image/png;base64," + base64.b64encode(f.read()).decode()

    async def main():
        res = {}
        # 1x1 through the FULL handler: rule A must hard-block before the graph
        ctx1 = FullCtx(env, {"description": "", "image": PIXEL}, {"x-debug": "1"})
        r1 = await ix.handler(ctx1)
        r1 = r1[0] if isinstance(r1, tuple) else r1
        res["e2e_1x1_handler"] = {
            "report_null": r1["report"] is None,
            "status": r1["image_status"],
            "refuse_reason": (r1.get("refuse_reason") or "")[:40],
            "nodes": [t["node"] for t in r1.get("node_timings", [])],
            "no_gen_check": not any(n in ("generate", "check")
                                    for n in [t["node"] for t in r1.get("node_timings", [])]),
            "build": r1["build"],
            "debug_rule_a": (r1.get("_debug") or {}).get("classify_inputs", {}).get("image_check"),
        }
        if run_real_ct:
            ctx2 = FullCtx(env, {"description": "", "image": real_url}, {"x-debug": "1"})
            r2 = await ix.handler(ctx2)
            r2 = r2[0] if isinstance(r2, tuple) else r2
            nodes = [t["node"] for t in r2.get("node_timings", [])]
            res["e2e_real_ct_handler"] = {
                "report_null": r2["report"] is None,
                "status": r2["image_status"],
                "iter": r2["iteration"],
                "nodes": nodes,
                "impression": ((r2.get("report") or {}).get("impression") or [])[:2],
                "build": r2["build"],
                "debug_vlm_json": (r2.get("_debug") or {}).get("classify_inputs", {}).get("vlm_json"),
                "findings_head": ((r2.get("_debug") or {}).get("findings_at_vision") or "")[:100],
            }
        return res

    res = asyncio.run(main())
    for k, v in res.items():
        print(k, "->", json.dumps(v, ensure_ascii=False, default=str))
    ok = (res["e2e_1x1_handler"]["report_null"]
          and res["e2e_1x1_handler"]["no_gen_check"])
    if run_real_ct:
        ok = ok and (not res["e2e_real_ct_handler"]["report_null"])
    print("E2E:", "ALL PASS" if ok else "FAILURES ABOVE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-network", action="store_true")
    a = ap.parse_args()
    unit()
    e2e(run_real_ct=not a.no_network)
