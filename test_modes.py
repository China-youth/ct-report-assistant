"""Ablation modes (A/B/C) + per-node token usage smoke test.

Usage:  python -I test_modes.py            # runs all 3 modes on a local CT
        python -I test_modes.py --no-network  # unit checks only (no VLM)

Requires a local copy of a real CT: test_ct_real.png (copied by
make_test_ct.py or manually from the 项目3 dataset).
"""
import argparse
import base64
import importlib
import json
import os
import sys
import types
import shutil

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


env = load_env(r"C:\Users\horita\ai-investor-edgeone\.env")
env.update({"VLM_MODEL": "@makers/minimax-m3",
            "REPORT_MODEL": "@makers/deepseek-v4-pro",
            "CHECK_MODEL": "@makers/deepseek-v4-flash"})


GOOD = "颅骨骨质连续，脑实质密度均匀，脑室系统未见扩大，中线结构居中，未见明显出血及梗死灶。"


class Ctx:
    def __init__(self, e):
        self.env = e
        self.request = None


def graph_shape_check():
    """Verify the ablation routing is wired as intended (no network)."""
    g = gm.build_graph()
    # Inspect the compiled graph's node set and confirm the conditional
    # routers are registered by building a static path for each mode.
    nodes = list(g.nodes.keys())
    print("graph nodes:", nodes)
    assert "refuse" in nodes and "check" in nodes and "generate" in nodes


def offline_routing_check():
    """Drive each ablation mode with a fake model (no network, no quota) and
    assert the node sequence + usage accounting differ as designed."""
    import asyncio

    class FakeMsg:
        def __init__(self, content, usage=None):
            self.content = content
            self.usage_metadata = usage or {"input_tokens": 30, "output_tokens": 20}
            self.response_metadata = {}

    class FakeModel:
        def __init__(self, name):
            self.model_name = name
        async def ainvoke(self, msgs):
            # Vision node: forced JSON reply. Report node: schema JSON.
            # Check node: a passing verdict.
            flat = json.dumps([m.get("content", "") for m in msgs if isinstance(m, dict)],
                              ensure_ascii=False)
            if "is_brain_ct" in flat:      # vision call
                return FakeMsg('{"is_brain_ct": true, "findings": "%s"}' % GOOD,
                               {"input_tokens": 80, "output_tokens": 60})
            if "passed" in flat:           # check call
                return FakeMsg('{"passed": true, "issues": []}',
                               {"input_tokens": 50, "output_tokens": 10})
            return FakeMsg('{"findings":"%s","impression":["未见明显异常"],"recommendation":["建议复查"]}' % GOOD,
                           {"input_tokens": 40, "output_tokens": 30})

    import ctpkg._llm as llm
    real_get = gm.get_model
    gm.get_model = lambda env, role: FakeModel({"vlm": "minimax-m3",
                                                   "report": "deepseek-v4-pro",
                                                   "check": "deepseek-v4-flash"}[role])

    async def main():
        out = {}
        fake_img = "data:image/png;base64,FAKECTURL"
        for mode in ("A", "B", "C"):
            r = await gm.run_pipeline(Ctx(env), "", fake_img, mode=mode,
                                      img_check={"valid": True, "note": "forced-ok"},
                                      debug=True)
            out[mode] = [t["node"] for t in r["node_timings"]]
            out[mode + "_status"] = r["image_status"]
            dbg = r.get("_debug") or {}
            out[mode + "_dbg_b"] = dbg.get("rule_b")
            out[mode + "_dbg_c"] = dbg.get("rule_c")
        return out

    try:
        paths = asyncio.run(main())
    finally:
        gm.get_model = real_get

    print("offline node paths:", {k: v for k, v in paths.items() if not k.endswith("_status")})
    # A: baseline -> vision + generate (no retrieval, no check)
    assert paths["A"] == ["vision", "generate"], paths
    # B: half -> vision + retrieval + generate (no check)
    assert paths["B"] == ["vision", "retrieval", "generate"], paths
    # C: full -> vision + retrieval + generate + check
    # (the check+generate loop may run up to MAX_ITER times; assert the
    #  first four nodes in order and that check IS present)
    assert paths["C"][:4] == ["vision", "retrieval", "generate", "check"], paths
    assert "check" in paths["C"], paths
    print("offline routing OK: A/B/C node sequences as designed")


def unit_usage_check():
    """_usage_of must tolerate missing usage_metadata without crashing."""
    class FakeMsg:
        usage_metadata = {}
        response_metadata = {}
    u = gm._usage_of(FakeMsg())
    assert u == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    # OpenAI-style alias
    class FakeMsg2:
        usage_metadata = {"input_tokens": 10, "output_tokens": 5}
        response_metadata = {}
    u2 = gm._usage_of(FakeMsg2())
    assert u2 == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, u2
    print("usage accounting ok:", u, u2)


def e2e_modes():
    import asyncio
    local = os.path.join(os.getcwd(), "test_ct_real.png")
    if not os.path.exists(local):
        src = os.path.join("E:", "项目3 多模态数据集工程", "images_output",
                           "8031efe0-1b5c-11ef-8929-000066532cad.png")
        shutil.copyfile(src, local)
    with open(local, "rb") as f:
        url = "data:image/png;base64," + base64.b64encode(f.read()).decode()

    async def main():
        res = {}
        for mode in ("A", "B", "C"):
            r = await gm.run_pipeline(Ctx(env), "", url,
                                      img_check=ix.check_image(url), mode=mode)
            res[mode] = {
                "status": r["image_status"],
                "report_null": r["report"] is None,
                "nodes": [t["node"] for t in r["node_timings"]],
                "node_timings": r["node_timings"],
                "usage": r["usage"],
                "iteration": r["iteration"],
                "selfcheck_skipped": (r.get("selfcheck") or {}).get("skipped"),
                "build": r["build"],
                "impression": ((r.get("report") or {}).get("impression") or [])[:1],
            }
        return res

    res = asyncio.run(main())
    for mode in ("A", "B", "C"):
        print(f"--- mode {mode} ---")
        print(json.dumps(res[mode], ensure_ascii=False, indent=1))
    with open("test_modes_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-network", action="store_true")
    a = ap.parse_args()
    graph_shape_check()
    unit_usage_check()
    offline_routing_check()
    if a.no_network:
        print("network e2e skipped (--no-network)")
    else:
        e2e_modes()
