"""Probe the LIVE deployment and print which LangGraph nodes were traversed.

Usage:
    python scripts/show_graph_nodes.py <image.png> [A|B|C]

Prints the real `nodes` list returned by /report, plus per-node timings,
iteration count and self-check verdict. Use this to demonstrate that the
five-node LangGraph state graph is genuinely executed, and that mode A/B/C
take different paths through it.
"""

import base64
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid

LIVE_URL = "https://ct.hyhagent.dpdns.org/report"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
NODE_LABELS = {
    "vision": "影像理解",
    "refuse": "拒答",
    "retrieval": "知识检索",
    "generate": "生成报告",
    "check": "一致性自检",
}


def data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    raw = open(path, "rb").read()
    return "data:" + mime + ";base64," + base64.b64encode(raw).decode()


def probe(path: str, mode: str) -> dict:
    payload = json.dumps({"mode": mode, "description": "请生成结构化报告",
                          "image": data_url(path)}, ensure_ascii=False).encode()
    req = urllib.request.Request(
        LIVE_URL, data=payload, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8",
                 "makers-conversation-id": uuid.uuid4().hex,
                 "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
        if isinstance(d, list):
            d = d[0]
        return d
    except urllib.error.HTTPError as e:
        return {"error": "HTTP " + str(e.code), "body": e.read().decode()[:300]}


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    img = sys.argv[1]
    if len(sys.argv) > 2:
        modes = [sys.argv[2].upper()]
    else:
        modes = ["A", "B", "C"]

    for m in modes:
        d = probe(img, m)
        print("=" * 62)
        print("档位", m, "| build", d.get("build"), "| image_status",
              d.get("image_status"))
        if "error" in d:
            print("  失败:", d["error"], d.get("body", ""))
            continue
        nodes = d.get("nodes", [])
        print("  执行路径:", " -> ".join(
            NODE_LABELS.get(n, n) + "(" + n + ")" for n in nodes) or "（空）")
        for t in d.get("node_timings", []):
            print("    %-8s %7d ms" % (NODE_LABELS.get(t["node"], t["node"]), t["ms"]))
        u = d.get("usage") or {}
        print("  迭代轮次:", d.get("iteration"), "| 模型调用:", u.get("model_calls"),
              "| token:", (u.get("totals") or {}).get("total_tokens"))
        sc = d.get("selfcheck") or {}
        if sc.get("skipped"):
            verdict = "跳过(A/B档)"
        elif sc.get("passed") is True:
            verdict = "通过"
        else:
            verdict = "未通过"
        print("  自检:", verdict, sc.get("issues") or "")
        rep = d.get("report") or {}
        print("  印象条数:", len(rep.get("impression") or []),
              "| 建议条数:", len(rep.get("recommendation") or []))


if __name__ == "__main__":
    main()
