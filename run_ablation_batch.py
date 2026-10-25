"""Ablation batch: 6 real CTs × modes A/B/C against the LIVE endpoint.

Pipeline:
  1. Pick 6 images from image_caption.csv (1 per category: 正常/出血/梗死/
     水肿/占位/骨折) using caption keywords, verify file exists in
     E:\\项目3 多模态数据集工程\\images_output\\
  2. For each image × mode: call the live /report endpoint with a unique
     makers-conversation-id, record latency/tokens/iteration/fallback
  3. Write CSV + anonymized report files (方案1/2/3, shuffled per-mode)
     + internal mapping file (NOT in the report dir)
  4. Hard budget: 250,000 tokens — stop on overshoot

Usage:
    python -I run_ablation_batch.py            # full 18-call run
    python -I run_ablation_batch.py --dry-run  # pick + plan only
    python -I run_ablation_batch.py --limit 6  # stop after N calls
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_DIR = r"E:\项目3 多模态数据集工程"
IMAGES_DIR = os.path.join(DATASET_DIR, "images_output")
CAPTIONS_CSV = os.path.join(DATASET_DIR, "image_ caption.csv")
LIVE_URL = "https://ct.hyhagent.dpdns.org/report"

CATEGORIES = ["正常", "出血", "梗死", "水肿", "占位", "骨折"]
# Keywords in the caption text that signal each category
CAT_KEYWORDS = {
    "出血": ["hemorrhage", "hematoma", "bleeding", "hyperdensit", "hyperdens"],
    "梗死": ["infarct", "ischemi", "hypodens", "low-dens"],
    "水肿": ["edema", "swelling", "mass effect"],
    "占位": ["mass", "lesion", "tumor", "tumour", "occupying", "neoplasm"],
    "骨折": ["fracture", "calvarial", "skull fracture"],
}
# "正常" = caption contains none of the pathological keywords above

MODES = ["A", "B", "C"]
ANON_NAMES = ["方案1", "方案2", "方案3"]
SEED = 42
PER_CATEGORY = 1
HARD_BUDGET = 250_000
PER_CALL_TIMEOUT = 180

ABLAB_DIR = Path("C:/Users/horita/ct_ablation")
REPO_DIR = Path("C:/Users/horita/ct_ablation/reports_anon_2026-10-23")
CSV_PATH = ABLAB_DIR / "ablation_batch_2026-10-23.csv"
MAPPING_INTERNAL = ABLAB_DIR / "mapping_INTERNAL_2026-10-23.json"


# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------
def pick_images() -> list[tuple[str, str, str]]:
    """Fixed final test set, per 2026-10-24 user decision:
    - 出血 x3 (ROI>=5%, >=100KB, 512x512, from formal2_picks.json candidates)
    - 占位 x1 (only 1 eligible in corpus after all filters)
    - 非CT负例 x1 (synthetic, expected image_status=invalid)
    - 正常 x1 (highest-file-size caption-clean case, as proxy since
      "正常" pool is empty under ROI>=5% filter — documented in
      ct_ablation/formal2_picks.json)
    See ct_ablation/formal2_picks.json for full provenance."""
    DATASET = r"E:\项目3 多模态数据集工程\images_output"
    NEG = r"C:\Users\horita\ct_vlm_compare\images\ct_01_hemorrhage_512.png"
    fixed = [
        ("CT-001", "804af205-1b5c-11ef-b715-000066532cad.png", "出血", DATASET),
        ("CT-002", "804af878-1b5c-11ef-8834-000066532cad.png", "出血", DATASET),
        ("CT-003", "804be0c2-1b5c-11ef-a3db-000066532cad.png", "占位", DATASET),
        ("CT-004", "ct_01_hemorrhage_512.png", "非CT负例",
         os.path.dirname(NEG)),
        ("CT-005", "80317ea0-1b5c-11ef-936e-000066532cad.png", "出血", DATASET),
        ("CT-006", "8030afc8-1b5c-11ef-8830-000066532cad.png", "正常", DATASET),
    ]
    out = []
    for ct_id, fname, label, base in fixed:
        p = os.path.join(base, fname)
        if not os.path.isfile(p):
            print(f"  MISSING: {ct_id} {fname}")
            continue
        out.append((ct_id, p, label))
    print(f"Using fixed final test set: {len(out)}/6 images")
    for ct_id, p, label in out:
        print(f"  {ct_id} [{label}]: {os.path.basename(p)}  "
              f"({os.path.getsize(p)} bytes)")
    return out


# ---------------------------------------------------------------------------
# Anon scheme
# ---------------------------------------------------------------------------
def make_scheme_mapping() -> dict:
    """Shuffle mode->anon-name. Deterministic (seed=42)."""
    rng = random.Random(SEED)
    perm = list(range(3))
    rng.shuffle(perm)
    return {MODES[i]: ANON_NAMES[perm[i]] for i in range(3)}


def load_image_data_url(path: str) -> str:
    raw = open(path, "rb").read()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def call_report(image_data_url: str | None, description: str, mode: str,
                conv_id: str, timeout: int = PER_CALL_TIMEOUT) -> dict:
    body = {"mode": mode, "description": description}
    if image_data_url:
        body["image"] = image_data_url
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        LIVE_URL,
        data=payload,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "makers-conversation-id": conv_id,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            d = json.loads(resp.read())
            d["_latency_s"] = round(time.perf_counter() - t0, 3)
            return d
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return {
            "_error": f"HTTP {e.code}",
            "_body": err_body[:500],
            "_latency_s": round(time.perf_counter() - t0, 3),
            "build": None, "usage": {"model_calls": 0,
                                     "totals": {"total_tokens": 0}},
        }
    except Exception as e:
        return {
            "_error": str(e)[:200],
            "_latency_s": round(time.perf_counter() - t0, 3),
            "build": None, "usage": {"model_calls": 0,
                                     "totals": {"total_tokens": 0}},
        }


def load_provenance(ct_id: str, path: str) -> dict:
    """Return {uuid, roi_pct, caption_snippet} for a picked image, from
    formal2_picks.json + image_ caption.csv. Empty values for the synthetic
    negative (CT-004, not in the corpus)."""
    picks = json.load(open(r"C:\Users\horita\ct_ablation\formal2_picks.json",
                           encoding="utf-8"))
    entry = picks.get(ct_id, {})
    fname = os.path.basename(path)
    uuid = fname.rsplit(".", 1)[0] if entry else fname.rsplit(".", 1)[0]
    if not entry:
        return {"uuid": fname, "roi_pct": "", "caption_snippet": "(synthetic negative, not in corpus)"}
    cap = entry.get("caption", "")
    return {"uuid": uuid, "roi_pct": entry.get("roi_pct", ""),
            "caption_snippet": cap[:80]}


def extract_metrics(resp: dict) -> dict:
    usage = resp.get("usage") or {}
    totals = usage.get("totals") or {}
    per_node = usage.get("per_node") or []
    token_total = totals.get("total_tokens", 0)
    gen_calls = sum(1 for n in per_node if n.get("node") == "generate")
    check_calls = sum(1 for n in per_node if n.get("node") == "check")
    fallback = (check_calls >= 1 and gen_calls >= 2)
    return {
        "latency_s": resp.get("_latency_s", 0),
        "token_total": token_total,
        "iterations": resp.get("iteration", 0),
        "fallback_triggered": fallback,
        "fallback_used": bool(resp.get("fallback_used", False)),
        "model_calls": usage.get("model_calls", 0),
        "image_status": resp.get("image_status", ""),
        "build": resp.get("build", ""),
    }


# ---------------------------------------------------------------------------
# Report export (anonymized)
# ---------------------------------------------------------------------------
def write_report_file(anon_dir: Path, ct_id: str, mode: str,
                      resp: dict, anon_name: str) -> str:
    sub = anon_dir / anon_name
    sub.mkdir(parents=True, exist_ok=True)
    fname = f"{ct_id}_{mode}.txt"
    report = resp.get("report")
    if report:
        text = (
            f"=== {ct_id} | build {resp.get('build','')} ===\n\n"
            f"【影像所见】\n{report.get('findings', '')}\n\n"
            f"【印象】\n" + "\n".join(
                f"  {i+1}. {x}" for i, x in enumerate(report.get("impression", []))
            ) + "\n\n"
            f"【建议】\n" + "\n".join(
                f"  {i+1}. {x}" for i, x in enumerate(report.get("recommendation", []))
            ) + "\n"
        )
    elif resp.get("refuse_reason"):
        text = f"[拒绝] {resp.get('refuse_reason')}\n"
    else:
        text = (f"[无报告] image_status={resp.get('image_status')} "
                f"error={resp.get('_error', resp.get('error',''))}\n")
    (sub / fname).write_text(text, encoding="utf-8")
    return str((sub / fname).relative_to(REPO_DIR.parent))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def print_stats(csv_path: Path, scheme: dict):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    by_mode: dict[str, list] = defaultdict(list)
    for r in rows:
        by_mode[r["模式"]].append(r)
    print("\n=== Per-mode statistics ===")
    for m in MODES:
        rs = by_mode.get(m, [])
        if not rs:
            print(f"mode {m} ({scheme[m]}): no data")
            continue
        lat = [float(r["延迟秒"]) for r in rs if r["延迟秒"]]
        tok = [int(r["token总量"]) for r in rs if r["token总量"]]
        fb = sum(1 for r in rs if r["是否触发回退"] == "是")
        errs = sum(1 for r in rs if r["错误"])
        print(f"mode {m} ({scheme[m]}): n={len(rs)}  "
              f"avg_lat={sum(lat)/len(lat):.2f}s  "
              f"avg_tok={sum(tok)/len(tok):.0f}  "
              f"fallback={fb}/{len(rs)}  errors={errs}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # 1. Pick images
    print("=== Step 1: picking images ===")
    picked = pick_images()
    if not picked:
        print("No images picked — aborting")
        sys.exit(1)
    print(f"\nSelected {len(picked)} images:")
    for ct_id, path, cat in picked:
        size = os.path.getsize(path)
        print(f"  {ct_id} [{cat}]: {os.path.basename(path)}  ({size} bytes)")

    scheme = make_scheme_mapping()
    print(f"\nMode -> Anon scheme: {scheme}")

    # Build work list
    work = []
    for ct_id, path, cat in picked:
        data_url = load_image_data_url(path)
        for mode in MODES:
            work.append((ct_id, path, cat, mode, data_url, scheme[mode]))

    print(f"\nTotal calls planned: {len(work)}")

    if args.dry_run:
        for i, (ct_id, _, cat, mode, _, anon) in enumerate(work, 1):
            print(f"  {i:2d}. {ct_id} [{cat}] mode={mode} -> {anon}")
        print("\nDry run complete. No API calls made.")
        return

    # 2. Save mapping
    ABLAB_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_INTERNAL.write_text(
        json.dumps({
            "description": "Internal: ablation mode -> anon scheme name. Do NOT share with judges.",
            "seed": SEED, "mapping": scheme,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "endpoint": LIVE_URL,
            "images": [{"ct_id": c, "path": p, "category": cat}
                       for c, p, cat, *_ in work],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Internal mapping: {MAPPING_INTERNAL}")

    # 3. Run batch
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    csv_f = open(CSV_PATH, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(csv_f)
    writer.writerow([
        "图ID", "图像类型", "模式", "方案编号",
        "uuid", "roi_pct", "caption_snippet",
        "延迟秒", "token总量", "迭代轮次", "是否触发回退",
        "model_calls", "image_status", "fallback_used", "build",
        "报告文件", "错误",
    ])
    # CT-004 is the synthetic non-CT negative; it has no corpus row.
    # The other 5 load from ct_ablation/formal2_picks.json (written separately).
    csv_f.flush()

    total_tokens = 0
    budget_exceeded = False
    calls_made = 0

    for i, (ct_id, path, cat, mode, data_url, anon) in enumerate(work, 1):
        if budget_exceeded:
            print(f"\n[BUDGET EXCEEDED] Stopping. Total: {total_tokens} tokens.")
            break
        if args.limit and calls_made >= args.limit:
            print(f"\n[CALL LIMIT] Stopping (--limit {args.limit}).")
            break

        conv_id = f"abl-{ct_id}-{mode}-{int(time.time())}"
        print(f"[{i:2d}/{len(work)}] {ct_id} [{cat}] mode={mode} ",
              end="", flush=True)

        resp = call_report(data_url, "（图像模式：所见由 VLM 读图生成）",
                           mode, conv_id)
        metrics = extract_metrics(resp)
        total_tokens += metrics["token_total"]
        calls_made += 1
        prov = load_provenance(ct_id, path)

        rel_path = write_report_file(REPO_DIR, ct_id, mode, resp, anon)
        err = resp.get("_error", resp.get("error", ""))

        print(f"lat={metrics['latency_s']}s  tok={metrics['token_total']}  "
              f"iter={metrics['iterations']}  "
              f"fb_flag={'true' if metrics.get('fallback_used') else 'false'}  "
              f"{'ERR:'+err[:60] if err else 'ok'}")

        writer.writerow([
            ct_id, cat, mode, anon,
            prov["uuid"], prov["roi_pct"], prov["caption_snippet"],
            metrics["latency_s"], metrics["token_total"],
            metrics["iterations"],
            "是" if metrics["fallback_triggered"] else "否",
            metrics["model_calls"], metrics["image_status"],
            "true" if metrics.get("fallback_used") else "false",
            metrics["build"], rel_path,
            err[:200] if err else "",
        ])
        csv_f.flush()

        if total_tokens > HARD_BUDGET:
            budget_exceeded = True
            print(f"  [!] Budget {HARD_BUDGET} exceeded ({total_tokens}) — halting.")

    csv_f.close()
    print(f"\nCSV: {CSV_PATH}")
    print(f"Reports: {REPO_DIR}")
    print(f"Tokens: {total_tokens} / {HARD_BUDGET}  Calls: {calls_made}")
    print_stats(CSV_PATH, scheme)


if __name__ == "__main__":
    main()
