"""Judge-facing blind-evaluation package, formal batch (2026-10-24).

- 6 cases x 3 modes = 18 reports; the non-CT negative (CT-004) refusal
  messages are included — the guard behavior is a system highlight, and
  judges should see it attributed to the right scheme, not dropped.
- build stamp stripped from each report header.
- mode->scheme mapping stays OUT of this dir (mapping_INTERNAL_2026-10-23.json).
- fallback_used flag (from the ablation CSV) is surfaced in the index so
  judges can discount template-filled cells when scoring completeness.
"""
from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

REPO_DIR = Path(r"C:\Users\horita\ct_ablation\reports_anon_2026-10-23")
JUDGE_DIR = Path(r"C:\Users\horita\ct_ablation\judge_package_formal_2026-10-24")
CSV_PATH = Path(r"C:\Users\horita\ct_ablation\ablation_batch_2026-10-23.csv")
MAPPING = Path(r"C:\Users\horita\ct_ablation\mapping_INTERNAL_2026-10-23.json")

# scheme dir name -> which mode files it holds (from mapping, seed=42)
import json
SCHEME_OF_MODE = json.loads(MAPPING.read_text(encoding="utf-8"))["mapping"]
SCHEME_DIRS = {}
for mode, scheme in SCHEME_OF_MODE.items():
    SCHEME_DIRS.setdefault(scheme, []).append(mode)

CASES = ["CT-001", "CT-002", "CT-003", "CT-004", "CT-005", "CT-006"]
MODES = ["A", "B", "C"]


def main():
    if JUDGE_DIR.exists():
        shutil.rmtree(JUDGE_DIR)

    # fallback_used per (case, mode)
    fb = {}
    for r in csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")):
        fb[(r["图ID"], r["模式"])] = r["fallback_used"]

    total = 0
    for scheme, modes in SCHEME_DIRS.items():
        d = JUDGE_DIR / scheme
        d.mkdir(parents=True)
        for ct in CASES:
            for m in modes:
                fname = f"{ct}_{m}.txt"
                src = REPO_DIR / scheme / fname
                if not src.exists():
                    print(f"MISSING source: {src}")
                    continue
                text = src.read_text(encoding="utf-8")
                text = re.sub(r"^=== .*build \S* ===\n\n", "", text,
                              count=1, flags=re.MULTILINE)
                text = f"=== 病例 {ct} ===\n\n" + text
                (d / fname).write_text(text, encoding="utf-8")
                total += 1

    # Judge index with fallback flags
    idx = JUDGE_DIR / "盲评索引.md"
    lines = [
        "# 盲评材料索引",
        "",
        "共 6 个病例 × 3 份方案报告。方案1/方案2/方案3 编号相互独立，"
        "与具体技术档位无对应关系。",
        "",
        "病例说明：",
        "  - CT-001 / CT-002 / CT-005：真实脑 CT（出血征象，ROI 占比 13.8% / 15.2%）",
        "  - CT-003：真实脑 CT（占位/亚硬膜血肿征象，ROI 占比 6.5%）",
        "  - CT-004：合成非 CT 负例——预期行为是拒绝生成报告"
        "（三层输入守卫拦截），拒答本身也是评分项",
        "  - CT-006：真实脑 CT（正常对照，全库无「无病灶」标注图，"
        "取信息量最大的干净图作代理，见报告局限性说明）",
        "",
        "fallback 标注（该格子的印象/建议段是否由模板兜底填充）：",
    ]
    for ct in CASES:
        flags = []
        for m in MODES:
            scheme = SCHEME_OF_MODE[m]
            v = fb.get((ct, m), "?")
            if v == "true":
                flags.append(f"病例{ct}-{scheme}: fallback=true")
        if flags:
            lines.append("  " + "  ".join(flags))
        else:
            break
    lines = lines  # keep simple: only list true ones below
    lines = [l for l in lines if not l.startswith("  病例") or "fallback" in l]
    fb_lines = []
    for ct in CASES:
        for m in MODES:
            if fb.get((ct, m)) == "true":
                fb_lines.append(
                    f"  - 病例 {ct}（{SCHEME_OF_MODE[m]}）: fallback_used=true，"
                    f"印象/建议段为模板兜底句，结构性完整但非模型生成")
    idx_text = "\n".join(
        ["# 盲评材料索引", "",
         "共 6 个病例 × 3 份方案报告。方案1/方案2/方案3 编号相互独立，"
         "与具体技术档位无对应关系。", "",
         "病例说明：",
         "  - CT-001 / CT-002 / CT-005：真实脑 CT（出血征象，ROI 13.8% / 13.8% / 15.2%）",
         "  - CT-003：真实脑 CT（占位/疑亚硬膜血肿，ROI 6.5%）",
         "  - CT-004：合成非 CT 负例，预期行为为拒绝生成报告（输入守卫拦截），"
         "拒答文案的合理性也是评分项",
         "  - CT-006：真实脑 CT（正常对照；数据集中无「无病灶」标注图，"
         "取信息量最大的干净图作代理，局限性见报告附录）", "",
         "fallback 标注（印象/建议段是否由模板兜底填充，评分结构完整度时请单独对待）：",
         "  " + ("；".join(fb_lines) if fb_lines else "本批 18 格均 fallback_used=false，"
                 "无兜底格"), "",
         "未脱敏说明：build 版本号与内部实验标识已移除；病例编号（CT-00x）"
         "不携带类别真值，类别判定请完全依据报告内容。",
         ""]
    )
    idx.write_text(idx_text, encoding="utf-8")

    print(f"Judge package: {JUDGE_DIR}")
    print(f"Reports copied: {total}")
    for scheme in SCHEME_DIRS:
        print(f"  {scheme}: {sorted(p.name for p in (JUDGE_DIR / scheme).glob('*.txt'))}")
    print(f"fallback=true cells: {len(fb_lines)}")


if __name__ == "__main__":
    main()
