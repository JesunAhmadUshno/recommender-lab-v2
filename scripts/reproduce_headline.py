"""Regenerate results/headline_table.md from the committed derived data.

The headline table is a pure function of results/tables/arm_metrics.csv and
results/tables/sampled_metrics.csv. CI runs this with --check so the
published table can never drift from the committed numbers. When no derived
data is committed yet, the check is a no-op with a notice.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "results" / "headline_table.md"


def _read(name: str) -> list[dict]:
    path = TABLES / name
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def build_table() -> str | None:
    metrics = _read("arm_metrics.csv")
    sampled = _read("sampled_metrics.csv")
    if not metrics:
        return None
    sampled_mean: dict[tuple[str, str], tuple[float, float]] = defaultdict(lambda: ([], []))
    for row in sampled:
        key = (row["arm"], row["ranker"])
        sampled_mean[key][0].append(float(row["sampled_ndcg"]))
        sampled_mean[key][1].append(float(row["corrected_ndcg"]))
    lines = [
        "| Arm | Ranker | n users | Exact NDCG@10 | Sampled NDCG@10 (mean over seeds) | Corrected NDCG@10 (mean over seeds) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(metrics, key=lambda r: (r["arm"], -float(r["ndcg"]))):
        key = (row["arm"], row["ranker"])
        if key in sampled_mean:
            s = sum(sampled_mean[key][0]) / len(sampled_mean[key][0])
            c = sum(sampled_mean[key][1]) / len(sampled_mean[key][1])
            s_txt, c_txt = f"{s:.4f}", f"{c:.4f}"
        else:
            s_txt = c_txt = "n/a"
        lines.append(
            f"| {row['arm']} | {row['ranker']} | {row['n_users']} | "
            f"{float(row['ndcg']):.4f} | {s_txt} | {c_txt} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    table = build_table()
    if table is None:
        print("reproduce_headline: no derived data committed yet; nothing to check.")
        return 0
    if "--check" in sys.argv:
        current = OUT.read_text() if OUT.exists() else ""
        if current != table:
            print("reproduce_headline: headline_table.md is out of sync with the committed tables.")
            return 1
        print("reproduce_headline: headline table matches the committed derived data.")
        return 0
    OUT.write_text(table)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
