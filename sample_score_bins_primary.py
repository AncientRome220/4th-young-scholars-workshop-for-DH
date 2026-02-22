#!/usr/bin/env python3
"""
Create an annotation sample from linked extraction rows using score bins.

Default behavior follows the project discussion:
- Primary rows only (Is_Primary == yes)
- Score bins: 0-10, 11-20, ..., 71-80
- Equal ambiguous/non-ambiguous sampling per bin when possible
- CSV output with annotation-ready columns

Example:
  python sample_score_bins_primary.py \
    --input linked_all_scores_4042.csv \
    --output score_bin_sample_primary_n10.csv \
    --per-bin 10 \
    --seed 42
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample primary rows by score bins for annotation")
    p.add_argument("--input", default="linked_all_scores_4042.csv", help="Input linked CSV")
    p.add_argument("--output", default="score_bin_sample_primary_n10.csv", help="Output sample CSV")
    p.add_argument("--per-bin", type=int, default=10, help="Target sample size per score bin")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--max-score", type=int, default=80, help="Maximum score range endpoint")
    p.add_argument(
        "--allow-fill",
        action="store_true",
        help=(
            "If one ambiguity group has too few rows in a bin, fill remaining slots from the other group."
        ),
    )
    return p.parse_args()


def score_bin(score: float, max_score: int) -> str | None:
    if not math.isfinite(score) or score < 0 or score > max_score:
        return None
    if score <= 10:
        return "00-10"
    lo = int((score - 1) // 10) * 10 + 1
    hi = lo + 9
    if hi > max_score:
        hi = max_score
    return f"{lo:02d}-{hi:02d}"


def ambiguous_group(value: str) -> str:
    return "ambiguous_yes" if str(value).strip().lower() == "yes" else "ambiguous_no"


def build_bins(max_score: int) -> List[str]:
    bins = ["00-10"]
    i = 11
    while i <= max_score:
        hi = min(i + 9, max_score)
        bins.append(f"{i:02d}-{hi:02d}")
        i += 10
    return bins


def main() -> None:
    args = parse_args()
    rnd = random.Random(args.seed)

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path.resolve()}")

    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    total = 0
    primary = 0

    with in_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if str(row.get("Is_Primary", "")).strip().lower() != "yes":
                continue
            primary += 1

            try:
                score = float(str(row.get("Score", "")).strip())
            except Exception:
                continue

            b = score_bin(score, args.max_score)
            if b is None:
                continue

            g = ambiguous_group(row.get("Ambiguous", ""))
            row["_Sample_Bin"] = b
            row["_Sample_Group"] = g
            grouped[(b, g)].append(row)

    bins = build_bins(args.max_score)
    sampled_rows: List[dict] = []

    print(f"Loaded rows: {total}")
    print(f"Primary rows: {primary}")
    print(f"Target per bin: {args.per_bin} (equal split by Ambiguous yes/no)")

    half = args.per_bin // 2
    for b in bins:
        yes_rows = grouped.get((b, "ambiguous_yes"), [])
        no_rows = grouped.get((b, "ambiguous_no"), [])

        take_yes = min(half, len(yes_rows))
        take_no = min(args.per_bin - half, len(no_rows))

        picked_yes = rnd.sample(yes_rows, take_yes) if take_yes > 0 else []
        picked_no = rnd.sample(no_rows, take_no) if take_no > 0 else []

        picked = picked_yes + picked_no

        if args.allow_fill and len(picked) < args.per_bin:
            remaining = args.per_bin - len(picked)
            used_ids = {id(x) for x in picked}
            pool = [r for r in (yes_rows + no_rows) if id(r) not in used_ids]
            fill_n = min(remaining, len(pool))
            if fill_n > 0:
                picked.extend(rnd.sample(pool, fill_n))

        sampled_rows.extend(picked)
        print(
            f"Bin {b}: yes={len(yes_rows)} no={len(no_rows)} sampled={len(picked)}"
            f" (yes={len(picked_yes)}, no={len(picked_no)})"
        )

    sampled_rows.sort(key=lambda r: (r["_Sample_Bin"], float(r.get("Score", 0) or 0), r.get("DDB_ID", "")))

    if not sampled_rows:
        raise RuntimeError("No rows were sampled. Check score ranges and primary rows.")

    base_fields = list(sampled_rows[0].keys())
    for hidden in ["_Sample_Bin", "_Sample_Group"]:
        if hidden in base_fields:
            base_fields.remove(hidden)

    out_fields = [
        "Sample_Bin",
        "Sample_Group",
        *base_fields,
        "Human_Label",
        "Annotation_Notes",
    ]

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for r in sampled_rows:
            out = {
                "Sample_Bin": r.get("_Sample_Bin", ""),
                "Sample_Group": r.get("_Sample_Group", ""),
                "Human_Label": "",
                "Annotation_Notes": "",
            }
            for k in base_fields:
                out[k] = r.get(k, "")
            writer.writerow(out)

    print(f"Saved sample: {out_path} (rows={len(sampled_rows)})")


if __name__ == "__main__":
    main()
