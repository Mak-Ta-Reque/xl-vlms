#!/usr/bin/env python3
"""
Plot combined concept deletion/insertion curves (token mode) across ranks.

- Scans an output directory for CSVs produced by eval/concept_deletion_eval.py
  with filenames like:
    c_deletion_token_rank{R}.csv
    c_insertion_token_rank{R}.csv
- Draws all deletion curves on one plot (legend: rank)
- Draws all insertion curves on one plot (legend: rank)

Usage:
python scripts/plot_concept_deletion_eval_token.py --out_dir /path/to/outputs/concept_deletion_token
"""
from __future__ import annotations

import os
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_curve(csv_path: Path) -> Tuple[List[float], List[float]]:
    with csv_path.open("r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        # Identify columns
        try:
            frac_idx = header.index("fraction_zeroed") if "fraction_zeroed" in header else header.index("fraction_inserted")
            mean_idx = header.index("mean_prob")
        except ValueError:
            # Fallback: assume first two columns are fraction and mean
            frac_idx, mean_idx = 0, 1
        xs: List[float] = []
        ys: List[float] = []
        for row in r:
            if not row:
                continue
            try:
                xs.append(float(row[frac_idx]))
                ys.append(float(row[mean_idx]))
            except Exception:
                continue
    return xs, ys


def _collect_by_rank(out_dir: Path, prefix: str) -> Dict[int, Path]:
    # prefix e.g., "c_deletion_token" or "c_insertion_token"
    files = sorted(out_dir.glob(f"{prefix}_rank*.csv"))
    by_rank: Dict[int, Path] = {}
    for p in files:
        m = re.search(r"_rank(\d+)\.csv$", p.name)
        if not m:
            continue
        rank = int(m.group(1))
        by_rank[rank] = p
    return dict(sorted(by_rank.items(), key=lambda kv: kv[0]))


def _plot_all(out_dir: Path, prefix: str, title: str, xlabel: str, outfile: str) -> None:
    by_rank = _collect_by_rank(out_dir, prefix)
    if not by_rank:
        return
    plt.figure(figsize=(7.0, 4.5))
    for rank, path in by_rank.items():
        xs, ys = _load_curve(path)
        if not xs:
            continue
        plt.plot(xs, ys, label=f"rank {rank}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("softmax probability of target token (mean)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="rank")
    plt.tight_layout()
    out_path = out_dir / outfile
    plt.savefig(out_path.as_posix(), dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot combined deletion/insertion curves across ranks (token mode)")
    ap.add_argument("--out_dir", required=True, help="Directory containing CSV outputs from concept_deletion_eval.py")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deletion combined plot
    _plot_all(
        out_dir,
        prefix="c_deletion_token",
        title="Concept deletion (token): all ranks",
        xlabel="fraction of concept coordinates zeroed (most → least important)",
        outfile="c_deletion_token_all_ranks.png",
    )

    # Insertion combined plot
    _plot_all(
        out_dir,
        prefix="c_insertion_token",
        title="Concept insertion (token): all ranks",
        xlabel="fraction of concept coordinates inserted (most → least important)",
        outfile="c_insertion_token_all_ranks.png",
    )


if __name__ == "__main__":
    main()
