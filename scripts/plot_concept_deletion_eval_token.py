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
python scripts/plot_concept_deletion_eval_token.py --out_dir /path/to/outputs/concept_deletion_token [--ymin 3.62e-6 --ymax 4.0e-6]
"""
from __future__ import annotations

import os
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Set global font sizes to 24 across the figure
matplotlib.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'legend.title_fontsize': 24,
})


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


def _plot_all(
    out_dir: Path,
    prefix: str,
    title: str,
    xlabel: str,
    outfile_stem: str,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> None:
    by_rank = _collect_by_rank(out_dir, prefix)
    if not by_rank:
        return
    plt.figure(figsize=(7.0, 4.5))
    for rank, path in by_rank.items():
        xs, ys = _load_curve(path)
        if not xs:
            continue
        plt.plot(xs, ys, label=f"Top-{rank}")
    # Titles and labels per request
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("f(x)")
    # Enforce y-axis limits if provided
    if y_min is not None and y_max is not None:
        try:
            plt.ylim(y_min, y_max)
        except Exception:
            pass
    # Show percentage numbers instead of fractions (no custom tick list)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{int(round(x * 100))}"))
    # Multiply displayed y-axis values by 1e5 (labels only)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _pos: f"{y * 1e4:.3f}"))
    # Y-axis: default numeric formatting (no scaling/exponent)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Concepts")
    plt.tight_layout()
    # Save both PNG (for pipeline checks) and PDF
    png_path = out_dir / f"{outfile_stem}.png"
    pdf_path = out_dir / f"{outfile_stem}.pdf"
    plt.savefig(png_path.as_posix(), dpi=160)
    plt.savefig(pdf_path.as_posix(), dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot combined deletion/insertion curves across ranks (token mode)")
    ap.add_argument("--out_dir", required=True, help="Directory containing CSV outputs from concept_deletion_eval.py")
    ap.add_argument("--ymin", type=float, default=3.62e-6, help="Lower y-axis limit (default: 3.62e-6)")
    ap.add_argument("--ymax", type=float, default=4.0e-6, help="Upper y-axis limit (default: 4.0e-6)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deletion combined plot
    _plot_all(
        out_dir,
        prefix="c_deletion_token",
        title="C-Deletion",
        xlabel="# of Concept",
        outfile_stem="c_deletion_token_all_ranks",
        y_min=args.ymin,
        y_max=args.ymax,
    )

    # Insertion combined plot
    _plot_all(
        out_dir,
        prefix="c_insertion_token",
        title="C-Insertion",
        xlabel="# of Concept",
        outfile_stem="c_insertion_token_all_ranks",
        y_min=args.ymin,
        y_max=args.ymax,
    )


if __name__ == "__main__":
    main()
