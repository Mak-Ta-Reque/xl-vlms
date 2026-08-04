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


def _load_y0_by_rank(out_dir: Path, prefix: str) -> Dict[int, float]:
    by_rank = _collect_by_rank(out_dir, prefix)
    y0s: Dict[int, float] = {}
    for rank, path in by_rank.items():
        _, ys = _load_curve(path)
        if ys:
            y0s[rank] = ys[0]
    return y0s


def _plot_all(
    out_dir: Path,
    prefix: str,
    title: str,
    xlabel: str,
    outfile_stem: str,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    baseline_by_rank: Optional[Dict[int, float]] = None,
) -> None:
    by_rank = _collect_by_rank(out_dir, prefix)
    if not by_rank:
        return
    curves = {}
    for rank, path in by_rank.items():
        xs, ys = _load_curve(path)
        if xs:
            curves[rank] = (xs, ys)
    if not curves:
        return
    # Rank-1's concept is, by construction, the most strongly activating one
    # for a given token -- so its curve starts from a genuinely higher
    # baseline probability (fraction=0, nothing zeroed yet) than rank-2/3's.
    # Pooling one shared min/max across ranks (the previous approach) lets
    # that baseline difference dominate the shared axis, squashing rank-2/3
    # near the bottom regardless of their own (real, meaningful) decline --
    # and would do the same to any AUC computed on that shared scale, unfairly
    # penalizing a rank whose concept simply carries less absolute weight
    # rather than one that orders deletion less faithfully.
    #
    # Deletion's own fraction=0 point already IS the true, complete,
    # unmasked baseline (nothing zeroed yet) -- dividing by its own y[0]
    # correctly starts every rank at 1.0 and falls as that rank's
    # top-activating coordinates get zeroed.
    #
    # Insertion's own fraction=0 point is the OPPOSITE: a blank/zeroed input
    # (nothing inserted yet), which sits near the vocab floor -- dividing by
    # THAT (the old behavior) makes the curve start at 1.0 and climb well
    # above 1 as real signal gets recovered, which is mathematically
    # consistent but reads backwards (an insertion curve should start low
    # and rise toward the true baseline, not start at an arbitrary 1.0 and
    # overshoot it). Fixed: for insertion, use the SAME rank's deletion
    # curve's y[0] (the true baseline, from baseline_by_rank) as the
    # reference instead of insertion's own -- now insertion starts near the
    # blank/floor ratio and rises toward (but not much past) 1.0 as it
    # recovers the true baseline, which is what "insertion" should look like.
    plt.figure(figsize=(7.0, 4.5))
    for rank, (xs, ys) in curves.items():
        y0 = baseline_by_rank.get(rank) if baseline_by_rank else ys[0]
        if not y0:
            y0 = ys[0]
        ys_plot = [y / y0 for y in ys] if y0 else ys
        plt.plot(xs, ys_plot, label=f"Top-{rank}")
    # Titles and labels per request
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("fraction of own starting probability")
    # Show percentage numbers instead of fractions (no custom tick list)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{int(round(x * 100))}"))
    # Y-axis: default numeric formatting, 0-1 relative scale (no
    # scaling/exponent needed now that values are rescaled per curve)
    plt.grid(True, alpha=0.3)
    # Legend placement rules:
    # - Insertion plots: bottom-right
    # - Deletion plots: top-right
    # Also, remove legend title
    legend_loc = "best"
    if "insertion" in prefix:
        legend_loc = "lower right"
    elif "deletion" in prefix:
        legend_loc = "upper right"
    plt.legend(loc=legend_loc)
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

    # Insertion combined plot -- self-relative: each curve divided by its
    # own fraction=0 value, no cross-referencing another curve.
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
