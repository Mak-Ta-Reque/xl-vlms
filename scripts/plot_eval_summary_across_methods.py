#!/usr/bin/env python3
"""
Aggregate concept deletion/insertion (token) curves across methods.

What it does
- Looks under an eval root for method subfolders (e.g., pca, simple, random, snmf).
- For each method, loads all CSVs of the form:
    c_deletion_token_rank{R}.csv
    c_insertion_token_rank{R}.csv
  and computes the mean curve across ranks (per prefix) within that method.
- Plots two summary figures where each method is a legend entry:
    1) c_deletion_token mean curve by method
    2) c_insertion_token mean curve by method

Outputs
- Saves PNG and PDF in <output_dir>/ (defaults to <eval_root>/../plots)

Usage
python scripts/plot_eval_summary_across_methods.py \
    --eval_dir /path/to/outputs/.../eval [--methods pca,simple,random,snmf] \
    [--out_dir /path/to/outputs/.../plots] \
    [--ymin 3.62e-6 --ymax 4.0e-6]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Consistent font sizing with existing plots
matplotlib.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'legend.title_fontsize': 24,
})


def _load_curve(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with csv_path.open("r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        try:
            # token deletion uses fraction_zeroed; insertion uses fraction_inserted
            if "fraction_zeroed" in header:
                frac_idx = header.index("fraction_zeroed")
            elif "fraction_inserted" in header:
                frac_idx = header.index("fraction_inserted")
            else:
                frac_idx = 0
            mean_idx = header.index("mean_prob") if "mean_prob" in header else 1
        except ValueError:
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
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _collect_by_rank(dir_path: Path, prefix: str) -> Dict[int, Path]:
    files = sorted(dir_path.glob(f"{prefix}_rank*.csv"))
    by_rank: Dict[int, Path] = {}
    for p in files:
        m = re.search(r"_rank(\d+)\.csv$", p.name)
        if not m:
            continue
        by_rank[int(m.group(1))] = p
    return dict(sorted(by_rank.items(), key=lambda kv: kv[0]))


def _mean_curve_across_ranks(method_dir: Path, prefix: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (xs, mean_ys) averaged across all ranks for given prefix in method_dir.

    If x-grids differ between ranks, curves are linearly interpolated onto a common grid
    derived from the first available CSV.
    """
    by_rank = _collect_by_rank(method_dir, prefix)
    if not by_rank:
        return None

    # Choose base grid from the smallest rank file
    first_path = next(iter(by_rank.values()))
    base_xs, _ = _load_curve(first_path)
    if base_xs.size == 0:
        return None

    acc = np.zeros_like(base_xs, dtype=float)
    count = 0
    for _rank, p in by_rank.items():
        xs, ys = _load_curve(p)
        if xs.size == 0:
            continue
        # If shapes/values match, fast path
        if xs.shape == base_xs.shape and np.allclose(xs, base_xs, atol=1e-12, rtol=1e-6):
            interp = ys
        else:
            # Interpolate onto base grid
            try:
                interp = np.interp(base_xs, xs, ys)
            except Exception:
                # Fallback: skip if cannot interpolate
                continue
        acc += interp
        count += 1

    if count == 0:
        return None
    mean_ys = acc / float(count)
    return base_xs, mean_ys


def _plot_by_method(
    out_dir: Path,
    data_by_method: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    xlabel: str,
    outfile_stem: str,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> None:
    if not data_by_method:
        return
    plt.figure(figsize=(7.0, 4.5))
    for method, (xs, ys) in sorted(data_by_method.items()):
        if xs.size == 0:
            continue
        plt.plot(xs, ys, label=method)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("f(x)")
    # Auto y-limits if not provided
    if y_min is None or y_max is None:
        try:
            all_y = np.concatenate([ys for (_m, (_xs, ys)) in data_by_method.items() if len(ys) > 0])
            if all_y.size > 0:
                y0, y1 = float(np.nanmin(all_y)), float(np.nanmax(all_y))
                if y0 == y1:
                    pad = y0 * 0.05 if y0 != 0 else 1e-12
                    y0, y1 = y0 - pad, y1 + pad
                plt.ylim(y0, y1)
        except Exception:
            pass
    else:
        try:
            plt.ylim(y_min, y_max)
        except Exception:
            pass
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{int(round(x * 100))}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _pos: f"{y * 1e4:.3f}"))
    plt.grid(True, alpha=0.3)
    plt.legend(title="Method")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{outfile_stem}.png"
    pdf_path = out_dir / f"{outfile_stem}.pdf"
    plt.savefig(png_path.as_posix(), dpi=160)
    plt.savefig(pdf_path.as_posix(), dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot mean token curves across ranks for each method, overlaid by method.")
    ap.add_argument("--eval_dir", required=True, help="Eval root containing method subfolders (e.g., pca, simple, random, snmf)")
    ap.add_argument("--methods", type=str, default="", help="Comma-separated method names to include (default: auto-detect)")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory for summary plots (default: <eval_dir>/../plots)")
    ap.add_argument("--ymin", type=float, default=3.62e-6, help="Lower y-axis limit")
    ap.add_argument("--ymax", type=float, default=4.0e-6, help="Upper y-axis limit")
    args = ap.parse_args()

    eval_root = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (eval_root.parent / "plots")

    # Discover methods
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        methods = [p.name for p in eval_root.iterdir() if p.is_dir() and p.name != "plots"]
    methods = sorted(methods)

    # Collect mean curves per method
    deletion_by_method: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    insertion_by_method: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for m in methods:
        mdir = eval_root / m
        if not mdir.is_dir():
            continue
        del_curve = _mean_curve_across_ranks(mdir, prefix="c_deletion_token")
        ins_curve = _mean_curve_across_ranks(mdir, prefix="c_insertion_token")
        if del_curve is not None:
            deletion_by_method[m] = del_curve
        if ins_curve is not None:
            insertion_by_method[m] = ins_curve

    # Plot method overlays
    _plot_by_method(
        out_dir,
        deletion_by_method,
        title="C-Deletion (mean across ranks)",
        xlabel="# of Concept",
        outfile_stem="c_deletion_token_methods_mean",
        y_min=args.ymin,
        y_max=args.ymax,
    )

    _plot_by_method(
        out_dir,
        insertion_by_method,
        title="C-Insertion (mean across ranks)",
        xlabel="# of Concept",
        outfile_stem="c_insertion_token_methods_mean",
        y_min=args.ymin,
        y_max=args.ymax,
    )


if __name__ == "__main__":
    main()
