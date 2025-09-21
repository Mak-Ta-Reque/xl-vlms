#!/usr/bin/env python3
"""
Compare two (or more) runs by aggregating their eval curves into a single curve per run
and overlaying them for deletion and insertion (token mode).

Aggregation per run:
1) For each method dir under <run>/eval, collect c_*_token_rank*.csv files.
2) For each prefix (deletion/insertion), average across ranks with linear interpolation
   onto a common x-grid from the first available CSV in that method.
3) Average the resulting method mean-curves to produce one curve per run per prefix.

Saves overlays into outputs/plots by default.

Usage:
  python scripts/plot_runs_aggregate_comparison.py \
    --outputs_dir /path/to/outputs \
    --runs cdgl,dl \
    [--methods pca,simple,random,snmf] \
    [--out_dir /path/to/outputs/plots] \
    [--ymin 3.62e-6 --ymax 4.0e-6]
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
        if "fraction_zeroed" in header:
            fx = header.index("fraction_zeroed")
        elif "fraction_inserted" in header:
            fx = header.index("fraction_inserted")
        else:
            fx = 0
        my = header.index("mean_prob") if "mean_prob" in header else 1
        xs, ys = [], []
        for row in r:
            if not row: continue
            try:
                xs.append(float(row[fx])); ys.append(float(row[my]))
            except Exception:
                continue
    return np.asarray(xs, float), np.asarray(ys, float)


def _collect_by_rank(dir_path: Path, prefix: str) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for p in sorted(dir_path.glob(f"{prefix}_rank*.csv")):
        m = re.search(r"_rank(\d+)\.csv$", p.name)
        if not m: continue
        out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def _mean_curve_across_ranks(method_dir: Path, prefix: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    files = _collect_by_rank(method_dir, prefix)
    if not files:
        return None
    first = next(iter(files.values()))
    base_x, _ = _load_curve(first)
    if base_x.size == 0:
        return None
    acc = np.zeros_like(base_x)
    n = 0
    for _rk, fp in files.items():
        xs, ys = _load_curve(fp)
        if xs.size == 0:
            continue
        if xs.shape == base_x.shape and np.allclose(xs, base_x, rtol=1e-6, atol=1e-12):
            y_interp = ys
        else:
            try:
                y_interp = np.interp(base_x, xs, ys)
            except Exception:
                continue
        acc += y_interp
        n += 1
    if n == 0:
        return None
    return base_x, acc / float(n)


def _aggregate_run_curve(eval_dir: Path, methods: Optional[Sequence[str]], prefix: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if methods:
        mdirs = [eval_dir / m for m in methods]
    else:
        mdirs = [p for p in eval_dir.iterdir() if p.is_dir() and p.name != "plots"]
    mcurves: List[Tuple[np.ndarray, np.ndarray]] = []
    for mdir in sorted(mdirs):
        cur = _mean_curve_across_ranks(mdir, prefix=prefix)
        if cur is not None:
            mcurves.append(cur)
    if not mcurves:
        return None
    # Choose base grid from first curve
    base_x = mcurves[0][0]
    acc = np.zeros_like(base_x)
    n = 0
    for xs, ys in mcurves:
        if xs.shape == base_x.shape and np.allclose(xs, base_x, rtol=1e-6, atol=1e-12):
            y_interp = ys
        else:
            try:
                y_interp = np.interp(base_x, xs, ys)
            except Exception:
                continue
        acc += y_interp
        n += 1
    if n == 0:
        return None
    return base_x, acc / float(n)


def _plot_overlay(out_dir: Path, curves: Dict[str, Tuple[np.ndarray, np.ndarray]], title: str, stem: str, ymin: Optional[float], ymax: Optional[float]) -> None:
    if not curves:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.5))
    for run_name, (xs, ys) in sorted(curves.items()):
        plt.plot(xs, ys, label=run_name)
    plt.title(title)
    plt.xlabel("# of Concept")
    plt.ylabel("f(x)")
    # Auto y-limits if not provided
    if ymin is None or ymax is None:
        try:
            all_y = np.concatenate([ys for (_rn, (_xs, ys)) in curves.items() if len(ys) > 0])
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
            plt.ylim(ymin, ymax)
        except Exception:
            pass
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{int(round(x * 100))}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _pos: f"{y * 1e4:.3f}"))
    plt.grid(True, alpha=0.3)
    plt.legend(title="Run")
    plt.tight_layout()
    (out_dir / f"{stem}.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig((out_dir / f"{stem}.png").as_posix(), dpi=160)
    plt.savefig((out_dir / f"{stem}.pdf").as_posix(), dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate and compare runs (cgdl vs dl) by overlaying their mean curves across methods.")
    ap.add_argument("--outputs_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "outputs"))
    ap.add_argument("--runs", type=str, default="cdgl,dl", help="Comma-separated tokens to match run directories")
    ap.add_argument("--methods", type=str, default="", help="Comma-separated method names to include (default: auto-detect)")
    ap.add_argument("--out_dir", type=str, default="", help="Directory to save plots (default: <outputs_dir>/plots)")
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (outputs_dir / "plots")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None

    tokens = [t.strip() for t in args.runs.split(",") if t.strip()]
    run_dirs = []
    for p in outputs_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if any(t in name for t in tokens):
            run_dirs.append(p)
    run_dirs = sorted(run_dirs)

    deletion_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    insertion_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for rd in run_dirs:
        eval_dir = rd / "eval"
        if not eval_dir.is_dir():
            continue
        del_curve = _aggregate_run_curve(eval_dir, methods, prefix="c_deletion_token")
        ins_curve = _aggregate_run_curve(eval_dir, methods, prefix="c_insertion_token")
        rname = rd.name
        if del_curve is not None:
            deletion_curves[rname] = del_curve
        if ins_curve is not None:
            insertion_curves[rname] = ins_curve

    _plot_overlay(out_dir, deletion_curves, title="C-Deletion (runs aggregated)", stem="runs_c_deletion_token_aggregated", ymin=args.ymin, ymax=args.ymax)
    _plot_overlay(out_dir, insertion_curves, title="C-Insertion (runs aggregated)", stem="runs_c_insertion_token_aggregated", ymin=args.ymin, ymax=args.ymax)


if __name__ == "__main__":
    main()
