#!/usr/bin/env python3
"""
Plot simple grounding metrics similar to token concept plots.

Reads eval/grounding_eval.py outputs and draws:
- objects curve: fraction_kept vs mean_objects
- bar chart: top-K object name counts (optional)
"""
from __future__ import annotations

import os
import csv
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_curve(csv_path: Path) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    with csv_path.open("r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row:
                continue
            try:
                xs.append(float(row[0]))
                ys.append(float(row[1]))
            except Exception:
                continue
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot grounding evaluation outputs")
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--topk", type=int, default=15)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    curve_csv = eval_dir / "grounding_curve.csv"
    agg_json = eval_dir / "aggregate.json"

    eval_dir.mkdir(parents=True, exist_ok=True)

    if curve_csv.exists():
        xs, ys = _load_curve(curve_csv)
        if xs:
            plt.figure(figsize=(7.0, 4.5))
            plt.plot(xs, ys, label="mean #objects")
            plt.title("Grounding Curve")
            plt.xlabel("fraction kept (%)")
            plt.ylabel("mean objects per image")
            # percentage on x-axis
            from matplotlib.ticker import FuncFormatter
            ax = plt.gca()
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{int(round(x*100))}"))
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            out_path = eval_dir / "grounding_curve.png"
            plt.savefig(out_path.as_posix(), dpi=160)
            plt.close()

    if agg_json.exists():
        try:
            agg = json.loads(agg_json.read_text())
            name_counts = agg.get("name_counts") or {}
            if name_counts:
                items = list(name_counts.items())[:int(args.topk)]
                names = [k for k, _ in items]
                counts = [v for _, v in items]
                plt.figure(figsize=(10, 5))
                plt.bar(names, counts)
                plt.title("Top Object Names")
                plt.ylabel("count")
                plt.xticks(rotation=60, ha='right')
                plt.tight_layout()
                plt.savefig((eval_dir / "top_names.png").as_posix(), dpi=160)
                plt.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
