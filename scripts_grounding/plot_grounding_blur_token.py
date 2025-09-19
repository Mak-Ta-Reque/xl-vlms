#!/usr/bin/env python3
"""
Plot token probability vs blur strength for grounding evaluation.

Reads token_blur_curve.csv and draws a simple line plot.
"""
from __future__ import annotations

import csv
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot token probability vs blur strength")
    ap.add_argument("--eval_dir", required=True)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    csv_path = eval_dir / "token_blur_curve.csv"
    if not csv_path.exists():
        print(f"Missing CSV: {csv_path}")
        return

    xs = []
    ys = []
    with csv_path.open("r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            try:
                xs.append(float(row[0]))
                ys.append(float(row[1]))
            except Exception:
                continue

    if not xs:
        print("No data to plot")
        return

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(xs, ys, marker="o")
    plt.title("Token prob vs blur strength")
    plt.xlabel("blur strength")
    plt.ylabel("mean token prob")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = eval_dir / "token_blur_curve.png"
    plt.savefig(out_path.as_posix(), dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
