#!/usr/bin/env python3
"""
Script 2 of 2: post-hoc AUC + scaling + comparison plots from the raw
probability curves written by scripts/gen_raw_prob_curves.py. NO MODEL --
pure array math on the stored raw curves, so it re-runs in seconds and any
AUC/normalization/plot change never touches the GPU again.

For every config's <config>/eval/snmf/raw_prob_curves.npz it:
  - groups configs by (condition, crop_mode), pooling seeds,
  - applies the scaling (self-relative: each curve / its own full-concept-
    vector reference point; a single swappable function),
  - computes insertion & deletion AUC as the trapezoid of the LOG-scaled
    curve over the full [0, 1] coordinate range (log(prob/ref) -- reveals
    effect size that the raw linear ratio compresses near the shared
    zero-vector floor/ceiling; linear AUC is also kept in the table for
    reference),
  - writes a comparison table (CSV) of insertion/deletion AUC per method +
    crop mode + rank + order, and
  - draws combined comparison plots: one insertion figure and one deletion
    figure per (crop_mode, rank), every condition as a line, real vs random.

Usage:
    python scripts/posthoc_auc_curves.py                       # rank 1, all crop modes
    python scripts/posthoc_auc_curves.py --ranks 1,2,3
    python scripts/posthoc_auc_curves.py --out-dir outputs/rebuttal_ablation_10class/_raw_report
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).parent.parent.resolve()
ABLATION_ROOT = ROOT_DIR / "outputs" / "rebuttal_ablation_10class"
METHOD = "snmf"

# longest-first so P_bin_fullpool / P_bin_shuf match before P_bin
CONDITIONS = ["P_bin_fullpool", "P_bin_shuf", "P_bin", "P_null", "P_open"]
CROP_MODES = ["sliding_window", "langsam", "none"]
COND_ORDER = ["P_bin", "P_bin_fullpool", "P_bin_shuf", "P_null", "P_open"]  # legend/plot order
COND_COLORS = {
    "P_bin": "#2f6f5e", "P_bin_fullpool": "#6fbfa4", "P_bin_shuf": "#a8631a",
    "P_null": "#5b6b78", "P_open": "#9a6fb0",
}


def parse_name(name: str):
    seed = None
    base = name
    if "_seed" in base:
        base, s = base.rsplit("_seed", 1)
        try:
            seed = int(s)
        except ValueError:
            seed = None
    for cond in CONDITIONS:
        if base == cond or base.startswith(cond + "_"):
            crop = base[len(cond):].lstrip("_")
            if crop in CROP_MODES:
                return cond, crop, seed
    return None, None, seed


def scale_curve(curve: np.ndarray, kind: str) -> Optional[np.ndarray]:
    """Scale each curve by its own FULL-CONCEPT-VECTOR reference point, so
    1.0 always means "the complete, unperturbed concept vector" -- the one
    state deletion and insertion curves share (same vector, same target
    token), just reached from opposite ends of their own curve:
      - deletion:  k=0   (nothing zeroed yet)   = the full vector
      - insertion: k=max (everything inserted)  = the full vector
    Dividing deletion by curve[0] (as before) was already correct -- that
    point IS its curve's max 100% of the time (verified empirically).
    Dividing insertion by curve[0] (the OLD bug) used the blank/zeroed
    state as the reference instead -- the curve's own minimum, not a
    ceiling, so climbing back to real signal legitimately exceeded 1.
    Fixed: insertion now divides by curve[-1] (its own full-vector point),
    NOT by curve.max() -- the raw curve isn't perfectly monotonic (model
    noise can make an interior point marginally higher than the true
    full-vector endpoint), so per-curve max would anchor 1.0 to an
    arbitrary noise bump instead of the one physically meaningful state.
    Still fully self-contained: every curve provides its own reference,
    no cross-file/cross-curve lookup needed."""
    ref = float(curve[0]) if kind == "del" else float(curve[-1])
    if ref == 0.0:
        return None
    return curve / ref


def curve_auc(fracs: np.ndarray, scaled: np.ndarray, log_scale: bool = False) -> float:
    """Trapezoid of the (optionally log-transformed) scaled curve over the
    coordinate axis, divided by the axis span (a proper average height).
    fracs spans the full [0, 1] (100% of coordinates).

    log_scale=True integrates log(scaled) = log(curve) - log(ref) instead of
    the raw ratio -- i.e. log(p / p_ref), the log of the exact same ratio
    scale_curve() already produces, so it costs nothing extra (no new data,
    same stored raw curves). This matters because the raw ratio is compressed
    near the zero-vector floor for weak methods (all cluster ~0.94-0.965)
    while the strong method sits at ~0.78 -- a ~1.2x linear gap. In log
    space the SAME numbers become -0.036..-0.062 (weak) vs -0.246 (strong),
    a ~4-7x gap -- log-ratio is "bits of information the concept adds", a
    more faithful measure of effect size than the linear ratio, which
    saturates as probabilities approach the shared ceiling/floor."""
    trapezoid = getattr(np, "trapezoid", np.trapz)
    span = float(fracs[-1] - fracs[0])
    if span <= 0:
        return float("nan")
    y = np.log(np.clip(scaled, 1e-12, None)) if log_scale else scaled
    return float(trapezoid(y, fracs)) / span


def shift_curve_and_auc(curve: np.ndarray, fracs: np.ndarray, kind: str) -> float:
    """Additive (not ratio) alternative: anchor each curve's own START point
    to a fixed shared value -- deletion starts at 1 (k=0 = full vector, same
    anchor the ratio scaling already uses), insertion starts at 0 (k=0 =
    blank/zero-vector state). Unlike dividing by (p_full - p_blank), this
    never renormalizes by a difference that can be near-zero for weak
    methods -- it only subtracts a single reference value, so it can't blow
    up. Returns the AUC re-expressed as a magnitude in raw probability
    units, comparable between insertion and deletion: for deletion this is
    "1 - auc" (the drop from the full-vector state), for insertion it's the
    auc itself (the climb from the blank state) -- so bigger always means
    "more sensitive to the concept" in both directions.
    Caveat: because this is additive in raw probability units (not a
    per-curve ratio), it does NOT cancel out the fact that different
    conditions' generated captions can pick target tokens with different
    natural base rates -- unlike scale_curve()'s ratio, which self-
    normalizes that away. Keep alongside the ratio/log-ratio AUCs, not as
    a standalone replacement."""
    ref = float(curve[0])  # k=0 is "nothing perturbed yet": full vector (del) or blank (ins)
    shifted = curve - ref + (1.0 if kind == "del" else 0.0)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    span = float(fracs[-1] - fracs[0])
    if span <= 0:
        return float("nan")
    auc = float(trapezoid(shifted, fracs)) / span
    return (1.0 - auc) if kind == "del" else auc


def load_all():
    """Returns {(cond, crop): {(rank, kind, order): {'fracs':.., 'scaled':[stacked scaled],
    'aucs':[linear per-image aucs], 'aucs_log':[log per-image aucs],
    'aucs_shift':[additive-shift per-image aucs, raw probability units]}}}."""
    data = defaultdict(lambda: defaultdict(lambda: {
        "fracs": None, "scaled": [], "aucs": [], "aucs_log": [], "aucs_shift": [],
    }))
    n_files = 0
    for d in sorted(ABLATION_ROOT.glob("P_*")):
        npz = d / "eval" / METHOD / "raw_prob_curves.npz"
        if not npz.exists():
            continue
        cond, crop, seed = parse_name(d.name)
        if cond is None:
            continue
        z = np.load(npz, allow_pickle=True)
        n_files += 1
        for rank in (1, 2, 3):
            fk = f"fracs_r{rank}"
            if fk not in z:
                continue
            fracs = z[fk].astype(np.float64)
            for kind in ("ins", "del"):
                for order in ("value", "random"):
                    ck = f"{kind}_{order}_r{rank}_curves"
                    if ck not in z:
                        continue
                    curves = z[ck].astype(np.float64)  # [N_img, P]
                    slot = data[(cond, crop)][(rank, kind, order)]
                    slot["fracs"] = fracs
                    for row in curves:
                        sc = scale_curve(row, kind)
                        if sc is None:
                            continue
                        slot["scaled"].append(sc)
                        slot["aucs"].append(curve_auc(fracs, sc, log_scale=False))
                        slot["aucs_log"].append(curve_auc(fracs, sc, log_scale=True))
                        slot["aucs_shift"].append(shift_curve_and_auc(row, fracs, kind))
    return data, n_files


def write_table(data, out_csv: Path):
    cols = [
        "condition", "crop_mode", "rank", "kind", "order", "n_images",
        "auc_log_mean", "auc_log_std", "auc_linear_mean", "auc_linear_std",
        "auc_shift_mean", "auc_shift_std",
    ]
    rows = []
    for (cond, crop), inner in data.items():
        for (rank, kind, order), slot in inner.items():
            aucs = np.asarray(slot["aucs"], dtype=np.float64)
            aucs_log = np.asarray(slot["aucs_log"], dtype=np.float64)
            aucs_shift = np.asarray(slot["aucs_shift"], dtype=np.float64)
            if aucs.size == 0:
                continue
            rows.append({
                "condition": cond, "crop_mode": crop, "rank": rank, "kind": kind, "order": order,
                "n_images": aucs.size,
                "auc_log_mean": float(aucs_log.mean()), "auc_log_std": float(aucs_log.std()),
                "auc_linear_mean": float(aucs.mean()), "auc_linear_std": float(aucs.std()),
                "auc_shift_mean": float(aucs_shift.mean()), "auc_shift_std": float(aucs_shift.std()),
            })
    rows.sort(key=lambda r: (r["rank"], r["kind"], r["crop_mode"],
                             COND_ORDER.index(r["condition"]) if r["condition"] in COND_ORDER else 99,
                             r["order"]))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows)")
    return rows


def plot_comparison(data, crop: str, rank: int, kind: str, out_png: Path):
    """One figure: every condition's mean log-scaled curve for this (crop,
    rank, kind), real order solid + random order dashed. Plotted in log
    space (log(prob / own full-vector reference)) rather than the raw ratio
    -- the raw ratio compresses all the weak methods near the shared
    zero-vector floor/ceiling, visually flattening real differences in
    effect size; log-ratio is the quantity curve_auc(log_scale=True)
    actually integrates, so the plot matches the reported AUC."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    plotted = False
    for cond in COND_ORDER:
        color = COND_COLORS[cond]
        for order, style, alpha in (("value", "-", 1.0), ("random", "--", 0.45)):
            slot = data.get((cond, crop), {}).get((rank, kind, order))
            if not slot or not slot["scaled"]:
                continue
            fracs = slot["fracs"]
            log_curves = np.log(np.clip(np.stack(slot["scaled"], axis=0), 1e-12, None))
            mean_curve = np.mean(log_curves, axis=0)
            label = cond if order == "value" else None
            ax.plot(fracs, mean_curve, style, color=color, alpha=alpha, linewidth=2 if order == "value" else 1.3, label=label)
            plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"C-{'Insertion' if kind == 'ins' else 'Deletion'} — {crop}, top-{rank}\n(solid = ranked order, dashed = random)")
    ax.set_xlabel("fraction of concept coordinates")
    ax.set_ylabel("log(prob / own full-vector reference)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png.as_posix(), dpi=150)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", default="1", help="comma list of ranks to plot/table, e.g. 1 or 1,2,3")
    ap.add_argument("--out-dir", default=str(ABLATION_ROOT / "_raw_report"))
    args = ap.parse_args()
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, n_files = load_all()
    print(f"Loaded raw curves from {n_files} configs")
    if n_files == 0:
        print("No raw_prob_curves.npz found yet -- run scripts/gen_raw_prob_curves.py first.")
        return

    write_table(data, out_dir / "auc_comparison.csv")

    crops = sorted({crop for (_, crop) in data.keys()})
    n_plots = 0
    for crop in crops:
        for rank in ranks:
            for kind in ("ins", "del"):
                png = out_dir / f"curves_{kind}_{crop}_rank{rank}.png"
                if plot_comparison(data, crop, rank, kind, png):
                    n_plots += 1
                    print(f"  plotted {png.name}")
    print(f"\nWrote {n_plots} comparison plots + auc_comparison.csv to {out_dir}")


if __name__ == "__main__":
    main()
