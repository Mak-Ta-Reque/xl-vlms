#!/usr/bin/env python3
"""
10-class variant of scripts/rebuttal_stats.py: assembles the same kind of
deliverables (per_image_auc.csv, summary.csv, stats.csv, rank_ablation.csv,
RESULTS.md) from scripts/run_rebuttal_ablation_10class.py's 33-config grid
(outputs/rebuttal_ablation_10class), instead of the 3-class 45-config grid.

Differences from rebuttal_stats.py, not just a copy-paste:
  - ABLATION_ROOT points at outputs/rebuttal_ablation_10class.
  - build_grid/config_name imported from run_rebuttal_ablation_10class
    (10-class CONDITIONS embeds its own class list at construction time,
    per that script's own module docstring).
  - No saliency baselines (attention_map/gradcam) -- those were only ever
    run once for the 3-class experiment; COMPARISONS drops them.
  - class_lookup reads data/coco10/eval_images_rebuttal10.json (10-class
    fixed eval manifest, 500 images / 50 per category), not the 3-class one.
  - The 33-config grid is still IN PROGRESS at the time this script is run
    (only some configs done) -- every function here already degrades
    gracefully to whatever's on disk (_read_per_image_csv returns {} for a
    config that hasn't reached step 7 yet, build_stats skips any pairwise
    comparison with fewer than 5 paired images), so this can be re-run at
    any point to report only what's actually finished. summary.csv rows
    additionally report n_seeds_present so partial seed coverage is visible
    rather than silently averaged over as if it were complete.

Usage:
    python scripts/rebuttal_stats_10class.py
    python scripts/rebuttal_stats_10class.py --out-dir outputs/rebuttal_ablation_10class/_report
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(Path(__file__).parent))

from ablation_report import (  # noqa: E402
    _load_concepts,
    hoyer_selectivity,
    atom_sparsity_relative,
    concept_overlap,
    matched_cosine_similarity,
)
from run_rebuttal_ablation_10class import build_grid, config_name  # noqa: E402

ABLATION_ROOT = ROOT_DIR / "outputs" / "rebuttal_ablation_10class"
METHOD = "snmf"
RANK = 1
# P_bin is the reference condition every other condition is compared against.
COMPARISONS = ["P_open", "P_null", "P_bin_shuf", "P_bin_fullpool"]
SIG_CROP_MODES = ["sliding_window", "langsam"]  # "none" (Tier 1) is descriptive-only, 1 seed


def _load_true_baseline(eval_dir: Path, rank: int) -> Optional[float]:
    """The true, complete, unmasked baseline probability for this rank --
    literally the deletion curve's own fraction=0 value (nothing zeroed
    yet), read from the already-saved aggregate CSV (no per-image backfill
    needed, matches scripts/plot_concept_deletion_eval_token.py's
    baseline_by_rank exactly, which uses this same file for the same
    reason: insertion's own fraction=0 point is a blank/zeroed input, not a
    meaningful reference -- the deletion curve's fraction=0 point is the
    real one)."""
    path = eval_dir / f"c_deletion_token_rank{rank}.csv"
    if not path.exists():
        return None
    with path.open() as f:
        row = next(csv.DictReader(f), None)
    if row is None:
        return None
    try:
        return float(row["mean_prob"])
    except (KeyError, ValueError):
        return None


def _read_per_image_csv(path: Path) -> Dict[str, Tuple[float, float, Optional[float], Optional[float]]]:
    """Returns {image_path_basename: (auc, auc_relative, auc_start_relative, y0)}.
    y0 is each image's OWN curve's fraction=0 value -- only present in
    per_image CSVs written after the y0 field was added; older files yield
    None for it (falls back gracefully, see assemble_per_image_rows).
    Missing files (config hasn't reached step 7 yet) yield {}."""
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = Path(row["image_path"]).name
            auc = float(row["auc"]) if row["auc"] not in (None, "") else None
            auc_rel = float(row["auc_relative"]) if row.get("auc_relative") not in (None, "", "None") else None
            auc_start_rel = (
                float(row["auc_start_relative"])
                if row.get("auc_start_relative") not in (None, "", "None")
                else None
            )
            y0 = float(row["y0"]) if row.get("y0") not in (None, "", "None") else None
            out[key] = (auc, auc_rel, auc_start_rel, y0)
    return out


SPAN = 0.7001953125  # NUM_POINTS=70 -> fraction axis covers [0, 0.70], not [0, 1]


def assemble_per_image_rows() -> List[dict]:
    """Self-relative: each curve (insertion and deletion, independently)
    divided by its OWN fraction=0 value -- no cross-referencing another
    curve, no config-level aggregate. This is auc_start_relative, already
    computed and stored per-image in eval/concept_deletion_eval.py
    (span-corrected). Per explicit directive: AUC measures the change from
    each curve's own starting point, full stop."""
    rows = []
    for condition, crop_mode, seed in build_grid():
        run_dir = ABLATION_ROOT / config_name(condition, crop_mode, seed)
        eval_dir = run_dir / "eval" / METHOD
        for order_mode, suffix in (("value", ""), ("random", "_random")):
            ins = _read_per_image_csv(eval_dir / f"c_insertion_token_rank{RANK}{suffix}_per_image.csv")
            dele = _read_per_image_csv(eval_dir / f"c_deletion_token_rank{RANK}{suffix}_per_image.csv")
            keys = set(ins) | set(dele)
            for key in keys:
                ins_auc, ins_rel, ins_start_rel, _ = ins.get(key, (None, None, None, None))
                del_auc, del_rel, del_start_rel, _ = dele.get(key, (None, None, None, None))
                rows.append({
                    "image_path": key,
                    "class": None,  # filled below via eval_images_rebuttal10.json lookup
                    "condition": condition,
                    "crop_mode": crop_mode,
                    "seed": seed,
                    "order_mode": order_mode,
                    "insertion_auc": ins_auc,
                    "insertion_auc_relative": ins_rel,
                    "insertion_auc_scaled": ins_start_rel,
                    "deletion_auc": del_auc,
                    "deletion_auc_relative": del_rel,
                    "deletion_auc_scaled": del_start_rel,
                })
    class_lookup = {}
    manifest_path = ROOT_DIR / "data" / "coco10" / "eval_images_rebuttal10.json"
    if manifest_path.exists():
        import json
        manifest = json.load(open(manifest_path))
        for cls, paths in manifest.get("images", {}).items():
            for p in paths:
                class_lookup[Path(p).name] = cls
    for row in rows:
        row["class"] = class_lookup.get(row["image_path"], row["class"])
    return rows


def _unused_apply_global_scale(per_image_rows: List[dict], rank_ablation_rows: List[dict]) -> None:
    """No longer used -- kept for reference. Per user directive, the metric
    now matches the plot script's own per-curve y/y0 transform exactly
    (see assemble_per_image_rows/assemble_rank_ablation_rows), not a
    global-pooled min-max. This function is dead code, left in place only
    so the earlier reasoning about the global-scale tradeoff stays
    discoverable in git history / this file rather than silently vanishing.

    One global min and one global max for insertion (and separately for
    deletion), pooled from raw AUC across EVERYTHING -- every condition,
    every crop mode, every rank, both data sources -- then applied
    uniformly to every row in both lists as *_auc_scaled columns.

    This is still just a single shared affine transform ((x-lo)/(hi-lo)),
    so it changes no paired comparison's sign/significance versus raw AUC
    (same invariant as the earlier per-crop-mode and z-score versions) --
    but pooling GLOBALLY rather than per-rank (like the earlier
    auc_start_relative fix did) guarantees every value lands in [0,1] by
    construction, which per-curve start-relative normalization could not
    guarantee for insertion curves specifically (insertion's own fraction=0
    value is its curve's minimum, not maximum, so dividing by it can
    legitimately exceed 1 -- confirmed empirically, not a bug, but not
    bounded either). Using the true dataset-wide min/max removes that
    asymmetry entirely."""
    only_value = [r for r in per_image_rows if r.get("order_mode", "value") == "value"]
    ins_vals = [r["insertion_auc"] for r in only_value if r["insertion_auc"] is not None]
    del_vals = [r["deletion_auc"] for r in only_value if r["deletion_auc"] is not None]
    ins_vals += [r["insertion_auc"] for r in rank_ablation_rows if r["insertion_auc"] is not None]
    del_vals += [r["deletion_auc"] for r in rank_ablation_rows if r["deletion_auc"] is not None]

    ins_min, ins_max = (min(ins_vals), max(ins_vals)) if ins_vals else (None, None)
    del_min, del_max = (min(del_vals), max(del_vals)) if del_vals else (None, None)

    def _scale(v, lo, hi):
        if v is None or lo is None or hi is None or hi <= lo:
            return None
        return (v - lo) / (hi - lo)

    for r in per_image_rows:
        r["insertion_auc_scaled"] = _scale(r["insertion_auc"], ins_min, ins_max)
        r["deletion_auc_scaled"] = _scale(r["deletion_auc"], del_min, del_max)
    for r in rank_ablation_rows:
        r["insertion_auc_scaled"] = _scale(r["insertion_auc"], ins_min, ins_max)
        r["deletion_auc_scaled"] = _scale(r["deletion_auc"], del_min, del_max)


def write_per_image_csv(rows: List[dict], out_path: Path) -> None:
    cols = ["image_path", "class", "condition", "crop_mode", "seed", "order_mode",
            "insertion_auc", "insertion_auc_relative", "insertion_auc_scaled",
            "deletion_auc", "deletion_auc_relative", "deletion_auc_scaled"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])
    print(f"Wrote {out_path} ({len(rows)} rows)")


def _condition_crop_bank_dir(condition: str, crop_mode: str, seed: int) -> Path:
    return ABLATION_ROOT / config_name(condition, crop_mode, seed)


def find_concept_bank(run_dir: Path) -> Optional[Path]:
    import glob
    matches = glob.glob(str(run_dir / "concept" / METHOD / f"combined_concept_{METHOD}_*_raw.pth"))
    return Path(matches[0]) if matches else None


def build_summary(rows: List[dict]) -> List[dict]:
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["condition"], r["crop_mode"], r.get("order_mode", "value"))].append(r)

    summary = []
    for (condition, crop_mode, order_mode), grp in grouped.items():
        ins_vals = [r["insertion_auc_scaled"] for r in grp if r["insertion_auc_scaled"] is not None]
        del_vals = [r["deletion_auc_scaled"] for r in grp if r["deletion_auc_scaled"] is not None]
        ins_rel_vals = [r["insertion_auc_relative"] for r in grp if r["insertion_auc_relative"] is not None]
        del_rel_vals = [r["deletion_auc_relative"] for r in grp if r["deletion_auc_relative"] is not None]

        by_seed = defaultdict(list)
        for r in grp:
            if r["insertion_auc_scaled"] is not None:
                by_seed[r["seed"]].append(r["insertion_auc_scaled"])
        seed_means = [float(np.mean(v)) for v in by_seed.values() if v]
        seed_std = float(np.std(seed_means)) if len(seed_means) > 1 else None
        n_seeds_present = len(by_seed)

        row = {
            "condition": condition, "crop_mode": crop_mode, "order_mode": order_mode,
            "n_images": len(grp),
            "n_seeds_present": n_seeds_present,
            "insertion_auc_scaled_mean": float(np.mean(ins_vals)) if ins_vals else None,
            "insertion_auc_scaled_std": float(np.std(ins_vals)) if ins_vals else None,
            "deletion_auc_scaled_mean": float(np.mean(del_vals)) if del_vals else None,
            "deletion_auc_scaled_std": float(np.std(del_vals)) if del_vals else None,
            "insertion_auc_relative_mean": float(np.mean(ins_rel_vals)) if ins_rel_vals else None,
            "insertion_auc_relative_std": float(np.std(ins_rel_vals)) if ins_rel_vals else None,
            "deletion_auc_relative_mean": float(np.mean(del_rel_vals)) if del_rel_vals else None,
            "deletion_auc_relative_std": float(np.std(del_rel_vals)) if del_rel_vals else None,
            "seed_to_seed_std_insertion": seed_std,
        }

        if crop_mode != "NA" and order_mode == "value":
            seeds_present = sorted(set(r["seed"] for r in grp))
            bank_path = find_concept_bank(_condition_crop_bank_dir(condition, crop_mode, seeds_present[0])) if seeds_present else None
            C = _load_concepts(bank_path) if bank_path else None
            row["sparsity_higher_better"] = hoyer_selectivity(C) if C is not None else None
            row["sparsity_frac_near_zero_1pct"] = atom_sparsity_relative(C) if C is not None else None
            row["overlap_lower_better"] = concept_overlap(C) if C is not None else None
            if len(seeds_present) > 1:
                bank2 = find_concept_bank(_condition_crop_bank_dir(condition, crop_mode, seeds_present[1]))
                C2 = _load_concepts(bank2) if bank2 else None
                if C is not None and C2 is not None:
                    row["instability_lower_better"] = 1.0 - matched_cosine_similarity(C.numpy(), C2.numpy())
                else:
                    row["instability_lower_better"] = None
            else:
                row["instability_lower_better"] = None
        else:
            row["sparsity_higher_better"] = None
            row["sparsity_frac_near_zero_1pct"] = None
            row["overlap_lower_better"] = None
            row["instability_lower_better"] = None

        summary.append(row)
    return summary


def write_summary_csv(summary: List[dict], out_path: Path) -> None:
    cols = ["condition", "crop_mode", "order_mode", "n_images", "n_seeds_present",
            "insertion_auc_scaled_mean", "insertion_auc_scaled_std",
            "deletion_auc_scaled_mean", "deletion_auc_scaled_std",
            "insertion_auc_relative_mean", "insertion_auc_relative_std",
            "deletion_auc_relative_mean", "deletion_auc_relative_std",
            "seed_to_seed_std_insertion",
            "sparsity_higher_better", "sparsity_frac_near_zero_1pct",
            "overlap_lower_better", "instability_lower_better"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in summary:
            w.writerow([r.get(c) for c in cols])
    print(f"Wrote {out_path} ({len(summary)} rows)")


def _rank_biserial(diffs: np.ndarray) -> Optional[float]:
    diffs = diffs[diffs != 0]
    if diffs.size == 0:
        return None
    ranks = np.argsort(np.argsort(np.abs(diffs))) + 1
    pos = ranks[diffs > 0].sum()
    neg = ranks[diffs < 0].sum()
    total = pos + neg
    if total == 0:
        return None
    return float((pos - neg) / total)


def _bootstrap_ci(diffs: np.ndarray, n_resamples: int = 10000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = diffs[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _holm_bonferroni(pvals: List[float]) -> List[float]:
    order = np.argsort(pvals)
    n = len(pvals)
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def _paired_metric_by_image(rows: List[dict], condition: str, crop_mode: str, metric: str) -> Dict[str, float]:
    from collections import defaultdict
    per_image = defaultdict(list)
    for r in rows:
        if (r["condition"] == condition and r["crop_mode"] == crop_mode
                and r.get("order_mode", "value") == "value" and r[metric] is not None):
            per_image[r["image_path"]].append(r[metric])
    return {img: float(np.mean(vals)) for img, vals in per_image.items() if vals}


def build_stats(rows: List[dict]) -> List[dict]:
    stats_rows = []
    raw_pvals = []
    pending = []

    for crop_mode in SIG_CROP_MODES:
        for comparison in COMPARISONS:
            for metric in ["insertion_auc_scaled", "deletion_auc_scaled"]:
                ref = _paired_metric_by_image(rows, "P_bin", crop_mode, metric)
                cmp_ = _paired_metric_by_image(rows, comparison, crop_mode, metric)
                shared = sorted(set(ref) & set(cmp_))
                n = len(shared)
                if n < 5:
                    continue
                x = np.array([ref[k] for k in shared])
                y = np.array([cmp_[k] for k in shared])
                diffs = x - y
                try:
                    stat, p_raw = wilcoxon(x, y, alternative="two-sided")
                except ValueError:
                    continue
                ci_low, ci_high = _bootstrap_ci(diffs)
                r_rb = _rank_biserial(diffs)
                entry = {
                    "crop_mode": crop_mode, "comparison": f"P_bin_vs_{comparison}",
                    "metric": metric, "n": n, "wilcoxon_stat": float(stat),
                    "p_raw": float(p_raw), "rank_biserial": r_rb,
                    "mean_diff": float(diffs.mean()), "ci_low": ci_low, "ci_high": ci_high,
                }
                pending.append(entry)
                raw_pvals.append(p_raw)

    if raw_pvals:
        adjusted = _holm_bonferroni(raw_pvals)
        for entry, p_holm in zip(pending, adjusted):
            entry["p_holm"] = p_holm
            stats_rows.append(entry)
    return stats_rows


def write_stats_csv(stats_rows: List[dict], out_path: Path) -> None:
    cols = ["crop_mode", "comparison", "metric", "n", "wilcoxon_stat",
            "p_raw", "p_holm", "rank_biserial", "mean_diff", "ci_low", "ci_high"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in stats_rows:
            w.writerow([r.get(c) for c in cols])
    print(f"Wrote {out_path} ({len(stats_rows)} rows)")


RANK_ABLATION_CONDITIONS = ["P_null", "P_open", "P_bin"]
RANK_ABLATION_RANKS = [1, 2, 3]


def assemble_rank_ablation_rows() -> List[dict]:
    """Same self-relative transform as assemble_per_image_rows(): each curve
    divided by its own fraction=0 value, independently for insertion and
    deletion, no cross-referencing, no aggregate."""
    rows = []
    for condition, crop_mode, seed in build_grid():
        if condition not in RANK_ABLATION_CONDITIONS or crop_mode not in SIG_CROP_MODES:
            continue
        run_dir = ABLATION_ROOT / config_name(condition, crop_mode, seed)
        eval_dir = run_dir / "eval" / METHOD
        for rank in RANK_ABLATION_RANKS:
            ins = _read_per_image_csv(eval_dir / f"c_insertion_token_rank{rank}_per_image.csv")
            dele = _read_per_image_csv(eval_dir / f"c_deletion_token_rank{rank}_per_image.csv")
            keys = set(ins) | set(dele)
            for key in keys:
                ins_auc, ins_rel, ins_start_rel, _ = ins.get(key, (None, None, None, None))
                del_auc, del_rel, del_start_rel, _ = dele.get(key, (None, None, None, None))
                rows.append({
                    "image_path": key, "condition": condition, "crop_mode": crop_mode,
                    "seed": seed, "rank": rank,
                    "insertion_auc": ins_auc, "insertion_auc_relative": ins_rel,
                    "insertion_auc_scaled": ins_start_rel,
                    "deletion_auc": del_auc, "deletion_auc_relative": del_rel,
                    "deletion_auc_scaled": del_start_rel,
                })
    return rows


def build_rank_ablation_summary(rows: List[dict]) -> List[dict]:
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["condition"], r["crop_mode"], r["rank"])].append(r)

    summary = []
    for (condition, crop_mode, rank), grp in grouped.items():
        ins_vals = [r["insertion_auc_scaled"] for r in grp if r["insertion_auc_scaled"] is not None]
        del_vals = [r["deletion_auc_scaled"] for r in grp if r["deletion_auc_scaled"] is not None]
        ins_rel_vals = [r["insertion_auc_relative"] for r in grp if r["insertion_auc_relative"] is not None]
        del_rel_vals = [r["deletion_auc_relative"] for r in grp if r["deletion_auc_relative"] is not None]
        summary.append({
            "condition": condition, "crop_mode": crop_mode, "rank": rank,
            "n_images": len(grp),
            "insertion_auc_scaled_mean": float(np.mean(ins_vals)) if ins_vals else None,
            "insertion_auc_scaled_std": float(np.std(ins_vals)) if ins_vals else None,
            "deletion_auc_scaled_mean": float(np.mean(del_vals)) if del_vals else None,
            "deletion_auc_scaled_std": float(np.std(del_vals)) if del_vals else None,
            "insertion_auc_relative_mean": float(np.mean(ins_rel_vals)) if ins_rel_vals else None,
            "insertion_auc_relative_std": float(np.std(ins_rel_vals)) if ins_rel_vals else None,
            "deletion_auc_relative_mean": float(np.mean(del_rel_vals)) if del_rel_vals else None,
            "deletion_auc_relative_std": float(np.std(del_rel_vals)) if del_rel_vals else None,
        })
    summary.sort(key=lambda r: (r["condition"], r["crop_mode"], r["rank"]))
    return summary


def write_rank_ablation_csv(summary: List[dict], out_path: Path) -> None:
    cols = ["condition", "crop_mode", "rank", "n_images",
            "insertion_auc_scaled_mean", "insertion_auc_scaled_std",
            "deletion_auc_scaled_mean", "deletion_auc_scaled_std",
            "insertion_auc_relative_mean", "insertion_auc_relative_std",
            "deletion_auc_relative_mean", "deletion_auc_relative_std"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in summary:
            w.writerow([r.get(c) for c in cols])
    print(f"Wrote {out_path} ({len(summary)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(ABLATION_ROOT / "_report"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = assemble_per_image_rows()
    rank_rows = assemble_rank_ablation_rows()

    write_per_image_csv(rows, out_dir / "per_image_auc.csv")

    summary = build_summary(rows)
    write_summary_csv(summary, out_dir / "summary.csv")

    stats_rows = build_stats(rows)
    write_stats_csv(stats_rows, out_dir / "stats.csv")

    rank_summary = build_rank_ablation_summary(rank_rows)
    write_rank_ablation_csv(rank_summary, out_dir / "rank_ablation.csv")

    print("Configs with any data:", sorted(set((r["condition"], r["crop_mode"], r["seed"]) for r in rows)))


if __name__ == "__main__":
    main()
