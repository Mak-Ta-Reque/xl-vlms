#!/usr/bin/env python3
"""
Assemble the TGCL rebuttal ablation's deliverables from the 45-run grid
(scripts/run_rebuttal_ablation.py) plus the 2 one-shot baselines
(eval/patch_deletion_eval.py --method attention/gradcam):

  - per_image_auc.csv : one row per (condition, crop_mode, seed, image_path),
    the actual significance-test deliverable.
  - summary.csv       : one row per condition x crop_mode, mean/std across
    seeds, plus concept-bank sparsity/overlap/instability (reusing
    scripts/ablation_report.py's already-validated functions).
  - stats.csv         : paired Wilcoxon signed-rank (insertion + deletion),
    bootstrap 95% CI, rank-biserial effect size, Holm-Bonferroni-corrected
    p-values, for P_bin vs each of {P_open, P_null, P_bin_shuf,
    P_bin_fullpool, attention_map, gradcam}, per crop mode.
  - RESULTS.md        : human-readable report assembled from the above.

Usage:
    python scripts/rebuttal_stats.py
    python scripts/rebuttal_stats.py --out-dir outputs/rebuttal_ablation/_report
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
from run_rebuttal_ablation import build_grid, config_name  # noqa: E402

ABLATION_ROOT = ROOT_DIR / "outputs" / "rebuttal_ablation"
METHOD = "snmf"
RANK = 1
BASELINE_DIRS = {
    "attention_map": ABLATION_ROOT / "attention_map_baseline",
    "gradcam": ABLATION_ROOT / "gradcam_baseline",
}
# P_bin is the reference condition every other condition is compared against.
COMPARISONS = ["P_open", "P_null", "P_bin_shuf", "P_bin_fullpool", "attention_map", "gradcam"]
SIG_CROP_MODES = ["sliding_window", "langsam"]  # "none" (Tier 1) is descriptive-only, 1 seed


def _read_per_image_csv(path: Path) -> Dict[str, Tuple[float, float, Optional[float]]]:
    """Returns {image_path_basename: (auc, auc_relative, auc_start_relative)}.
    auc_start_relative is only present in per_image CSVs regenerated after
    the rank-comparison fix (eval/concept_deletion_eval.py's
    _curve_auc_start_relative) -- missing/older files yield None for it."""
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
            out[key] = (auc, auc_rel, auc_start_rel)
    return out


def _image_class(image_path: str) -> str:
    return Path(image_path).parent.name


def assemble_per_image_rows() -> List[dict]:
    rows = []
    # order_mode="value": the real, saliency/gradient-ranked masking order.
    # order_mode="random": chance-level control, same masking mechanism with
    # coordinates/patches shuffled instead of ranked -- already computed for
    # every grid config by concept_deletion_eval.py's existing --order_mode
    # random pass (run_full_pipeline.py's step 7 does this unconditionally
    # for every rank), so no new compute needed for the grid side.
    for condition, crop_mode, seed in build_grid():
        run_dir = ABLATION_ROOT / config_name(condition, crop_mode, seed)
        eval_dir = run_dir / "eval" / METHOD
        for order_mode, suffix in (("value", ""), ("random", "_random")):
            ins = _read_per_image_csv(eval_dir / f"c_insertion_token_rank{RANK}{suffix}_per_image.csv")
            dele = _read_per_image_csv(eval_dir / f"c_deletion_token_rank{RANK}{suffix}_per_image.csv")
            keys = set(ins) | set(dele)
            for key in keys:
                ins_auc, ins_rel, _ = ins.get(key, (None, None, None))
                del_auc, del_rel, _ = dele.get(key, (None, None, None))
                rows.append({
                    "image_path": key,
                    "class": None,  # filled below via eval_images_rebuttal.json lookup
                    "condition": condition,
                    "crop_mode": crop_mode,
                    "seed": seed,
                    "order_mode": order_mode,
                    "insertion_auc": ins_auc,
                    "insertion_auc_relative": ins_rel,
                    "deletion_auc": del_auc,
                    "deletion_auc_relative": del_rel,
                })
    # Baselines: crop_mode/seed = NA, single run each. File naming uses
    # patch_deletion_eval.py's --method value ("attention"/"gradcam"), which
    # differs slightly from the "attention_map" condition name used
    # elsewhere for readability -- map explicitly rather than string-munge.
    # order_mode="random" here needs patch_deletion_eval.py --order_mode
    # random to have actually been run separately (new compute, unlike the
    # grid side) -- reads {}/None gracefully if that hasn't happened yet.
    method_arg = {"attention_map": "attention", "gradcam": "gradcam"}
    for condition, base_dir in BASELINE_DIRS.items():
        m = method_arg[condition]
        for order_mode, suffix in (("value", ""), ("random", "_random")):
            ins = _read_per_image_csv(base_dir / f"patch_insertion_{m}{suffix}_per_image.csv")
            dele = _read_per_image_csv(base_dir / f"patch_deletion_{m}{suffix}_per_image.csv")
            keys = set(ins) | set(dele)
            for key in keys:
                ins_auc, ins_rel, _ = ins.get(key, (None, None, None))
                del_auc, del_rel, _ = dele.get(key, (None, None, None))
                rows.append({
                    "image_path": key, "class": None,
                    "condition": condition, "crop_mode": "NA", "seed": "NA",
                    "order_mode": order_mode,
                    "insertion_auc": ins_auc, "insertion_auc_relative": ins_rel,
                    "deletion_auc": del_auc, "deletion_auc_relative": del_rel,
                })
    # Fill "class" properly from any grid-run's original image_path (baselines
    # and grid rows share the same underlying eval_images_rebuttal.json set).
    class_lookup = {}
    manifest_path = ROOT_DIR / "data" / "coco10" / "eval_images_rebuttal.json"
    if manifest_path.exists():
        import json
        manifest = json.load(open(manifest_path))
        for cls, paths in manifest.get("images", {}).items():
            for p in paths:
                class_lookup[Path(p).name] = cls
    for row in rows:
        row["class"] = class_lookup.get(row["image_path"], row["class"])
    apply_method_scaling(rows)
    return rows


def apply_method_scaling(rows: List[dict]) -> None:
    """Replace per-CURVE min-max rescaling (each individual image/rank
    independently stretched to its own [0,1] range) with one shared
    (min, max) reference, derived by pooling raw AUC values
    (order_mode="value" only -- the reference shouldn't be contaminated by
    the chance-level control). That reference is then applied uniformly to
    every row in the group (value AND random), in place, as *_auc_scaled
    columns.

    Grouping key is the MEASUREMENT PROTOCOL, not the individual condition:
    - "grid" rows (P_null/P_open/P_bin/P_bin_shuf/P_bin_fullpool) all track
      the probability of the actual generated token via the same
      concept-hidden-state masking mechanism -- they share ONE scale per
      crop_mode (pooling all 5 conditions' raw AUC together). A shared
      linear rescaling applied identically to every row in a Wilcoxon pair
      cannot change the sign of (x - y), so this is exactly
      equivalent to comparing raw AUC directly -- it only rescales for
      readability, changing no conclusion. (Grouping by *condition*
      instead -- what an earlier version of this function did -- gives
      EACH condition its own separate affine map, which reintroduces
      the same kind of confound the original per-curve bug had, just at
      a coarser granularity: confirmed empirically, P_bin vs P_open on
      langsam insertion flips sign between raw AUC and that per-condition
      scaling, p~4e-65 both ways.)
    - "baseline" rows (attention_map/gradcam) use a genuinely different
      protocol (patch-masking, tracking the model's own greedy-argmax
      token on the unmasked image, not the actual generated word) --
      raw AUC there is ~0.9 vs the grid's ~1e-6 (a real, ~1/vocab-size
      floor, not an error), six orders of magnitude apart and not
      meaningfully comparable without SOME bridging transform. These get
      their own separate shared reference (pooling both baselines
      together), used only when comparing a grid condition against them.
    """
    from collections import defaultdict
    pools = defaultdict(lambda: {"ins": [], "del": []})

    def _group_key(r: dict) -> str:
        return "baseline" if r["crop_mode"] == "NA" else r["crop_mode"]

    for r in rows:
        if r.get("order_mode", "value") != "value":
            continue
        key = _group_key(r)
        if r["insertion_auc"] is not None:
            pools[key]["ins"].append(r["insertion_auc"])
        if r["deletion_auc"] is not None:
            pools[key]["del"].append(r["deletion_auc"])

    ranges = {}
    for key, vals in pools.items():
        ins = vals["ins"]
        dele = vals["del"]
        ranges[key] = {
            "ins_min": min(ins) if ins else None, "ins_max": max(ins) if ins else None,
            "del_min": min(dele) if dele else None, "del_max": max(dele) if dele else None,
        }

    def _scale(v, lo, hi):
        if v is None or lo is None or hi is None or hi <= lo:
            return None
        return (v - lo) / (hi - lo)

    for r in rows:
        rng = ranges.get(_group_key(r), {})
        r["insertion_auc_scaled"] = _scale(r["insertion_auc"], rng.get("ins_min"), rng.get("ins_max"))
        r["deletion_auc_scaled"] = _scale(r["deletion_auc"], rng.get("del_min"), rng.get("del_max"))


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
        # insertion_auc_scaled/deletion_auc_scaled (method-pooled scale) are
        # now the PRIMARY reported metric -- insertion_auc_relative (the old
        # per-curve-normalized version) is kept alongside for transparency,
        # since it can disagree with the scaled metric on direction (confirmed
        # for P_bin vs P_null/P_open: per-curve normalization reversed a
        # p=2.3e-65 result).
        ins_vals = [r["insertion_auc_scaled"] for r in grp if r["insertion_auc_scaled"] is not None]
        del_vals = [r["deletion_auc_scaled"] for r in grp if r["deletion_auc_scaled"] is not None]
        ins_rel_vals = [r["insertion_auc_relative"] for r in grp if r["insertion_auc_relative"] is not None]
        del_rel_vals = [r["deletion_auc_relative"] for r in grp if r["deletion_auc_relative"] is not None]

        # Seed-level means, for reporting seed-to-seed std separately (spec
        # sec 9.3): flag if this exceeds the between-condition difference
        # it's meant to support, don't silently average it away.
        by_seed = defaultdict(list)
        for r in grp:
            if r["insertion_auc_scaled"] is not None:
                by_seed[r["seed"]].append(r["insertion_auc_scaled"])
        seed_means = [float(np.mean(v)) for v in by_seed.values() if v]
        seed_std = float(np.std(seed_means)) if len(seed_means) > 1 else None

        row = {
            "condition": condition, "crop_mode": crop_mode, "order_mode": order_mode,
            "n_images": len(grp),
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

        # Bank metrics don't depend on masking order -- only compute them
        # once, under the "value" row, to avoid implying "random" has its
        # own (it doesn't; same concept bank either way).
        if crop_mode != "NA" and order_mode == "value":
            seeds_present = sorted(set(r["seed"] for r in grp))
            bank_path = find_concept_bank(_condition_crop_bank_dir(condition, crop_mode, seeds_present[0]))
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
    cols = ["condition", "crop_mode", "order_mode", "n_images",
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
    """Effect size from the sign/rank structure underlying Wilcoxon: matched
    pairs favoring positive minus negative, normalized -- standard
    rank-biserial correlation for the signed-rank test."""
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
    """Average a condition's own seeds first (necessary: Tier 0 has 5 seeds,
    Tiers 2/3 have 3 -- can't pair mismatched seed counts directly), giving
    one value per image."""
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
            other_crop_mode = "NA" if comparison in BASELINE_DIRS else crop_mode
            for metric in ["insertion_auc_scaled", "deletion_auc_scaled"]:
                ref = _paired_metric_by_image(rows, "P_bin", crop_mode, metric)
                cmp_ = _paired_metric_by_image(rows, comparison, other_crop_mode, metric)
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


def render_results_md(summary: List[dict], stats_rows: List[dict], out_path: Path) -> None:
    lines = []
    lines.append("# TGCL Rebuttal Ablation — Results\n")
    lines.append(
        "**Invariants:** Gemma-3n-E4B (frozen), penultimate residual layer, "
        "Semi-NMF K=2/tag, alpha=20, CLEAN_EXAMPLE_RATIO=0.8, fixed 384-image "
        "eval set (apple/cat/bird, data/coco10/val_masked_rebuttal), "
        "BAG_SIZE=2000.\n"
    )

    lines.append(
        "**Metric note:** `insertion_auc_scaled`/`deletion_auc_scaled` (below) pool each "
        "(condition, crop_mode)'s own raw AUC values into ONE shared min-max reference, "
        "applied uniformly across every seed/image/rank for that group -- not the older "
        "per-CURVE normalization (each individual image/rank independently rescaled to "
        "its own [0,1] range), which was found to reverse a p=2.3e-65 significant result "
        "for P_bin vs P_null. The old per-curve `auc_relative` is kept alongside for "
        "reference/comparison only; treat `*_scaled` as the primary metric.\n"
    )

    lines.append("## Summary (mean ± std across seeds, real ranked order)\n")
    lines.append("| Condition | Crop | N | Ins-AUC (scaled) | Del-AUC (scaled) | Ins−Del gap | Ins-AUC (old per-curve rel.) | Del-AUC (old per-curve rel.) | Sparsity | Overlap | Instability |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary:
        if r.get("order_mode", "value") != "value":
            continue
        ins = r["insertion_auc_scaled_mean"]
        dele = r["deletion_auc_scaled_mean"]
        gap = (ins - dele) if (ins is not None and dele is not None) else None

        def _fmt(v, nd=3):
            return f"{v:.{nd}f}" if v is not None else "—"

        ins_s = f"{_fmt(ins)}±{_fmt(r['insertion_auc_scaled_std'])}"
        del_s = f"{_fmt(dele)}±{_fmt(r['deletion_auc_scaled_std'])}"
        ins_rel_s = f"{_fmt(r['insertion_auc_relative_mean'])}±{_fmt(r['insertion_auc_relative_std'])}"
        del_rel_s = f"{_fmt(r['deletion_auc_relative_mean'])}±{_fmt(r['deletion_auc_relative_std'])}"
        lines.append(
            f"| {r['condition']} | {r['crop_mode']} | {r['n_images']} | {ins_s} | {del_s} | "
            f"{_fmt(gap)} | {ins_rel_s} | {del_rel_s} | {_fmt(r['sparsity_higher_better'])} | "
            f"{_fmt(r['overlap_lower_better'])} | {_fmt(r['instability_lower_better'])} |"
        )
    lines.append("")

    lines.append("## Random-order chance-level controls (mean ± std across seeds)\n")
    lines.append(
        "Same `*_scaled` reference (min/max) as the real-order table above -- these rows are "
        "the same (condition, crop_mode) but with insertion/deletion order randomized instead "
        "of rank/saliency-ranked, so values can fall outside [0,1] (random order doesn't reach "
        "the same extremes as the real ranked order the scale was calibrated on). Included to "
        "show the chance-level floor each real-order row should be compared against.\n"
    )
    lines.append("| Condition | Crop | N | Ins-AUC (scaled) | Del-AUC (scaled) |")
    lines.append("|---|---|---|---|---|")
    for r in summary:
        if r.get("order_mode", "value") != "random":
            continue

        def _fmt(v, nd=3):
            return f"{v:.{nd}f}" if v is not None else "—"

        ins_s = f"{_fmt(r['insertion_auc_scaled_mean'])}±{_fmt(r['insertion_auc_scaled_std'])}"
        del_s = f"{_fmt(r['deletion_auc_scaled_mean'])}±{_fmt(r['deletion_auc_scaled_std'])}"
        lines.append(f"| {r['condition']} | {r['crop_mode']} | {r['n_images']} | {ins_s} | {del_s} |")
    lines.append("")

    lines.append("## Significance (P_bin vs. each control, Holm-corrected)\n")
    lines.append("| Crop | Comparison | Metric | n | W | p_raw | p_holm | rank-biserial | mean diff | 95% CI |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in stats_rows:
        lines.append(
            f"| {r['crop_mode']} | {r['comparison']} | {r['metric']} | {r['n']} | "
            f"{r['wilcoxon_stat']:.2f} | {r['p_raw']:.4g} | {r['p_holm']:.4g} | "
            f"{r['rank_biserial']:.3f} | {r['mean_diff']:.4f} | "
            f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] |"
        )
    lines.append("")

    lines.append("## Seed variance flags\n")
    flagged_any = False
    for r in summary:
        if r.get("order_mode", "value") != "value":
            continue  # random-order rows are a reference baseline, not a compared condition
        seed_std = r.get("seed_to_seed_std_insertion")
        if seed_std is None:
            continue
        # Compare against the smallest between-condition insertion-AUC gap
        # this cell participates in, per crop mode -- flag if noise exceeds signal.
        same_crop = [
            s for s in summary
            if s["crop_mode"] == r["crop_mode"] and s["condition"] != r["condition"]
            and s.get("order_mode", "value") == "value"
        ]
        gaps = [
            abs(r["insertion_auc_scaled_mean"] - s["insertion_auc_scaled_mean"])
            for s in same_crop
            if r["insertion_auc_scaled_mean"] is not None and s["insertion_auc_scaled_mean"] is not None
        ]
        min_gap = min(gaps) if gaps else None
        if min_gap is not None and seed_std > min_gap:
            flagged_any = True
            lines.append(
                f"- **{r['condition']}/{r['crop_mode']}**: seed-to-seed std "
                f"({seed_std:.4f}) EXCEEDS the smallest between-condition gap "
                f"it's being compared against ({min_gap:.4f}) — this "
                "conclusion is NOT supported by seed replication, flagged not smoothed over."
            )
    if not flagged_any:
        lines.append("No condition's seed-to-seed std exceeded the between-condition difference it supports.")
    lines.append("")

    lines.append("## Direction of evidence\n")
    for comparison in COMPARISONS:
        rel = [r for r in stats_rows if r["comparison"] == f"P_bin_vs_{comparison}" and r["metric"] == "insertion_auc_scaled"]
        if not rel:
            continue
        r = rel[0]
        direction = "higher" if r["mean_diff"] > 0 else "lower"
        sig = "significantly" if r["p_holm"] < 0.05 else "not significantly"
        lines.append(
            f"- P_bin's insertion-AUC is **{direction}** than {comparison}'s "
            f"({sig} different, p_holm={r['p_holm']:.4g}, r={r['rank_biserial']:.3f})."
        )
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


# Rank-based ablation: how faithfulness changes for the 2nd/3rd most-
# activating concept per generated token, not just the top-1 concept used
# everywhere else in this report. run_full_pipeline.py's step 7 already
# computes ranks 1-3 for every config (per the spec's own per_image_auc.csv
# schema, sec 8.1: "rank in {1,2,3}") -- this was previously left unused
# beyond rank 1. Restricted to the 3 main, fully seed-matched conditions
# (P_null, P_open, P_bin) since those are the only ones with clean 5-seed
# coverage on both crop modes.
RANK_ABLATION_CONDITIONS = ["P_null", "P_open", "P_bin"]
RANK_ABLATION_RANKS = [1, 2, 3]


def assemble_rank_ablation_rows() -> List[dict]:
    """Rank comparisons (rank1 vs rank2 vs rank3, within one condition/crop_mode)
    are a fundamentally different comparison than condition-vs-condition: the
    rank-1 concept is, by construction, the most strongly activating one for
    a token, so it starts from a genuinely higher baseline probability than
    rank-2/3's -- pooling one shared min/max across ranks (the earlier
    approach here) let that baseline difference dominate, unfairly rewarding/
    penalizing a rank for how large its own concept's absolute weight happens
    to be rather than how faithfully deleting its coordinates crashes ITS OWN
    confidence (confirmed by inspecting P_bin_sliding_window_seed1's actual
    curves: rank1 y0=4.90e-6 vs rank2 y0=4.28e-6/rank3 y0=4.20e-6, all
    converging to nearly the same floor -- so raw/pooled-scaled AUC is
    dominated by each rank's own starting point, not by curve shape).
    Uses auc_start_relative instead (eval/concept_deletion_eval.py's
    _curve_auc_start_relative): each image's own curve divided by its own
    fraction=0 value before integrating -- already a fair, rank-comparable
    quantity per image, no further pooling/scaling needed. Only present in
    per_image CSVs regenerated after this fix; older ones yield None here
    until their eval step is re-run."""
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
                ins_auc, ins_rel, ins_start_rel = ins.get(key, (None, None, None))
                del_auc, del_rel, del_start_rel = dele.get(key, (None, None, None))
                rows.append({
                    "image_path": key, "condition": condition, "crop_mode": crop_mode,
                    "seed": seed, "rank": rank,
                    "insertion_auc": ins_auc, "insertion_auc_relative": ins_rel,
                    "insertion_auc_start_relative": ins_start_rel,
                    "deletion_auc": del_auc, "deletion_auc_relative": del_rel,
                    "deletion_auc_start_relative": del_start_rel,
                })
    return rows


def build_rank_ablation_summary(rows: List[dict]) -> List[dict]:
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["condition"], r["crop_mode"], r["rank"])].append(r)

    summary = []
    for (condition, crop_mode, rank), grp in grouped.items():
        ins_vals = [r["insertion_auc_start_relative"] for r in grp if r["insertion_auc_start_relative"] is not None]
        del_vals = [r["deletion_auc_start_relative"] for r in grp if r["deletion_auc_start_relative"] is not None]
        ins_rel_vals = [r["insertion_auc_relative"] for r in grp if r["insertion_auc_relative"] is not None]
        del_rel_vals = [r["deletion_auc_relative"] for r in grp if r["deletion_auc_relative"] is not None]
        summary.append({
            "condition": condition, "crop_mode": crop_mode, "rank": rank,
            "n_images": len(grp),
            "insertion_auc_start_relative_mean": float(np.mean(ins_vals)) if ins_vals else None,
            "insertion_auc_start_relative_std": float(np.std(ins_vals)) if ins_vals else None,
            "deletion_auc_start_relative_mean": float(np.mean(del_vals)) if del_vals else None,
            "deletion_auc_start_relative_std": float(np.std(del_vals)) if del_vals else None,
            "insertion_auc_relative_mean": float(np.mean(ins_rel_vals)) if ins_rel_vals else None,
            "insertion_auc_relative_std": float(np.std(ins_rel_vals)) if ins_rel_vals else None,
            "deletion_auc_relative_mean": float(np.mean(del_rel_vals)) if del_rel_vals else None,
            "deletion_auc_relative_std": float(np.std(del_rel_vals)) if del_rel_vals else None,
        })
    summary.sort(key=lambda r: (r["condition"], r["crop_mode"], r["rank"]))
    return summary


def write_rank_ablation_csv(summary: List[dict], out_path: Path) -> None:
    cols = ["condition", "crop_mode", "rank", "n_images",
            "insertion_auc_start_relative_mean", "insertion_auc_start_relative_std",
            "deletion_auc_start_relative_mean", "deletion_auc_start_relative_std",
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
    write_per_image_csv(rows, out_dir / "per_image_auc.csv")

    summary = build_summary(rows)
    write_summary_csv(summary, out_dir / "summary.csv")

    stats_rows = build_stats(rows)
    write_stats_csv(stats_rows, out_dir / "stats.csv")

    rank_rows = assemble_rank_ablation_rows()
    rank_summary = build_rank_ablation_summary(rank_rows)
    write_rank_ablation_csv(rank_summary, out_dir / "rank_ablation.csv")

    render_results_md(summary, stats_rows, out_dir / "RESULTS.md")


if __name__ == "__main__":
    main()
