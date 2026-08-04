#!/usr/bin/env python3
"""
Consolidated reporting for the 10-class rebuttal ablation
(outputs/rebuttal_ablation_10class), pulling together everything needed to
compare all 5 conditions (P_bin, P_bin_fullpool, P_bin_shuf, P_null, P_open)
on one basis:

  - insertion/deletion AUC (log/linear/shift, from the validated raw-curve
    pipeline: scripts/gen_raw_prob_curves.py + scripts/posthoc_auc_curves.py
    -- NOT the older truncated NUM_POINTS=70 per-run CSVs that
    scripts/rebuttal_stats_10class.py reads),
  - per-rank (1/2/3) "ranking AUC" for all 5 conditions,
  - unsupervised concept-vector quality: sparsity (Hoyer selectivity,
    fraction near-zero), overlap, seed-to-seed instability
    (reusing scripts/ablation_report.py's functions on the concept banks),
  - BERT-score + CLIP-score (from each config's already-computed
    eval/snmf/clip_bert_topk_table.csv),
  - paired significance tests (Wilcoxon signed-rank, bootstrap 95% CI,
    rank-biserial correlation, Holm-Bonferroni correction) comparing P_bin
    against each of the other 4 conditions, on the SAME log-AUC values
    reported in ranking_auc.csv/auc_summary.csv -- not a separately-scaled
    number.

Writes to outputs/rebuttal_ablation_10class/_report/:
  per_image_auc.csv    one row per (image, condition, crop_mode, seed, order),
                        rank=1 only (top-activated concept)
  ranking_auc.csv       per (condition, crop_mode, rank, kind, order) --
                        literally scripts/posthoc_auc_curves.py's table,
                        rewritten here so the full report lives in one place
  concept_quality.csv   per (condition, crop_mode): sparsity/overlap/instability
  clip_bert_summary.csv per (condition, crop_mode, rank): bert/clip score
  auc_summary.csv       per (condition, crop_mode, order): master table,
                        joins AUC + concept quality + clip/bert (rank=1)
  stats.csv             paired Wilcoxon/bootstrap/rank-biserial/Holm-Bonferroni

Usage:
    python scripts/rebuttal_report_10class.py
"""
import csv
import glob
import json
import sys
from collections import defaultdict
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
from posthoc_auc_curves import (  # noqa: E402
    parse_name,
    scale_curve,
    curve_auc,
    shift_curve_and_auc,
    load_all as posthoc_load_all,
    write_table as posthoc_write_table,
    CONDITIONS,
    COND_ORDER,
)
from run_rebuttal_ablation_10class import build_grid, config_name  # noqa: E402

ABLATION_ROOT = ROOT_DIR / "outputs" / "rebuttal_ablation_10class"
METHOD = "snmf"
OUT_DIR = ABLATION_ROOT / "_report"
COMPARISONS = ["P_open", "P_null", "P_bin_shuf", "P_bin_fullpool"]
SIG_CROP_MODES = ["sliding_window", "langsam"]
RANK1 = 1


def load_class_lookup() -> Dict[str, str]:
    manifest_path = ROOT_DIR / "data" / "coco10" / "eval_images_rebuttal10.json"
    if not manifest_path.exists():
        return {}
    manifest = json.load(open(manifest_path))
    lookup = {}
    for cls, paths in manifest.get("images", {}).items():
        for p in paths:
            lookup[Path(p).name] = cls
    return lookup


# ---------------------------------------------------------------------------
# per_image_auc.csv (rank=1, all 3 AUC flavors, for traceability + stats)
# ---------------------------------------------------------------------------

def load_per_image_rows(class_lookup: Dict[str, str]) -> List[dict]:
    by_key: Dict[Tuple, dict] = {}
    for d in sorted(ABLATION_ROOT.glob("P_*")):
        npz = d / "eval" / METHOD / "raw_prob_curves.npz"
        if not npz.exists():
            continue
        cond, crop, seed = parse_name(d.name)
        if cond is None:
            continue
        z = np.load(npz, allow_pickle=True)
        fk = f"fracs_r{RANK1}"
        if fk not in z:
            continue
        fracs = z[fk].astype(np.float64)
        for kind in ("ins", "del"):
            for order in ("value", "random"):
                ck = f"{kind}_{order}_r{RANK1}_curves"
                ik = f"{kind}_{order}_r{RANK1}_images"
                if ck not in z or ik not in z:
                    continue
                curves = z[ck].astype(np.float64)
                images = z[ik]
                for img, row in zip(images, curves):
                    sc = scale_curve(row, kind)
                    if sc is None:
                        continue
                    key = (str(img), cond, crop, seed, order)
                    entry = by_key.setdefault(key, {
                        "image_path": str(img),
                        "class": class_lookup.get(str(img)),
                        "condition": cond, "crop_mode": crop, "seed": seed,
                        "order_mode": order,
                    })
                    prefix = "insertion" if kind == "ins" else "deletion"
                    entry[f"{prefix}_auc_log"] = curve_auc(fracs, sc, log_scale=True)
                    entry[f"{prefix}_auc_linear"] = curve_auc(fracs, sc, log_scale=False)
                    entry[f"{prefix}_auc_shift"] = shift_curve_and_auc(row, fracs, kind)
    return list(by_key.values())


def write_per_image_csv(rows: List[dict], out_path: Path) -> None:
    cols = ["image_path", "class", "condition", "crop_mode", "seed", "order_mode",
            "insertion_auc_log", "insertion_auc_linear", "insertion_auc_shift",
            "deletion_auc_log", "deletion_auc_linear", "deletion_auc_shift"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])
    print(f"Wrote {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# ranking_auc.csv -- per (condition, crop_mode, rank, kind, order), all ranks
# ---------------------------------------------------------------------------

def load_posthoc_data():
    data, n_files = posthoc_load_all()
    print(f"[ranking_auc] loaded raw curves from {n_files} configs")
    return data


def write_ranking_auc_csv(data, out_path: Path) -> None:
    posthoc_write_table(data, out_path)


# ---------------------------------------------------------------------------
# shift_auc.csv -- dedicated table for the additive/shift AUC, isolated from
# the log/linear columns since it lives on a very different numeric scale
# (raw probability units, ~1e-7 to 1e-6) and is easy to misread buried next
# to log-AUC (~-0.2 to -0.01). Deletion is anchored so its curve starts at 1
# (k=0 = full concept vector, same anchor the ratio/log metrics use) and
# reported here as "1 - shift_auc" = the drop from that anchor; insertion is
# anchored so its curve starts at 0 (k=0 = blank/zero-vector state) and
# reported as-is = the climb from that anchor. Both directions are then
# directly comparable in the same units, bigger magnitude = more sensitive
# to the concept. See shift_curve_and_auc() in posthoc_auc_curves.py.
# ---------------------------------------------------------------------------

def write_shift_auc_csv(data, out_path: Path) -> None:
    """auc_shift_normalized: within each (crop_mode, rank, kind, order) group
    -- i.e. across the 5 conditions being directly compared -- divide by the
    group's own max |auc_shift_mean|, so the strongest condition reads 1.0
    and the rest read as a fraction of it. This is a single shared affine
    rescale per group (same denominator for every condition in the group),
    so it cannot change which condition wins or flip any comparison's sign
    -- purely a readability fix for auc_shift_mean's raw ~1e-7..1e-6 scale.
    Unlike per-curve (p_full - p_blank) normalization, the denominator here
    is a fixed constant per group, not a per-image quantity that can be
    near-zero -- so it carries none of that blow-up risk either."""
    cols = ["condition", "crop_mode", "rank", "kind", "order", "n_images",
            "auc_shift_mean", "auc_shift_std", "auc_shift_normalized"]
    rows = []
    for (cond, crop), inner in data.items():
        for (rank, kind, order), slot in inner.items():
            aucs_shift = np.asarray(slot["aucs_shift"], dtype=np.float64)
            if aucs_shift.size == 0:
                continue
            rows.append({
                "condition": cond, "crop_mode": crop, "rank": rank, "kind": kind, "order": order,
                "n_images": aucs_shift.size,
                "auc_shift_mean": float(aucs_shift.mean()), "auc_shift_std": float(aucs_shift.std()),
            })

    group_max: Dict[Tuple, float] = {}
    for r in rows:
        key = (r["crop_mode"], r["rank"], r["kind"], r["order"])
        group_max[key] = max(group_max.get(key, 0.0), abs(r["auc_shift_mean"]))
    for r in rows:
        key = (r["crop_mode"], r["rank"], r["kind"], r["order"])
        gmax = group_max[key]
        r["auc_shift_normalized"] = (r["auc_shift_mean"] / gmax) if gmax > 0 else None

    rows.sort(key=lambda r: (r["rank"], r["kind"], r["crop_mode"],
                             COND_ORDER.index(r["condition"]) if r["condition"] in COND_ORDER else 99,
                             r["order"]))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])
    print(f"Wrote {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# concept_quality.csv -- sparsity / overlap / seed-to-seed instability
# ---------------------------------------------------------------------------

def find_concept_bank(run_dir: Path) -> Optional[Path]:
    matches = glob.glob(str(run_dir / "concept" / METHOD / f"combined_concept_{METHOD}_*_raw.pth"))
    return Path(matches[0]) if matches else None


def build_concept_quality() -> List[dict]:
    grid = build_grid()
    by_cond_crop: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for cond, crop, seed in grid:
        by_cond_crop[(cond, crop)].append(seed)

    rows = []
    for (cond, crop), seeds in sorted(by_cond_crop.items()):
        seeds = sorted(seeds)
        banks = {}
        for seed in seeds:
            run_dir = ABLATION_ROOT / config_name(cond, crop, seed)
            bank_path = find_concept_bank(run_dir)
            C = _load_concepts(bank_path) if bank_path else None
            if C is not None:
                banks[seed] = C
        if not banks:
            continue
        sparsity_vals = [hoyer_selectivity(C) for C in banks.values()]
        nz_vals = [atom_sparsity_relative(C) for C in banks.values()]
        overlap_vals = [concept_overlap(C) for C in banks.values()]
        instability_vals = []
        seed_list = sorted(banks.keys())
        for i in range(len(seed_list)):
            for j in range(i + 1, len(seed_list)):
                sim = matched_cosine_similarity(banks[seed_list[i]].numpy(), banks[seed_list[j]].numpy())
                instability_vals.append(1.0 - sim)
        rows.append({
            "condition": cond, "crop_mode": crop, "n_seeds": len(banks),
            "sparsity_higher_better_mean": float(np.mean(sparsity_vals)),
            "sparsity_higher_better_std": float(np.std(sparsity_vals)) if len(sparsity_vals) > 1 else None,
            "sparsity_frac_near_zero_1pct_mean": float(np.mean(nz_vals)),
            "sparsity_frac_near_zero_1pct_std": float(np.std(nz_vals)) if len(nz_vals) > 1 else None,
            "overlap_lower_better_mean": float(np.mean(overlap_vals)),
            "overlap_lower_better_std": float(np.std(overlap_vals)) if len(overlap_vals) > 1 else None,
            "instability_lower_better_mean": float(np.mean(instability_vals)) if instability_vals else None,
            "instability_lower_better_std": float(np.std(instability_vals)) if len(instability_vals) > 1 else None,
        })
    rows.sort(key=lambda r: (COND_ORDER.index(r["condition"]) if r["condition"] in COND_ORDER else 99, r["crop_mode"]))
    return rows


def write_concept_quality_csv(rows: List[dict], out_path: Path) -> None:
    cols = ["condition", "crop_mode", "n_seeds",
            "sparsity_higher_better_mean", "sparsity_higher_better_std",
            "sparsity_frac_near_zero_1pct_mean", "sparsity_frac_near_zero_1pct_std",
            "overlap_lower_better_mean", "overlap_lower_better_std",
            "instability_lower_better_mean", "instability_lower_better_std"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])
    print(f"Wrote {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# clip_bert_summary.csv -- from each config's clip_bert_topk_table.csv
# ---------------------------------------------------------------------------

def build_clip_bert_summary() -> List[dict]:
    grouped = defaultdict(list)  # (cond, crop, rank) -> list of row dicts
    for cond, crop, seed in build_grid():
        run_dir = ABLATION_ROOT / config_name(cond, crop, seed)
        table_path = run_dir / "eval" / METHOD / "clip_bert_topk_table.csv"
        if not table_path.exists():
            continue
        with open(table_path) as f:
            for row in csv.DictReader(f):
                try:
                    rank_k = int(row["rank_k"])
                except (KeyError, ValueError):
                    continue
                grouped[(cond, crop, rank_k)].append(row)

    rows = []
    for (cond, crop, rank_k), grp in sorted(
        grouped.items(),
        key=lambda kv: (COND_ORDER.index(kv[0][0]) if kv[0][0] in COND_ORDER else 99, kv[0][1], kv[0][2]),
    ):
        def _col(name):
            vals = [float(r[name]) for r in grp if r.get(name) not in (None, "")]
            return vals

        bert_vals = _col("bert_mean")
        clip_vals = _col("clip_mean")
        rbert_vals = _col("random_bert_mean")
        rclip_vals = _col("random_clip_mean")
        rows.append({
            "condition": cond, "crop_mode": crop, "rank_k": rank_k,
            "n_configs_averaged": len(grp),
            "bert_mean_mean": float(np.mean(bert_vals)) if bert_vals else None,
            "bert_mean_std": float(np.std(bert_vals)) if len(bert_vals) > 1 else None,
            "clip_mean_mean": float(np.mean(clip_vals)) if clip_vals else None,
            "clip_mean_std": float(np.std(clip_vals)) if len(clip_vals) > 1 else None,
            "random_bert_mean_mean": float(np.mean(rbert_vals)) if rbert_vals else None,
            "random_clip_mean_mean": float(np.mean(rclip_vals)) if rclip_vals else None,
        })
    return rows


def write_clip_bert_summary_csv(rows: List[dict], out_path: Path) -> None:
    cols = ["condition", "crop_mode", "rank_k", "n_configs_averaged",
            "bert_mean_mean", "bert_mean_std", "clip_mean_mean", "clip_mean_std",
            "random_bert_mean_mean", "random_clip_mean_mean"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])
    print(f"Wrote {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# auc_summary.csv -- master table, rank=1, joins AUC + concept quality + clip/bert
# ---------------------------------------------------------------------------

def build_auc_summary(per_image_rows: List[dict], concept_quality: List[dict], clip_bert: List[dict]) -> List[dict]:
    cq_lookup = {(r["condition"], r["crop_mode"]): r for r in concept_quality}
    cb_lookup = {(r["condition"], r["crop_mode"]): r for r in clip_bert if r["rank_k"] == RANK1}

    grouped = defaultdict(list)
    for r in per_image_rows:
        grouped[(r["condition"], r["crop_mode"], r["order_mode"])].append(r)

    rows = []
    for (cond, crop, order_mode), grp in sorted(
        grouped.items(),
        key=lambda kv: (COND_ORDER.index(kv[0][0]) if kv[0][0] in COND_ORDER else 99, kv[0][1], kv[0][2]),
    ):
        def _agg(field):
            vals = [r[field] for r in grp if r.get(field) is not None]
            return (float(np.mean(vals)) if vals else None, float(np.std(vals)) if len(vals) > 1 else None)

        by_seed = defaultdict(list)
        for r in grp:
            if r.get("insertion_auc_log") is not None:
                by_seed[r["seed"]].append(r["insertion_auc_log"])
        seed_means = [float(np.mean(v)) for v in by_seed.values() if v]
        seed_std = float(np.std(seed_means)) if len(seed_means) > 1 else None

        row = {"condition": cond, "crop_mode": crop, "order_mode": order_mode,
               "n_images": len(grp), "n_seeds_present": len(by_seed)}
        for field in ["insertion_auc_log", "insertion_auc_linear", "insertion_auc_shift",
                      "deletion_auc_log", "deletion_auc_linear", "deletion_auc_shift"]:
            m, s = _agg(field)
            row[f"{field}_mean"] = m
            row[f"{field}_std"] = s
        row["seed_to_seed_std_insertion_auc_log"] = seed_std

        if order_mode == "value":
            cq = cq_lookup.get((cond, crop), {})
            row["sparsity_higher_better"] = cq.get("sparsity_higher_better_mean")
            row["overlap_lower_better"] = cq.get("overlap_lower_better_mean")
            row["instability_lower_better"] = cq.get("instability_lower_better_mean")
            cb = cb_lookup.get((cond, crop), {})
            row["bert_score_rank1"] = cb.get("bert_mean_mean")
            row["clip_score_rank1"] = cb.get("clip_mean_mean")
        else:
            row["sparsity_higher_better"] = None
            row["overlap_lower_better"] = None
            row["instability_lower_better"] = None
            row["bert_score_rank1"] = None
            row["clip_score_rank1"] = None

        rows.append(row)
    return rows


def write_auc_summary_csv(rows: List[dict], out_path: Path) -> None:
    cols = ["condition", "crop_mode", "order_mode", "n_images", "n_seeds_present",
            "insertion_auc_log_mean", "insertion_auc_log_std",
            "insertion_auc_linear_mean", "insertion_auc_linear_std",
            "insertion_auc_shift_mean", "insertion_auc_shift_std",
            "deletion_auc_log_mean", "deletion_auc_log_std",
            "deletion_auc_linear_mean", "deletion_auc_linear_std",
            "deletion_auc_shift_mean", "deletion_auc_shift_std",
            "seed_to_seed_std_insertion_auc_log",
            "sparsity_higher_better", "overlap_lower_better", "instability_lower_better",
            "bert_score_rank1", "clip_score_rank1"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])
    print(f"Wrote {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# stats.csv -- Wilcoxon / bootstrap CI / rank-biserial / Holm-Bonferroni,
# on insertion_auc_log / deletion_auc_log, paired by image
# ---------------------------------------------------------------------------

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
    per_image = defaultdict(list)
    for r in rows:
        if (r["condition"] == condition and r["crop_mode"] == crop_mode
                and r.get("order_mode", "value") == "value" and r.get(metric) is not None):
            per_image[r["image_path"]].append(r[metric])
    return {img: float(np.mean(vals)) for img, vals in per_image.items() if vals}


def build_stats(rows: List[dict]) -> List[dict]:
    stats_rows = []
    raw_pvals = []
    pending = []

    for crop_mode in SIG_CROP_MODES:
        for comparison in COMPARISONS:
            for metric in ["insertion_auc_log", "deletion_auc_log"]:
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
            entry["significant_holm_0.05"] = p_holm < 0.05
            stats_rows.append(entry)
    return stats_rows


def write_stats_csv(stats_rows: List[dict], out_path: Path) -> None:
    cols = ["crop_mode", "comparison", "metric", "n", "wilcoxon_stat",
            "p_raw", "p_holm", "significant_holm_0.05", "rank_biserial",
            "mean_diff", "ci_low", "ci_high"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in stats_rows:
            w.writerow([r.get(c) for c in cols])
    print(f"Wrote {out_path} ({len(stats_rows)} rows)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    class_lookup = load_class_lookup()

    per_image_rows = load_per_image_rows(class_lookup)
    write_per_image_csv(per_image_rows, OUT_DIR / "per_image_auc.csv")

    posthoc_data = load_posthoc_data()
    write_ranking_auc_csv(posthoc_data, OUT_DIR / "ranking_auc.csv")
    write_shift_auc_csv(posthoc_data, OUT_DIR / "shift_auc.csv")

    concept_quality = build_concept_quality()
    write_concept_quality_csv(concept_quality, OUT_DIR / "concept_quality.csv")

    clip_bert = build_clip_bert_summary()
    write_clip_bert_summary_csv(clip_bert, OUT_DIR / "clip_bert_summary.csv")

    auc_summary = build_auc_summary(per_image_rows, concept_quality, clip_bert)
    write_auc_summary_csv(auc_summary, OUT_DIR / "auc_summary.csv")

    stats_rows = build_stats(per_image_rows)
    write_stats_csv(stats_rows, OUT_DIR / "stats.csv")

    print(f"\nAll reports written to {OUT_DIR}")


if __name__ == "__main__":
    main()
