#!/usr/bin/env python3
"""
Aggregate ablation results (see run_ablation.py) into one comparison table.

For each (prompt_template, crop_mode) config, reads the seed=42 run's:
  - BERT/CLIP top-k table (eval/snmf/clip_bert_topk_table.csv)
  - Faithfulness AUC table (eval/snmf/concept_curve_auc_token_table.csv)
and computes directly from the concept bank (concept/snmf/combined_concept_snmf_*_raw.pth):
  - sparsity (Hoyer selectivity, higher = better)
  - overlap (mean off-diagonal cosine similarity between concept vectors, lower = better)
  - instability (1 - mean matched-cosine similarity between the seed=42 and
    seed=43 concept banks for the same config, lower = better)

Sparsity/overlap/instability only depend on the learned concept dictionary
itself, so they're computed identically regardless of whether the run used
crops or whole images.

Usage:
    python scripts/ablation_report.py
    python scripts/ablation_report.py --out_csv outputs/ablation/report.csv
"""

import argparse
import csv
import glob
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(Path(__file__).parent))
# Single source of truth for the config matrix + naming convention, shared
# with the script that actually runs the ablation.
from run_ablation import (  # noqa: E402
    PROMPT_TEMPLATES,
    CROP_MODES,
    DECOMP_STRATEGIES,
    default_decomp_strategy,
    config_name,
)

PRIMARY_SEED = 42
REPEAT_SEED = 43
METHOD = "snmf"

# Must match DATASET_PRESETS in run_ablation.py.
DATASET_ABLATION_ROOTS = {
    "imagenet10": "outputs/ablation",
    "coco10": "outputs/ablation_coco10",
    # Same 16 configs, evaluated on single-object crops (val_masked) instead
    # of grids (val_grids) -- see run_ablation_singleobj.py.
    "coco10_singleobj": "outputs/ablation_coco10_singleobj",
}


# ---------- concept-bank metrics (validated earlier in this session) ----------

def _orient_atoms(C: torch.Tensor) -> torch.Tensor:
    return C if C.size(0) <= C.size(1) else C.T.contiguous()


def _load_concepts(path: Path) -> Optional[torch.Tensor]:
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu")
    C = torch.as_tensor(blob["concepts"], dtype=torch.float32)
    if C.ndim != 2 or C.numel() == 0:
        # Degenerate/empty bank (e.g. a config where 0 concepts passed the
        # CLEAN_EXAMPLE_RATIO purity filter still gets an empty file saved to
        # disk, shape torch.Size([0])) -- report as missing, don't crash.
        return None
    return _orient_atoms(C)


def hoyer_selectivity(C: torch.Tensor) -> float:
    """Higher = more concentrated/sparse atoms."""
    eps = 1e-12
    n = C.size(1)
    sqrt_n = torch.sqrt(torch.tensor(float(n)))
    l1 = C.abs().sum(dim=1)
    l2 = torch.linalg.norm(C, dim=1) + eps
    s = (sqrt_n - (l1 / l2)) / (sqrt_n - 1.0 + eps)
    return s.clamp_(0.0, 1.0).mean().item()


def atom_sparsity_relative(C: torch.Tensor, threshold: float = 1e-2) -> float:
    """Fraction of near-zero entries per atom, averaged across atoms.

    threshold is RELATIVE to each atom's own max |value| (not an absolute
    cutoff): concept vector scales vary a lot across decomposition methods, so
    an absolute threshold can silently flag ~100% or ~0% of entries regardless
    of real structure. Same convention as notebooks/CAV_eval.ipynb's
    compute_atom_sparsity.
    """
    atom_scale = C.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12)
    near_zero = C.abs() < threshold * atom_scale
    return near_zero.float().mean(dim=1).mean().item()


def concept_overlap(C: torch.Tensor, threshold: float = 1e-3) -> float:
    """Mean off-diagonal cosine similarity between atoms. Lower = more disentangled."""
    C_thr = C.clone()
    C_thr[C_thr.abs() < threshold] = 0.0
    Cn = C_thr / (C_thr.norm(dim=1, keepdim=True) + 1e-12)
    sim = Cn @ Cn.T
    n = sim.size(0)
    if n <= 1:
        return 0.0
    mask = ~torch.eye(n, dtype=torch.bool)
    return sim[mask].mean().item()


def matched_cosine_similarity(V1: np.ndarray, V2: np.ndarray) -> float:
    """Mean matched cosine similarity between two atom banks via Hungarian
    matching on the full rectangular similarity matrix (no truncation)."""
    V1n = V1 / (np.linalg.norm(V1, axis=1, keepdims=True) + 1e-12)
    V2n = V2 / (np.linalg.norm(V2, axis=1, keepdims=True) + 1e-12)
    S = V1n @ V2n.T
    row_ind, col_ind = linear_sum_assignment(-S)
    return float(S[row_ind, col_ind].mean())


# ---------- per-config result assembly ----------

def config_dir(ablation_root: Path, prompt_template: str, crop_mode: str, decomp_strategy: str, seed: int) -> Path:
    return ablation_root / config_name(prompt_template, crop_mode, decomp_strategy, seed)


def find_concept_bank(run_dir: Path) -> Optional[Path]:
    matches = glob.glob(str(run_dir / "concept" / METHOD / f"combined_concept_{METHOD}_*_raw.pth"))
    return Path(matches[0]) if matches else None


def read_csv_mean(path: Path, columns: list) -> dict:
    """Average each named column across all rows of a CSV; {} if missing."""
    if not path.exists():
        return {}
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for col in columns:
        vals = [float(r[col]) for r in rows if col in r and r[col] not in (None, "")]
        out[col] = float(np.mean(vals)) if vals else None
    return out


def read_csv_per_rank(path: Path, columns: list, rank_col: str = "rank") -> dict:
    """Return {rank: {col: value}} for each row, keyed by its own rank column."""
    if not path.exists():
        return {}
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        try:
            rank = int(r[rank_col])
        except (KeyError, ValueError):
            continue
        out[rank] = {
            col: (float(r[col]) if col in r and r[col] not in (None, "") else None)
            for col in columns
        }
    return out


def compute_row(ablation_root: Path, prompt_template: str, crop_mode: str, decomp_strategy: str) -> dict:
    primary_dir = config_dir(ablation_root, prompt_template, crop_mode, decomp_strategy, PRIMARY_SEED)
    repeat_dir = config_dir(ablation_root, prompt_template, crop_mode, decomp_strategy, REPEAT_SEED)

    row = {"prompt_template": prompt_template, "crop_mode": crop_mode, "decomp_strategy": decomp_strategy}

    bank_path = find_concept_bank(primary_dir)
    C = _load_concepts(bank_path) if bank_path else None
    row["sparsity_higher_better"] = hoyer_selectivity(C) if C is not None else None
    row["sparsity_frac_near_zero_1pct"] = atom_sparsity_relative(C) if C is not None else None
    row["overlap_lower_better"] = concept_overlap(C) if C is not None else None

    repeat_bank_path = find_concept_bank(repeat_dir)
    C2 = _load_concepts(repeat_bank_path) if repeat_bank_path else None
    if C is not None and C2 is not None:
        sim = matched_cosine_similarity(C.numpy(), C2.numpy())
        row["instability_lower_better"] = 1.0 - sim
    else:
        row["instability_lower_better"] = None

    bert_clip = read_csv_mean(
        primary_dir / "eval" / METHOD / "clip_bert_topk_table.csv",
        ["bert_mean", "clip_mean", "random_bert_mean", "random_clip_mean"],
    )
    row["bert_score"] = bert_clip.get("bert_mean")
    row["clip_score"] = bert_clip.get("clip_mean")
    row["random_bert_score"] = bert_clip.get("random_bert_mean")
    row["random_clip_score"] = bert_clip.get("random_clip_mean")

    # Per-rank bert/clip (real + random-direction baseline), each with its own
    # std -- clip_bert_topk_table.csv already reports mean+std per rank_k, so
    # unlike AUC's cross-rank std below, these stds are real (across images).
    bert_clip_by_rank = read_csv_per_rank(
        primary_dir / "eval" / METHOD / "clip_bert_topk_table.csv",
        ["bert_mean", "bert_std", "clip_mean", "clip_std",
         "random_bert_mean", "random_bert_std", "random_clip_mean", "random_clip_std"],
        rank_col="rank_k",
    )
    for rank in (1, 2, 3):
        vals = bert_clip_by_rank.get(rank, {})
        row[f"bert_rank{rank}"] = vals.get("bert_mean")
        row[f"bert_std_rank{rank}"] = vals.get("bert_std")
        row[f"clip_rank{rank}"] = vals.get("clip_mean")
        row[f"clip_std_rank{rank}"] = vals.get("clip_std")
        row[f"random_bert_rank{rank}"] = vals.get("random_bert_mean")
        row[f"random_clip_rank{rank}"] = vals.get("random_clip_mean")

    # Relative AUC (min-max rescaled to the curve's own [0,1] range before
    # integrating; see eval/concept_curve_auc_eval.py::_compute_auc_relative).
    # The raw AUC is ~1e-6 (dominated by vocab-size scaling) and not
    # meaningfully comparable across configs; the relative version is.
    # Reported per-rank (top-1/2/3 activated concept), not just averaged.
    auc_by_rank = read_csv_per_rank(
        primary_dir / "eval" / METHOD / "concept_curve_auc_token_table.csv",
        ["insertion_auc_relative", "deletion_auc_relative",
         "insertion_auc_relative_random", "deletion_auc_relative_random"],
    )
    for rank in (1, 2, 3):
        vals = auc_by_rank.get(rank, {})
        row[f"addition_auc_rank{rank}"] = vals.get("insertion_auc_relative")
        row[f"deletion_auc_rank{rank}"] = vals.get("deletion_auc_relative")
        # Random-order baseline (chance-level faithfulness) -- None if that
        # pass wasn't run for this config (older results predating this
        # feature), not an error.
        row[f"addition_auc_rank{rank}_random"] = vals.get("insertion_auc_relative_random")
        row[f"deletion_auc_rank{rank}_random"] = vals.get("deletion_auc_relative_random")

    # Mean +/- std across ranks 1-3, alongside the per-rank breakdown above.
    ins_vals = [row[f"addition_auc_rank{r}"] for r in (1, 2, 3) if row[f"addition_auc_rank{r}"] is not None]
    del_vals = [row[f"deletion_auc_rank{r}"] for r in (1, 2, 3) if row[f"deletion_auc_rank{r}"] is not None]
    row["addition_auc_mean"] = float(np.mean(ins_vals)) if ins_vals else None
    row["addition_auc_std"] = float(np.std(ins_vals)) if ins_vals else None
    row["deletion_auc_mean"] = float(np.mean(del_vals)) if del_vals else None
    row["deletion_auc_std"] = float(np.std(del_vals)) if del_vals else None

    ins_rand_vals = [row[f"addition_auc_rank{r}_random"] for r in (1, 2, 3) if row[f"addition_auc_rank{r}_random"] is not None]
    del_rand_vals = [row[f"deletion_auc_rank{r}_random"] for r in (1, 2, 3) if row[f"deletion_auc_rank{r}_random"] is not None]
    row["addition_auc_mean_random"] = float(np.mean(ins_rand_vals)) if ins_rand_vals else None
    row["addition_auc_std_random"] = float(np.std(ins_rand_vals)) if ins_rand_vals else None
    row["deletion_auc_mean_random"] = float(np.mean(del_rand_vals)) if del_rand_vals else None
    row["deletion_auc_std_random"] = float(np.std(del_rand_vals)) if del_rand_vals else None

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASET_ABLATION_ROOTS.keys()), default="imagenet10")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument(
        "--decomp-strategies",
        choices=["default", "all"],
        default="default",
        help="Must match the --decomp-strategies value used to run the ablation: "
             "'default' (18 rows, one per prompt_template x crop_mode combo, each "
             "at its own default strategy) or 'all' (36 rows, every combo at both "
             "per_tag and pooled).",
    )
    args = parser.parse_args()

    ablation_root = ROOT_DIR / DATASET_ABLATION_ROOTS[args.dataset]
    out_csv = args.out_csv or str(ablation_root / "ablation_report.csv")

    strategies_for = DECOMP_STRATEGIES if args.decomp_strategies == "all" else None

    rows = []
    for prompt_template in PROMPT_TEMPLATES:
        for crop_mode in CROP_MODES:
            for decomp_strategy in (strategies_for or [default_decomp_strategy(prompt_template)]):
                rows.append(compute_row(ablation_root, prompt_template, crop_mode, decomp_strategy))

    columns = [
        "prompt_template", "crop_mode", "decomp_strategy",
        "bert_score", "clip_score", "random_bert_score", "random_clip_score",
        "bert_rank1", "bert_rank2", "bert_rank3",
        "bert_std_rank1", "bert_std_rank2", "bert_std_rank3",
        "clip_rank1", "clip_rank2", "clip_rank3",
        "clip_std_rank1", "clip_std_rank2", "clip_std_rank3",
        "random_bert_rank1", "random_bert_rank2", "random_bert_rank3",
        "random_clip_rank1", "random_clip_rank2", "random_clip_rank3",
        "addition_auc_mean", "addition_auc_std",
        "deletion_auc_mean", "deletion_auc_std",
        "addition_auc_rank1", "addition_auc_rank2", "addition_auc_rank3",
        "deletion_auc_rank1", "deletion_auc_rank2", "deletion_auc_rank3",
        "addition_auc_mean_random", "addition_auc_std_random",
        "deletion_auc_mean_random", "deletion_auc_std_random",
        "addition_auc_rank1_random", "addition_auc_rank2_random", "addition_auc_rank3_random",
        "deletion_auc_rank1_random", "deletion_auc_rank2_random", "deletion_auc_rank3_random",
        "sparsity_higher_better", "sparsity_frac_near_zero_1pct",
        "instability_lower_better", "overlap_lower_better",
    ]

    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            # AUC values are ~1e-6; .4f would silently round them to 0.0000.
            return f"{v:.4g}"
        return str(v)

    widths = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(fmt(r[c]).ljust(widths[c]) for c in columns))

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
