#!/usr/bin/env python3
"""
Report for a coco10 pipeline run (outputs/coco10_pilot or outputs/coco10).

AUC is reported as mean +/- std across ranks 1-3 (plus the raw per-rank
breakdown), using the relative (min-max rescaled) AUC -- see
eval/concept_curve_auc_eval.py::_compute_auc_relative.

Usage:
    python scripts/coco10_report.py                      # outputs/coco10_pilot
    python scripts/coco10_report.py --run-dir outputs/coco10
"""

import argparse
import csv
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT_DIR = Path(__file__).parent.parent.resolve()
METHOD = "snmf"


def _orient_atoms(C: torch.Tensor) -> torch.Tensor:
    return C if C.size(0) <= C.size(1) else C.T.contiguous()


def _load_concepts(path: Path) -> Optional[torch.Tensor]:
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu")
    return _orient_atoms(torch.as_tensor(blob["concepts"], dtype=torch.float32))


def hoyer_selectivity(C: torch.Tensor) -> float:
    eps = 1e-12
    n = C.size(1)
    sqrt_n = torch.sqrt(torch.tensor(float(n)))
    l1 = C.abs().sum(dim=1)
    l2 = torch.linalg.norm(C, dim=1) + eps
    s = (sqrt_n - (l1 / l2)) / (sqrt_n - 1.0 + eps)
    return s.clamp_(0.0, 1.0).mean().item()


def concept_overlap(C: torch.Tensor, threshold: float = 1e-3) -> float:
    C_thr = C.clone()
    C_thr[C_thr.abs() < threshold] = 0.0
    Cn = C_thr / (C_thr.norm(dim=1, keepdim=True) + 1e-12)
    sim = Cn @ Cn.T
    n = sim.size(0)
    if n <= 1:
        return 0.0
    mask = ~torch.eye(n, dtype=torch.bool)
    return sim[mask].mean().item()


def find_concept_bank(run_dir: Path) -> Optional[Path]:
    matches = glob.glob(str(run_dir / "concept" / METHOD / f"combined_concept_{METHOD}_*_raw.pth"))
    return Path(matches[0]) if matches else None


def read_csv_rows(path: Path) -> list:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def compute_row(run_dir: Path) -> dict:
    row = {"run_dir": str(run_dir)}

    bank_path = find_concept_bank(run_dir)
    C = _load_concepts(bank_path) if bank_path else None
    row["n_concepts"] = C.shape[0] if C is not None else None
    row["sparsity_higher_better"] = hoyer_selectivity(C) if C is not None else None
    row["overlap_lower_better"] = concept_overlap(C) if C is not None else None

    bc_rows = read_csv_rows(run_dir / "eval" / METHOD / "clip_bert_topk_table.csv")
    if bc_rows:
        row["bert_score"] = float(np.mean([float(r["bert_mean"]) for r in bc_rows]))
        row["clip_score"] = float(np.mean([float(r["clip_mean"]) for r in bc_rows]))
    else:
        row["bert_score"] = row["clip_score"] = None

    auc_rows = read_csv_rows(run_dir / "eval" / METHOD / "concept_curve_auc_token_table.csv")
    if auc_rows:
        ins = [float(r["insertion_auc_relative"]) for r in auc_rows if r.get("insertion_auc_relative")]
        dele = [float(r["deletion_auc_relative"]) for r in auc_rows if r.get("deletion_auc_relative")]
        row["addition_auc_mean"] = float(np.mean(ins)) if ins else None
        row["addition_auc_std"] = float(np.std(ins)) if ins else None
        row["deletion_auc_mean"] = float(np.mean(dele)) if dele else None
        row["deletion_auc_std"] = float(np.std(dele)) if dele else None
        for i, r in enumerate(auc_rows, start=1):
            row[f"addition_auc_rank{i}"] = float(r["insertion_auc_relative"]) if r.get("insertion_auc_relative") else None
            row[f"deletion_auc_rank{i}"] = float(r["deletion_auc_relative"]) if r.get("deletion_auc_relative") else None
    else:
        row["addition_auc_mean"] = row["addition_auc_std"] = None
        row["deletion_auc_mean"] = row["deletion_auc_std"] = None
        for i in (1, 2, 3):
            row[f"addition_auc_rank{i}"] = row[f"deletion_auc_rank{i}"] = None

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=str(ROOT_DIR / "outputs" / "coco10_pilot"))
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    row = compute_row(run_dir)

    columns = [
        "run_dir", "n_concepts", "bert_score", "clip_score",
        "addition_auc_mean", "addition_auc_std",
        "deletion_auc_mean", "deletion_auc_std",
        "addition_auc_rank1", "addition_auc_rank2", "addition_auc_rank3",
        "deletion_auc_rank1", "deletion_auc_rank2", "deletion_auc_rank3",
        "sparsity_higher_better", "overlap_lower_better",
    ]

    def fmt(v):
        if v is None:
            return "N/A"
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    for c in columns:
        print(f"{c:28s}: {fmt(row[c])}")

    out_path = Path(args.out_csv) if args.out_csv else (run_dir / "report.csv")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
