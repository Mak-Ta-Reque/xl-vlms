#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _load_curve_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_auc(payload: Optional[Dict[str, Any]]) -> Optional[float]:
    if payload is None:
        return None

    fractions = payload.get("fractions", [])
    means = payload.get("mean", [])
    if not fractions or not means:
        return None
    if len(fractions) != len(means):
        return None

    x = np.asarray(fractions, dtype=np.float64)
    y = np.asarray(means, dtype=np.float64)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if x.size < 2:
        return float(y[0]) if x.size == 1 else None

    trapezoid = getattr(np, "trapezoid", np.trapz)
    auc = float(trapezoid(y, x))
    return auc


def _compute_auc_normalized(auc: Optional[float], payload: Optional[Dict[str, Any]]) -> Optional[float]:
    if auc is None or payload is None:
        return None

    fractions = payload.get("fractions", [])
    if not fractions:
        return None

    x = np.asarray(fractions, dtype=np.float64)
    span = float(np.max(x) - np.min(x))
    if span <= 0.0:
        return None

    return float(auc / span)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute AUC summary for concept insertion/deletion curves.")
    parser.add_argument("--out_dir", required=True, help="Directory containing c_insertion/c_deletion JSON curve files")
    parser.add_argument("--top_n", type=int, required=True, help="Maximum rank to summarize (1..TOP_N)")
    parser.add_argument("--mode", default="token", choices=["token", "sequence"], help="Curve mode suffix used in file names")
    parser.add_argument("--output_prefix", default="concept_curve_auc", help="Prefix for output CSV/JSON")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for rank in range(1, max(1, args.top_n) + 1):
        ins_path = out_dir / f"c_insertion_{args.mode}_rank{rank}.json"
        del_path = out_dir / f"c_deletion_{args.mode}_rank{rank}.json"

        ins_payload = _load_curve_json(ins_path)
        del_payload = _load_curve_json(del_path)

        ins_auc = _compute_auc(ins_payload)
        del_auc = _compute_auc(del_payload)

        row = {
            "rank": rank,
            "insertion_curve_json": str(ins_path),
            "deletion_curve_json": str(del_path),
            "insertion_auc": ins_auc,
            "deletion_auc": del_auc,
            "insertion_auc_norm": _compute_auc_normalized(ins_auc, ins_payload),
            "deletion_auc_norm": _compute_auc_normalized(del_auc, del_payload),
            "insertion_exists": ins_payload is not None,
            "deletion_exists": del_payload is not None,
        }
        rows.append(row)

    out_json = out_dir / f"{args.output_prefix}_table.json"
    out_csv = out_dir / f"{args.output_prefix}_table.csv"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "top_n": args.top_n,
                "mode": args.mode,
                "rows": rows,
            },
            f,
            indent=2,
        )

    headers = [
        "rank",
        "insertion_auc",
        "deletion_auc",
        "insertion_auc_norm",
        "deletion_auc_norm",
        "insertion_exists",
        "deletion_exists",
        "insertion_curve_json",
        "deletion_curve_json",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved AUC JSON: {out_json}")
    print(f"Saved AUC CSV: {out_csv}")


if __name__ == "__main__":
    main()
