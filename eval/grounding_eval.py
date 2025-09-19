#!/usr/bin/env python3
"""
Grounding Evaluation for vlm_prompt_grounding_explainer outputs.

Reads the JSON produced by inference/vlm_prompt_grounding_explainer.py and computes simple
summary metrics per image, saving a CSV for plotting:
- num_objects: number of grounded objects
- has_gt_match: if ground_truth label provided, whether any grounded object name matches (case-insensitive substring)
- names: semicolon-joined names (for reference)

Also produces an aggregate JSON with counts per object name and mean objects per image.

Usage:
python -m eval.grounding_eval --results_json outputs/vlm_groundings.json --out_dir outputs/grounding_eval
"""
from __future__ import annotations

import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


def normalize_name(s: str) -> str:
    s = (s or "").strip().lower()
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate grounding JSON outputs")
    ap.add_argument("--results_json", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    in_path = Path(args.results_json)
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(in_path.read_text())
    results: List[Dict[str, Any]] = payload.get("results", payload)

    rows: List[Dict[str, Any]] = []
    name_counts: Dict[str, int] = {}
    match_count = 0
    for item in results:
        objs = item.get("grounded_objects") or []
        names = [normalize_name(o.get("name", "")) for o in objs if isinstance(o, dict)]
        for n in names:
            if not n:
                continue
            name_counts[n] = name_counts.get(n, 0) + 1
        num_objects = len(names)
        gt = normalize_name(item.get("ground_truth") or "")
        has_gt_match = False
        if gt:
            has_gt_match = any((gt in n) or (n in gt) for n in names if n)
            if has_gt_match:
                match_count += 1
        rows.append({
            "image_path": item.get("image_path"),
            "num_objects": num_objects,
            "has_gt_match": int(has_gt_match),
            "names": ";".join([n for n in names if n]),
        })

    # Write CSV
    csv_path = out_dir / "grounding_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "num_objects", "has_gt_match", "names"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregate JSON
    agg = {
        "images": len(results),
        "mean_objects_per_image": (sum(r["num_objects"] for r in rows) / max(1, len(rows))),
        "gt_match_rate": (match_count / max(1, len(rows))),
        "name_counts": dict(sorted(name_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))

    # Prepare a simple curve CSV similar in structure to concept deletion for plotting reuse
    # Here we create a pseudo-curve over fraction of objects kept (0..1) vs avg kept per image.
    # This is a placeholder to allow a common plotting utility.
    curve_path = out_dir / "grounding_curve.csv"
    if rows:
        import numpy as np
        counts = [r["num_objects"] for r in rows]
        max_c = max(counts)
        xs = np.linspace(0, 1, 21)
        with curve_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fraction_kept", "mean_objects"])
            for x in xs:
                k = int(round(x * max_c))
                # assume we keep min(count, k) objects per image
                kept = [min(c, k) for c in counts]
                w.writerow([x, float(sum(kept)) / max(1, len(kept))])

    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
