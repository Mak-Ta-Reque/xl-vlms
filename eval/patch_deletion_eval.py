#!/usr/bin/env python3
"""
Patch/pixel-space insertion-deletion evaluation for the attention-map and
GradCAM rebuttal baselines (eval/baseline_saliency.py). This is deliberately
a SEPARATE mechanism from eval/concept_deletion_eval.py's hidden-state-
dimension masking: attention/GradCAM produce a spatial saliency map over
image patches, not a direction in concept space, so insertion/deletion here
masks PIXEL PATCHES in the original image and re-runs the frozen VLM's
forward pass, tracking the same fixed target token's probability -- the
standard way these two baselines are evaluated in the saliency literature.

Runs once each (attention-map, gradcam) over the fixed apple/cat/bird eval
set (data/coco10/val_masked_rebuttal), not per crop-mode/seed like the main
45-run concept-decomposition grid.

Usage:
    python eval/patch_deletion_eval.py --method gradcam \
        --image_root data/coco10/val_masked_rebuttal \
        --prompt "What are the objects in the image?" \
        --patch_grid 8 --num_points 10 --out_dir outputs/rebuttal_ablation/gradcam_baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import gc

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(_ROOT / "eval"))

from baseline_saliency import (  # noqa: E402
    compute_saliency,
    load_model_and_processor,
    pool_to_patch_grid,
)


def _iter_images(image_root: Path):
    for cls_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                yield cls_dir.name, f


def _masked_image(image: Image.Image, patch_grid: int, keep_mask: np.ndarray, fill=(127, 127, 127)) -> Image.Image:
    """keep_mask: (patch_grid, patch_grid) bool, True = keep original pixels,
    False = replace with a flat gray fill (the standard occlusion baseline)."""
    w, h = image.size
    out = image.copy()
    px = out.load()
    pw = w / patch_grid
    ph = h / patch_grid
    for gy in range(patch_grid):
        for gx in range(patch_grid):
            if keep_mask[gy, gx]:
                continue
            x0, x1 = int(gx * pw), int((gx + 1) * pw) if gx < patch_grid - 1 else w
            y0, y1 = int(gy * ph), int((gy + 1) * ph) if gy < patch_grid - 1 else h
            for y in range(y0, y1):
                for x in range(x0, x1):
                    px[x, y] = fill
    return out


def _target_prob(model, processor, image: Image.Image, prompt: str, target_id: int) -> float:
    device = next(model.parameters()).device
    conversation = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]
    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1]
        prob = torch.softmax(logits.float(), dim=-1)[target_id]
    return float(prob.item())


def _build_ks(n_patches: int, num_points: int) -> List[int]:
    if num_points <= 1 or n_patches == 0:
        return [0, n_patches]
    xs = np.linspace(0, n_patches, int(num_points))
    ks = sorted(set(int(round(x)) for x in xs))
    if ks[0] != 0:
        ks.insert(0, 0)
    if ks[-1] != n_patches:
        ks.append(n_patches)
    return ks


def _curve_auc_relative(fracs: np.ndarray, y: np.ndarray) -> Optional[float]:
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if y_max <= y_min:
        return None
    y_rel = (y - y_min) / (y_max - y_min)
    span = float(np.max(fracs) - np.min(fracs))
    if span <= 0:
        return None
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y_rel, fracs)) / span


def run(args: argparse.Namespace) -> None:
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model ({args.devices})...")
    model, processor, _ = load_model_and_processor(args.devices)
    print("Loaded.")

    n_patches = args.patch_grid * args.patch_grid
    ks = _build_ks(n_patches, args.num_points)
    fracs = np.array([k / float(n_patches) for k in ks], dtype=np.float64)

    insertion_curves: List[List[float]] = []
    deletion_curves: List[List[float]] = []
    per_image_insertion: List[Dict] = []
    per_image_deletion: List[Dict] = []

    images = list(_iter_images(image_root))
    print(f"{len(images)} images under {image_root}")
    for idx, (cls, img_path) in enumerate(images):
        image = Image.open(img_path).convert("RGB")
        try:
            saliency_16, target_id, target_text = compute_saliency(
                model, processor, image, args.prompt, method=args.method
            )
        except Exception as exc:
            print(f"  [skip] {img_path}: {exc}")
            continue
        patch_saliency = pool_to_patch_grid(saliency_16, args.patch_grid)
        if args.order_mode == "random":
            # Chance-level control, mirroring concept_deletion_eval.py's
            # order_mode=random convention: same masking mechanism, patches
            # shuffled instead of ranked by saliency. Seeded per-image (not
            # globally) so it's reproducible yet independent across images.
            rng = np.random.default_rng(hash((args.seed, str(img_path))) % (2**32))
            order = rng.permutation(n_patches)
        else:
            order = np.argsort(-patch_saliency.reshape(-1))  # high -> low saliency, flat patch index

        ins_probs, del_probs = [], []
        for k in ks:
            top_k = set(order[:k].tolist())
            keep_mask = np.zeros((args.patch_grid, args.patch_grid), dtype=bool)
            for flat_idx in top_k:
                keep_mask[flat_idx // args.patch_grid, flat_idx % args.patch_grid] = True
            ins_img = _masked_image(image, args.patch_grid, keep_mask)
            ins_probs.append(_target_prob(model, processor, ins_img, args.prompt, target_id))

            del_keep_mask = np.ones((args.patch_grid, args.patch_grid), dtype=bool)
            for flat_idx in top_k:
                del_keep_mask[flat_idx // args.patch_grid, flat_idx % args.patch_grid] = False
            del_img = _masked_image(image, args.patch_grid, del_keep_mask)
            del_probs.append(_target_prob(model, processor, del_img, args.prompt, target_id))

        insertion_curves.append(ins_probs)
        deletion_curves.append(del_probs)
        ins_arr, del_arr = np.array(ins_probs), np.array(del_probs)
        trapezoid = getattr(np, "trapezoid", np.trapz)
        per_image_insertion.append({
            "image_path": str(img_path), "class": cls,
            "auc": float(trapezoid(ins_arr, fracs)),
            "auc_relative": _curve_auc_relative(fracs, ins_arr),
            "n_tokens": 1,
        })
        per_image_deletion.append({
            "image_path": str(img_path), "class": cls,
            "auc": float(trapezoid(del_arr, fracs)),
            "auc_relative": _curve_auc_relative(fracs, del_arr),
            "n_tokens": 1,
        })
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (idx + 1) % 20 == 0 or idx == len(images) - 1:
            print(f"  [{idx + 1}/{len(images)}] {img_path.name} target='{target_text.strip()}'")

    def _write(direction: str, curves: List[List[float]], per_image: List[Dict]) -> None:
        if not curves:
            print(f"No curves computed for {direction}; skipping output.")
            return
        arr = np.array(curves)
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        suffix = "_random" if args.order_mode == "random" else ""
        base = f"patch_{direction}_{args.method}{suffix}"
        with open(out_dir / f"{base}.json", "w") as f:
            json.dump({
                "fractions": fracs.tolist(), "mean": mean.tolist(), "std": std.tolist(),
                "method": args.method, "direction": direction, "prompt": args.prompt,
                "patch_grid": args.patch_grid, "n_images": len(curves),
            }, f, indent=2)
        with open(out_dir / f"{base}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fraction", "mean_prob", "std_prob"])
            for x, m, s in zip(fracs.tolist(), mean.tolist(), std.tolist()):
                w.writerow([x, m, s])
        with open(out_dir / f"{base}_per_image.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "class", "auc", "auc_relative", "n_tokens"])
            for row in per_image:
                w.writerow([row["image_path"], row["class"], row["auc"], row["auc_relative"], row["n_tokens"]])
        print(f"Saved {direction}: mean AUC(relative)="
              f"{np.mean([r['auc_relative'] for r in per_image if r['auc_relative'] is not None]):.4f} "
              f"(n={len(per_image)})")

    _write("insertion", insertion_curves, per_image_insertion)
    _write("deletion", deletion_curves, per_image_deletion)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["attention", "gradcam"], required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--prompt", default="What are the objects in the image?")
    parser.add_argument("--patch_grid", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=10)
    parser.add_argument("--devices", default="cuda:0,1")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--order_mode", choices=["saliency", "random"], default="saliency",
                         help="'random' shuffles patch order instead of ranking by saliency -- chance-level control, same masking mechanism.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --order_mode random.")
    parser.add_argument("--limit", type=int, default=-1, help="For smoke testing: process only the first N images.")
    args = parser.parse_args()

    if args.limit > 0:
        # Monkeypatch _iter_images consumption via a wrapper is overkill;
        # simplest smoke-test knob is just slicing inside run() -- handled
        # by capping the images list directly here.
        global _iter_images
        _orig = _iter_images

        def _limited(image_root):
            for i, item in enumerate(_orig(image_root)):
                if i >= args.limit:
                    break
                yield item
        _iter_images = _limited

    run(args)


if __name__ == "__main__":
    main()
