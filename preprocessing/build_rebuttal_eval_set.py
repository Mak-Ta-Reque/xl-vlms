#!/usr/bin/env python3
"""
Build a fixed, class-restricted evaluation image set for the TGCL rebuttal
ablation (apple/cat/bird only), so every one of the 45 grid runs + 2 baseline
runs sees the byte-identical set of validation images -- required for the
per-image paired Wilcoxon signed-rank test (pairing is invalid if the
evaluation image set differs between conditions).

Source: data/coco10/val_masked/{apple,cat,bird}/*.jpg (single-object crops --
grid images can't be used here since a 2x2 grid's 4 quadrants aren't
distinguishable in the explainer's output JSON, only the whole grid image is).

Outputs:
  - data/coco10/eval_images_rebuttal.json   -- sorted manifest (audit trail)
  - data/coco10/val_masked_rebuttal/{apple,cat,bird}/  -- symlinks to the
    exact files listed in the manifest. Every run points IMAGE_ROOT here.

Usage:
    python preprocessing/build_rebuttal_eval_set.py
    python preprocessing/build_rebuttal_eval_set.py --classes apple,cat,bird --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", default="apple,cat,bird")
    parser.add_argument("--val-masked-dir", default=str(ROOT_DIR / "data" / "coco10" / "val_masked"))
    parser.add_argument("--dest-dir", default=str(ROOT_DIR / "data" / "coco10" / "val_masked_rebuttal"))
    parser.add_argument("--manifest", default=str(ROOT_DIR / "data" / "coco10" / "eval_images_rebuttal.json"))
    parser.add_argument(
        "--cap-per-class", type=int, default=-1,
        help="Max images per class (-1 = all available). Seeded random sample "
        "when a class has more than this many images, so the fixed eval set "
        "stays reproducible.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for --cap-per-class sampling.")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    val_masked_dir = Path(args.val_masked_dir)
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    manifest = {"classes": classes, "images": {}}
    total = 0
    for cls in classes:
        src_cls_dir = val_masked_dir / cls
        if not src_cls_dir.is_dir():
            raise RuntimeError(f"Expected class dir not found: {src_cls_dir}")
        files = sorted(
            f for f in src_cls_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.is_file()
        )
        if not files:
            raise RuntimeError(f"No images found under {src_cls_dir}")
        if args.cap_per_class > 0 and len(files) > args.cap_per_class:
            files = sorted(rng.sample(files, args.cap_per_class))

        dest_cls_dir = dest_dir / cls
        dest_cls_dir.mkdir(parents=True, exist_ok=True)
        # Clear stale symlinks from a prior build so the dest dir exactly
        # mirrors this manifest (no leftover files from an older class list).
        for stale in dest_cls_dir.iterdir():
            if stale.is_symlink() or stale.is_file():
                stale.unlink()

        rel_paths = []
        for f in files:
            link_path = dest_cls_dir / f.name
            link_path.symlink_to(f.resolve())
            rel_paths.append(str(f.resolve()))
        manifest["images"][cls] = rel_paths
        total += len(rel_paths)
        print(f"{cls}: {len(rel_paths)} images -> {dest_cls_dir}")

    with open(args.manifest, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nTotal: {total} images across {len(classes)} classes")
    print(f"Manifest written to: {args.manifest}")
    print(f"Fixed IMAGE_ROOT for all rebuttal runs: {dest_dir}")


if __name__ == "__main__":
    main()
