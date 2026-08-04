#!/usr/bin/env python3
"""
Build segmentation-masked object crops for coco10's test/val split, then
assemble them into 2x2 grids using the existing grid builder unchanged.

For each (category, test image) in manifest.json (see build_coco10_dataset.py):
  1. Decode the ground-truth COCO segmentation mask(s) for that category in
     that image (unioned across instances, e.g. multiple apples in one bowl
     -- same "concept_union" convention already used elsewhere in this repo's
     crop pipeline, preprocessing/crops_to_json.py).
  2. Crop to the mask's tight bounding box + MASK_CONTEXT_PIXELS-style
     padding, save under data/coco10/val_masked/<category>/<file>.jpg.

Then preprocessing/create_grids.py's create_image_grids() is called
UNCHANGED, pointed at val_masked/ (which has the same one-subfolder-per-class
layout it expects) -- this produces data/coco10/val_grids/ with the exact
same grid_{n}_{idx}__cat1__cat2..._.jpg naming the existing explainer/pipeline
already consumes, so no changes are needed there.

Usage:
    python preprocessing/build_coco10_masked_grids.py
    python preprocessing/build_coco10_masked_grids.py --num-grids 10  # pilot
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

COCO_ROOT = Path("/media/NVME_8TB/abka03/Projects/vlm_space/local-evidence-vlm/data/raw")
IMAGES_DIR = COCO_ROOT / "val2017"

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(Path(__file__).parent))
from create_grids import create_image_grids  # noqa: E402


def _annotation_to_mask(ann: dict, height: int, width: int) -> np.ndarray:
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict):
        rle = seg if isinstance(seg["counts"], (bytes, str)) else mask_utils.frPyObjects(seg, height, width)
    else:
        raise ValueError(f"Unexpected segmentation type: {type(seg)}")
    return mask_utils.decode(rle).astype(bool)


def build_masked_crop(entry: dict, context_pixels: int = 16, min_size: int = 32):
    """Union all of this category's instance masks in the image, crop to the
    tight bbox + context padding. Returns a PIL Image, or None if degenerate."""
    file_name = entry["file_name"]
    img_path = IMAGES_DIR / file_name
    image = Image.open(img_path).convert("RGB")
    width, height = image.size

    union_mask = np.zeros((height, width), dtype=bool)
    for ann in entry["annotations"]:
        union_mask |= _annotation_to_mask(ann, height, width)

    ys, xs = np.where(union_mask)
    if len(ys) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    x1 = max(0, x1 - context_pixels)
    y1 = max(0, y1 - context_pixels)
    x2 = min(width, x2 + context_pixels)
    y2 = min(height, y2 + context_pixels)

    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None

    return image.crop((x1, y1, x2, y2))


def build_all_masked_crops(manifest: dict, output_dir: Path, context_pixels: int = 10) -> None:
    for category, splits in manifest["data"].items():
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        n_saved = 0
        for entry in splits["test"]:
            crop = build_masked_crop(entry, context_pixels=context_pixels)
            if crop is None:
                continue
            out_path = cat_dir / entry["file_name"]
            crop.save(out_path)
            n_saved += 1
        print(f"  {category:8s}: {n_saved}/{len(splits['test'])} masked crops saved -> {cat_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(ROOT_DIR / "data" / "coco10" / "manifest.json"))
    parser.add_argument("--output-root", default=str(ROOT_DIR / "data" / "coco10"))
    parser.add_argument("--context-pixels", type=int, default=16)
    parser.add_argument("--num-grids", type=int, default=50)
    parser.add_argument("--grid-n", type=int, default=4, help="Images per grid (perfect square, default 2x2=4)")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    masked_dir = output_root / "val_masked"
    grids_dir = output_root / "val_grids"
    # build_all_masked_crops/create_image_grids only add-if-missing; clear any
    # prior build first so a re-run with different selection/sizing can't mix
    # stale files in alongside the new ones.
    if masked_dir.exists():
        shutil.rmtree(masked_dir)
    if grids_dir.exists():
        shutil.rmtree(grids_dir)

    with open(args.manifest) as f:
        manifest = json.load(f)

    print("Building segmentation-masked crops:")
    build_all_masked_crops(manifest, masked_dir, context_pixels=args.context_pixels)

    import random
    random.seed(args.seed)

    print(f"\nBuilding {args.num_grids} grids ({args.grid_n}-cell) from masked crops...")
    create_image_grids(
        input_dir=str(masked_dir),
        output_dir=str(grids_dir),
        n=args.grid_n,
        num_grids=args.num_grids,
        image_size=args.image_size,
    )
    print(f"\nSaved grids to: {grids_dir}")


if __name__ == "__main__":
    main()
