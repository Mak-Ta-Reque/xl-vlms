#!/usr/bin/env python3
"""
Build a 10-category COCO subset ("coco10") analogous to data/imagenet10.

Source: real COCO train2017 (for the train split) + val2017 (for the test/val
split), both found at
/media/NVME_8TB/abka03/Projects/vlm_space/local-evidence-vlm/data/raw/
(train2017/, val2017/, annotations/instances_{train,val}2017.json).

This is the standard COCO train/val convention -- no artificial re-split is
needed since train2017 and val2017 are already disjoint, official pools.

Per category, up to TRAIN_CAP (default 300) images go to train (from
train2017), up to TEST_CAP (default 50) go to test (from val2017) -- capped
by whatever's actually available.

Categories: apple, banana, bird, cake, cat, cup, dog, donut, knife, orange
(COCO's official name is "donut", not "doughnut").

Output layout (mirrors data/imagenet10):
    data/coco10/train_all/<file>.jpg              flat pool, union of all
                                                   categories' train images
    data/coco10/val/<category>/<file>.jpg         per-category test images
    data/coco10/manifest.json                     full record: per-category
                                                   train/test image ids, file
                                                   names, and annotation ids
                                                   (for the mask-crop/grid
                                                   builder script)

Usage:
    python preprocessing/build_coco10_dataset.py
    python preprocessing/build_coco10_dataset.py --train-cap 20 --test-cap 5  # pilot subset
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

COCO_ROOT = Path("/media/NVME_8TB/abka03/Projects/vlm_space/local-evidence-vlm/data/raw")
TRAIN_ANNOTATIONS_PATH = COCO_ROOT / "annotations" / "instances_train2017.json"
TRAIN_IMAGES_DIR = COCO_ROOT / "train2017"
VAL_ANNOTATIONS_PATH = COCO_ROOT / "annotations" / "instances_val2017.json"
VAL_IMAGES_DIR = COCO_ROOT / "val2017"

TARGET_CATEGORIES = [
    "apple", "banana", "bird", "cake", "cat",
    "cup", "dog", "donut", "knife", "orange",
]

ROOT_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "coco10"


def load_annotations(path: Path):
    with path.open() as f:
        return json.load(f)


def build_indices(coco_json: dict):
    cat_id_to_name = {c["id"]: c["name"] for c in coco_json["categories"]}
    target_name_to_id = {name: cid for cid, name in cat_id_to_name.items() if name in TARGET_CATEGORIES}
    missing = set(TARGET_CATEGORIES) - set(target_name_to_id.keys())
    if missing:
        raise ValueError(f"Category names not found in COCO categories: {missing}")

    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_json["images"]}

    # image_id -> {category_name: [annotation, ...]}
    img_cat_annotations: Dict[int, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for ann in coco_json["annotations"]:
        cid = ann["category_id"]
        if cid in cat_id_to_name and cat_id_to_name[cid] in TARGET_CATEGORIES:
            img_cat_annotations[ann["image_id"]][cat_id_to_name[cid]].append(ann)

    return target_name_to_id, image_id_to_filename, img_cat_annotations


def select_per_category(
    train_img_cat_annotations: Dict[int, Dict[str, List[dict]]],
    test_img_cat_annotations: Dict[int, Dict[str, List[dict]]],
    seed: int,
    train_cap: int,
    test_cap: int,
):
    rng = random.Random(seed)
    per_category = {}
    for category in TARGET_CATEGORIES:
        train_candidates = sorted(
            img_id for img_id, cats in train_img_cat_annotations.items() if category in cats
        )
        rng.shuffle(train_candidates)

        # Validation crops are ground-truth-mask crops shown directly to the
        # VLM (as 2x2 grid cells) -- many COCO instances are tiny (a few px
        # across), which upscales to blurry/unrecognizable grid cells. Prefer
        # the largest instances (by total annotated pixel area) for the val
        # split so the object is actually legible; training doesn't need this
        # (it crops from whole images via its own detector, not these masks).
        test_candidates = [
            img_id for img_id, cats in test_img_cat_annotations.items() if category in cats
        ]
        test_candidates.sort(
            key=lambda img_id: sum(a.get("area", 0) for a in test_img_cat_annotations[img_id][category]),
            reverse=True,
        )
        per_category[category] = {
            "train_image_ids": train_candidates[:train_cap],
            "test_image_ids": test_candidates[:test_cap],
        }
    return per_category


def copy_images(
    per_category: dict,
    train_image_id_to_filename: Dict[int, str],
    test_image_id_to_filename: Dict[int, str],
    output_dir: Path,
) -> None:
    train_all_dir = output_dir / "train_all"
    # Only add-if-missing below; clear stale files from a prior build first so
    # re-running with a different seed/cap/selection logic can't leave old and
    # new selections mixed together in the same flat pool.
    if train_all_dir.exists():
        shutil.rmtree(train_all_dir)
    train_all_dir.mkdir(parents=True, exist_ok=True)
    val_dir = output_dir / "val"
    if val_dir.exists():
        shutil.rmtree(val_dir)

    copied_train_files = set()
    for category, split in per_category.items():
        for img_id in split["train_image_ids"]:
            fname = train_image_id_to_filename[img_id]
            if fname in copied_train_files:
                continue
            copied_train_files.add(fname)
            src = TRAIN_IMAGES_DIR / fname
            dst = train_all_dir / fname
            if not dst.exists():
                shutil.copy2(src, dst)

        val_cat_dir = output_dir / "val" / category
        val_cat_dir.mkdir(parents=True, exist_ok=True)
        for img_id in split["test_image_ids"]:
            fname = test_image_id_to_filename[img_id]
            src = VAL_IMAGES_DIR / fname
            dst = val_cat_dir / fname
            if not dst.exists():
                shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-cap", type=int, default=300, help="Max train images per category")
    parser.add_argument("--test-cap", type=int, default=50, help="Max test images per category")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    train_json = load_annotations(TRAIN_ANNOTATIONS_PATH)
    _, train_image_id_to_filename, train_img_cat_annotations = build_indices(train_json)

    val_json = load_annotations(VAL_ANNOTATIONS_PATH)
    _, val_image_id_to_filename, val_img_cat_annotations = build_indices(val_json)

    per_category = select_per_category(
        train_img_cat_annotations, val_img_cat_annotations,
        seed=args.seed, train_cap=args.train_cap, test_cap=args.test_cap,
    )

    print("Per-category selection:")
    for category, split in per_category.items():
        print(f"  {category:8s}: train={len(split['train_image_ids']):4d}  test={len(split['test_image_ids']):3d}")

    copy_images(per_category, train_image_id_to_filename, val_image_id_to_filename, output_dir)

    # Manifest: everything the mask-crop/grid builder needs, keyed by
    # (category, split) -> [{"image_id", "file_name", "annotations": [...]}]
    manifest = {"categories": TARGET_CATEGORIES, "seed": args.seed, "data": {}}
    for category, split in per_category.items():
        manifest["data"][category] = {}
        for split_name, ids_key, id_to_fname, cat_ann in (
            ("train", "train_image_ids", train_image_id_to_filename, train_img_cat_annotations),
            ("test", "test_image_ids", val_image_id_to_filename, val_img_cat_annotations),
        ):
            entries = []
            for img_id in split[ids_key]:
                entries.append({
                    "image_id": img_id,
                    "file_name": id_to_fname[img_id],
                    "annotations": cat_ann[img_id][category],
                })
            manifest["data"][category][split_name] = entries

    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f)
    print(f"\nSaved manifest: {manifest_path}")
    print(f"train_all pool: {output_dir / 'train_all'}")
    print(f"val (per-category): {output_dir / 'val'}")


if __name__ == "__main__":
    main()
