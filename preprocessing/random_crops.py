import os
import random
import argparse
from PIL import Image
from pathlib import Path
import json
import csv
from typing import Dict, List, Tuple, Optional

# Utility
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# Simplified: removed unused frequency/quantile helper logic.

def _resize_keep_aspect(img, target_width):
    if target_width is None:
        return img
    w, h = img.size
    if w == target_width:
        return img
    new_height = int(round(h * (target_width / float(w))))
    return img.resize((target_width, new_height), Image.LANCZOS)

# --- Functional helpers for concept-focused mode ---

def load_mapping(json_file: str) -> Dict[str, List[str]]:
    with open(json_file, 'r') as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if isinstance(v, list)}

def random_square_crops(img: Image.Image, patch_size: int, k: int, rng: random.Random) -> List[Image.Image]:
    w, h = img.size
    if w < patch_size or h < patch_size:
        return []
    if w == patch_size and h == patch_size:
        return [img]
    max_left = w - patch_size
    max_top = h - patch_size
    if max_left < 0 or max_top < 0:
        return []
    crops = []
    for _ in range(k):
        left = rng.randint(0, max_left)
        top = rng.randint(0, max_top)
        crops.append(img.crop((left, top, left + patch_size, top + patch_size)))
    return crops

def process_single_image(rel_path: str, tag: str, input_root: str, resize_size: int, patch_size: int, k: int, rng: random.Random) -> Tuple[List[Tuple[Image.Image, str]], Optional[Tuple[str, str, str]]]:
    image_path = os.path.join(input_root, rel_path)
    if not os.path.isfile(image_path):
        return [], (rel_path, tag, 'missing_file')
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        return [], (rel_path, tag, f'open_error:{e.__class__.__name__}')
    img = _resize_keep_aspect(img, resize_size)
    crops = random_square_crops(img, patch_size, k, rng)
    if not crops:
        w, h = img.size
        reason = 'too_small' if (w < patch_size or h < patch_size) else 'no_space'
        return [], (rel_path, tag, reason)
    base = Path(rel_path).stem
    labeled = [(c, f"{tag.replace(' ', '_')}_{base}_concept_{i}.png") for i, c in enumerate(crops)]
    return labeled, None

def write_csv(path: str, header: List[str], rows: List[Tuple]):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def write_json(path: str, obj: dict):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def concept_process_json_mapping(json_file, input_root, output_root, resize_size, max_crops_per_image, patch_size, min_images_per_tag=30, max_images_per_tag=0):
    # Process all tags with >= min_images_per_tag images; optionally cap images per tag
    mapping = load_mapping(json_file)
    rng = random
    ensure_dir(output_root)
    failures: List[Tuple[str, str, str]] = []
    processed_tags = 0
    skipped_tags = 0
    for tag, rel_paths in mapping.items():
        if not isinstance(rel_paths, list):
            continue
        if len(rel_paths) < min_images_per_tag:
            skipped_tags += 1
            continue
        if max_images_per_tag > 0 and len(rel_paths) > max_images_per_tag:
            rel_paths = rng.sample(rel_paths, max_images_per_tag)
        processed_tags += 1
        safe_tag = tag.replace(' ', '_')
        tag_out_dir = os.path.join(output_root, safe_tag)
        ensure_dir(tag_out_dir)
        for rel_path in rel_paths:
            crops_with_names, failure = process_single_image(rel_path, tag, input_root, resize_size, patch_size, max_crops_per_image, rng)
            if failure:
                failures.append(failure)
                continue
            for crop_img, fname in crops_with_names:
                crop_img.save(os.path.join(tag_out_dir, fname))
    write_csv(os.path.join(output_root, 'failures.csv'), ['image_path', 'tag', 'reason'], failures)
    write_json(os.path.join(output_root, 'stats.json'), {
        'processed_tags': processed_tags,
        'skipped_tags': skipped_tags,
        'min_images_threshold': min_images_per_tag,
        'max_images_cap': max_images_per_tag if max_images_per_tag > 0 else None
    })
    print(f"Concept mode: processed {processed_tags} tags, skipped {skipped_tags} (<{min_images_per_tag} images).")
    if max_images_per_tag > 0:
        print(f"Per-tag image cap: {max_images_per_tag}")
    if failures:
        print("Failures recorded in failures.csv")

def create_grid_patches(image_path, patch_size, output_dir, resize_size=None):
    img = Image.open(image_path)
    img_name = Path(image_path).stem
    img = _resize_keep_aspect(img, resize_size)
    w, h = img.size
    if w < patch_size or h < patch_size:
        new_w = max(w, patch_size)
        scale = new_w / float(w)
        new_h = max(h, int(round(h * scale)))
        if new_h < patch_size:
            new_h = patch_size
        img = img.resize((new_w, new_h), Image.LANCZOS)
        w, h = img.size
    os.makedirs(output_dir, exist_ok=True)
    pid = 0
    for top in range(0, h - patch_size + 1, patch_size):
        for left in range(0, w - patch_size + 1, patch_size):
            crop = img.crop((left, top, left + patch_size, top + patch_size)).convert('RGB')
            crop.save(os.path.join(output_dir, f"{img_name}_patch_{pid}.png"))
            pid += 1

def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    interA = inter_w * inter_h
    if interA == 0:
        return 0.0
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - interA
    if union <= 0:
        return 0.0
    return interA / union

def create_random_patches(image_path, patch_size, output_dir, P, max_overlap_ratio=0.25, max_attempts_per_patch=100, resize_size=None):
    img = Image.open(image_path)
    name = Path(image_path).stem
    img = _resize_keep_aspect(img, resize_size)
    w, h = img.size
    if w < patch_size or h < patch_size:
        scale_w = patch_size / float(w) if w < patch_size else 1.0
        scale_h = patch_size / float(h) if h < patch_size else 1.0
        scale = max(scale_w, scale_h)
        if scale > 1.0:
            img = img.resize((int(round(w*scale)), int(round(h*scale))), Image.LANCZOS)
            w, h = img.size
    os.makedirs(output_dir, exist_ok=True)
    if w == patch_size and h == patch_size and P > 0:
        img.convert('RGB').save(os.path.join(output_dir, f"{name}_patch_0.png"))
        return
    saved = []
    created = 0
    attempts = 0
    max_attempts_total = P * max_attempts_per_patch
    while created < P and attempts < max_attempts_total:
        attempts += 1
        left = random.randint(0, w - patch_size)
        top = random.randint(0, h - patch_size)
        box = (left, top, left + patch_size, top + patch_size)
        if any(calculate_iou(box, b) > max_overlap_ratio for b in saved):
            continue
        img.crop(box).convert('RGB').save(os.path.join(output_dir, f"{name}_patch_{created}.png"))
        saved.append(box)
        created += 1
    if created < P:
        print(f"Info: only generated {created}/{P} patches for {image_path} (overlap constraints).")

def process_folder_structure(root_input, root_output, patch_size=128, P=10, max_overlap=0.25, resize_size=None, grid=False):
    for subdir, _, files in os.walk(root_input):
        rel = os.path.relpath(subdir, root_input)
        out_sub = os.path.join(root_output, rel)
        os.makedirs(out_sub, exist_ok=True)
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(subdir, f)
                if grid:
                    create_grid_patches(path, patch_size, out_sub, resize_size)
                else:
                    create_random_patches(path, patch_size, out_sub, P, max_overlap_ratio=max_overlap, resize_size=resize_size)

def process_json_mapping(json_file, input_root, output_root, patch_size=128, P=10, max_overlap=0.25, resize_size=None, grid=False, min_images_per_tag=30, max_images_per_tag=0):
    with open(json_file, 'r') as f:
        mapping = json.load(f)
    rng = random
    for tag, rels in mapping.items():
        if not isinstance(rels, list):
            continue
        if len(rels) < min_images_per_tag:
            print(f"Skipping tag '{tag}' (<{min_images_per_tag} images)")
            continue
        if max_images_per_tag > 0 and len(rels) > max_images_per_tag:
            rels = rng.sample(rels, max_images_per_tag)
        safe = tag.replace(' ', '_')
        out_dir = os.path.join(output_root, safe)
        os.makedirs(out_dir, exist_ok=True)
        for rel in rels:
            img_path = os.path.join(input_root, rel)
            if not os.path.isfile(img_path):
                print(f"Warning: missing image {img_path}")
                continue
            if grid:
                create_grid_patches(img_path, patch_size, out_dir, resize_size)
            else:
                create_random_patches(img_path, patch_size, out_dir, P, max_overlap_ratio=max_overlap, resize_size=resize_size)

def main():
    parser = argparse.ArgumentParser(description="Extract image patches (random, grid, or concept-focused) with optional JSON tag mapping.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--patch_size", type=int, default=200, help="Size for square patches / concept crops")
    parser.add_argument("--patches_per_image", type=int, default=18, help="Random patches per image (random mode)")
    parser.add_argument("--max_overlap", type=float, default=0.8, help="Max IoU overlap (random mode)")
    parser.add_argument("--resize", type=int, default=500, help="Target width resize (keep aspect)")
    parser.add_argument("--grid", action='store_true', help="Enable grid mode")
    parser.add_argument("--json_mapping", type=str, default=None, help="Tag -> [relative paths] JSON")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--concept_mode", action='store_true', help="Enable concept cropping (min images threshold)")
    parser.add_argument("--concept_crops_per_image", type=int, default=3, help="Crops per image in concept mode")
    parser.add_argument("--min_images_per_tag", type=int, default=30, help="Minimum images required for a tag to be processed")
    parser.add_argument("--max_images_per_tag", type=int, default=0, help="Cap on number of images sampled per tag (0 = no cap)")
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    if args.concept_mode:
        if not args.json_mapping:
            raise ValueError('--concept_mode requires --json_mapping')
        concept_process_json_mapping(
            json_file=args.json_mapping,
            input_root=args.input_root,
            output_root=args.output_root,
            resize_size=args.resize,
            max_crops_per_image=args.concept_crops_per_image,
            patch_size=args.patch_size,
            min_images_per_tag=args.min_images_per_tag,
            max_images_per_tag=args.max_images_per_tag,
        )
        return
    if args.json_mapping:
        process_json_mapping(args.json_mapping, args.input_root, args.output_root,
                             args.patch_size, args.patches_per_image, args.max_overlap,
                             args.resize, args.grid, min_images_per_tag=args.min_images_per_tag,
                             max_images_per_tag=args.max_images_per_tag)
    else:
        process_folder_structure(args.input_root, args.output_root, args.patch_size,
                                 args.patches_per_image, args.max_overlap, args.resize, args.grid)

if __name__ == '__main__':
    main()

# ...existing example usage docstring remains below...
"""
Example usage with JSON mapping (with width resize):
python preprocessing/random_crops.py \
   --input_root /mnt/abka03/xlvlm_data/imagenet_1000 \
   --output_root /mnt/abka03/xlvlm_data/imagenet_1000_auto_crops \
   --json_mapping /mnt/abka03/xlvlm_data/imagenet_1000/coco_1000_concept_image_mapping.json \
   --patch_size 200 --patches_per_image 16 --max_overlap 0.7 --seed 42 --resize 512

Concept-focused cropping (skips tags with <30 images by default):
python preprocessing/random_crops.py \
   --input_root  /mnt/abka03/xlvlm_data/imagenet_1000  \
   --output_root  /mnt/abka03/xlvlm_data/imagenet_1000_auto_crops/train  \
   --json_mapping /mnt/abka03/xlvlm_data/imagenet_1000/coco_1000_concept_image_mapping.json \
   --concept_mode --concept_crops_per_image 10 --patch_size 189 --resize 512 --seed 123 --min_images_per_tag 30 --max_images_per_tag 50

Concept-focused cropping with per-tag cap (e.g., max 50 images used per tag):
python preprocessing/random_crops.py \
   --input_root /data/images_root \
   --output_root /data/concept_crops_capped \
   --json_mapping /data/tags.json \
   --concept_mode --concept_crops_per_image 8 --patch_size 224 --resize 512 \
   --min_images_per_tag 30 --max_images_per_tag 50

1) Random patches over a directory tree (defaults patch_size=200, resize width=500):
python preprocessing/random_crops.py \
   --input_root /data/images \
   --output_root /data/patches_random

2) Random patches specifying all key parameters:
python preprocessing/random_crops.py \
   --input_root /data/images \
   --output_root /data/patches_random_custom \
   --patch_size 160 --patches_per_image 12 --max_overlap 0.3 --resize 640 --seed 123

3) Grid patches over a directory tree:
python preprocessing/random_crops.py \
   --input_root /data/images \
   --output_root /data/patches_grid \
   --grid --patch_size 128 --resize 512

4) Random patches using a JSON mapping file (skips tags with <30 images by default):
python preprocessing/random_crops.py \
   --input_root /data/images_root \
   --output_root /data/patches_by_tag \
   --json_mapping /data/tags.json \
   --patch_size 128 --patches_per_image 40 --resize 600 --min_images_per_tag 30

5) Grid patches with JSON mapping:
python preprocessing/random_crops.py \
   --input_root /data/images_root \
   --output_root /data/patches_by_tag_grid \
   --json_mapping /data/tags.json \
   --grid --patch_size 128 --resize 600 --min_images_per_tag 30

6) Disable resizing (operate on original widths):
python preprocessing/random_crops.py \
   --input_root /data/images \
   --output_root /data/patches_no_resize \
   --patch_size 224 --patches_per_image 16 --max_overlap 0.4

Notes:
 - --resize sets target width only; height is scaled to keep aspect ratio.
 - For grid mode, patches stride equals patch_size (no overlap).
 - For random mode, IoU between accepted patches is kept <= --max_overlap.
 - Concept mode processes tags with >= --min_images_per_tag images (default 30).
 - Failures (concept mode): missing_file, open_error, too_small, no_space.
 - Outputs (concept mode): stats.json, failures.csv at --output_root.
"""
