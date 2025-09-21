import os
import sys
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

# --- Optional LangSAM integration for object detection ---
_LANGSAM_MODEL = None

def _ensure_repo_root_on_sys_path():
    # Repo root is two levels up from this file (xl-vlms/)
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.append(repo_root)

def _load_langsam_model():
    global _LANGSAM_MODEL
    if _LANGSAM_MODEL is not None:
        return _LANGSAM_MODEL
    _ensure_repo_root_on_sys_path()
    try:
        from src.langsam_utils import load_langsam
    except Exception as e:
        raise RuntimeError(f"Could not import src.langsam_utils.load_langsam: {e}")
    _LANGSAM_MODEL = load_langsam()
    return _LANGSAM_MODEL

def run_langsam_batched(image_paths: List[str], tag: str, batch_size: int = 8) -> List[List[Tuple[int, int, int, int]]]:
    """Run LangSAM batched detection for a text tag.
    Returns a list parallel to image_paths where each element is a list of (x,y,w,h) boxes.
    """
    model = _load_langsam_model()
    try:
        from src.langsam_utils import predict_bboxes_for_tag_batched
    except Exception as e:
        raise RuntimeError(f"Could not import src.langsam_utils.predict_bboxes_for_tag_batched: {e}")
    return predict_bboxes_for_tag_batched(model, image_paths, tag=tag, batch_size=batch_size)

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

def concept_process_json_mapping(
    json_file,
    input_root,
    output_root,
    resize_size,
    max_crops_per_image,
    patch_size,
    min_images_per_tag: int = 30,
    max_images_per_tag: int = 0,
    object_detection: bool = False,
    batch_size: int = 8,
):
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
        # Prepare abs paths and detection boxes per image for this tag if enabled
        boxes_map: Dict[str, List[Tuple[int, int, int, int]]] = {}
        abs_paths: List[str] = []
        if object_detection:
            for rel_path in rel_paths:
                abs_p = os.path.join(input_root, rel_path)
                if os.path.isfile(abs_p):
                    abs_paths.append(abs_p)
                else:
                    failures.append((rel_path, tag, 'missing_file'))
            if abs_paths:
                try:
                    boxes_list = run_langsam_batched(abs_paths, tag=tag, batch_size=batch_size)
                    boxes_map = {p: b for p, b in zip(abs_paths, boxes_list)}
                except Exception as e:
                    print(f"Warning: LangSAM detection failed for tag '{tag}': {e}")
                    boxes_map = {}

        for rel_path in rel_paths:
            image_path = os.path.join(input_root, rel_path)
            if not os.path.isfile(image_path):
                # Recorded above for detection; add here if detection disabled
                if not object_detection:
                    failures.append((rel_path, tag, 'missing_file'))
                continue
            try:
                img = Image.open(image_path).convert('RGB')
            except Exception as e:
                failures.append((rel_path, tag, f'open_error:{e.__class__.__name__}'))
                continue

            # Resize by width; maintain aspect
            orig_w, orig_h = img.size
            img = _resize_keep_aspect(img, resize_size)
            w, h = img.size
            s1 = w / float(orig_w)

            saved_count = 0
            # If detection is enabled and we have boxes, save up to k crops from detections first
            if object_detection:
                boxes_xywh = boxes_map.get(image_path)
                if boxes_xywh:
                    boxes_xyxy = _scale_boxes_x1y1x2y2(_xywh_to_x1y1x2y2(boxes_xywh), s1)
                    # Ensure image is large enough for patch; upscale if needed and scale boxes accordingly
                    if w < patch_size or h < patch_size:
                        scale_w = patch_size / float(w) if w < patch_size else 1.0
                        scale_h = patch_size / float(h) if h < patch_size else 1.0
                        s2 = max(scale_w, scale_h)
                        if s2 > 1.0:
                            img = img.resize((int(round(w * s2)), int(round(h * s2))), Image.LANCZOS)
                            w, h = img.size
                            boxes_xyxy = _scale_boxes_x1y1x2y2(boxes_xyxy, s2)
                    boxes_xyxy = _clip_boxes_x1y1x2y2(boxes_xyxy, w, h)
                    k = int(max_crops_per_image)
                    base = Path(rel_path).stem
                    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy[:k]):
                        crop = img.crop((x1, y1, x2, y2)).convert('RGB')
                        crop = crop.resize((patch_size, patch_size), Image.LANCZOS)
                        crop.save(os.path.join(tag_out_dir, f"{tag.replace(' ', '_')}_{base}_concept_{saved_count + i}.png"))
                    saved_count = min(len(boxes_xyxy), k)

            # Fill remaining with random crops
            remaining = int(max_crops_per_image) - saved_count
            if remaining > 0:
                # Ensure image is large enough
                w, h = img.size
                if w < patch_size or h < patch_size:
                    new_w = max(w, patch_size)
                    s2 = new_w / float(w)
                    new_h = max(h, int(round(h * s2)))
                    if new_h < patch_size:
                        new_h = patch_size
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                crops = random_square_crops(img, patch_size, remaining, rng)
                if not crops:
                    reason = 'too_small' if (img.size[0] < patch_size or img.size[1] < patch_size) else 'no_space'
                    failures.append((rel_path, tag, reason))
                else:
                    base = Path(rel_path).stem
                    for i, c in enumerate(crops):
                        c.save(os.path.join(tag_out_dir, f"{tag.replace(' ', '_')}_{base}_concept_{saved_count + i}.png"))
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

def _xywh_to_x1y1x2y2(boxes: List[Tuple[float, float, float, float]]):
    return [(x, y, x + w, y + h) for (x, y, w, h) in boxes]

def _scale_boxes_x1y1x2y2(boxes: List[Tuple[float, float, float, float]], scale: float):
    return [(x1 * scale, y1 * scale, x2 * scale, y2 * scale) for (x1, y1, x2, y2) in boxes]

def _clip_boxes_x1y1x2y2(boxes: List[Tuple[float, float, float, float]], w: int, h: int) -> List[Tuple[int, int, int, int]]:
    clipped: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes:
        ix1 = max(0, min(int(round(x1)), w))
        iy1 = max(0, min(int(round(y1)), h))
        ix2 = max(0, min(int(round(x2)), w))
        iy2 = max(0, min(int(round(y2)), h))
        if ix2 > ix1 and iy2 > iy1:
            clipped.append((ix1, iy1, ix2, iy2))
    return clipped

def _save_object_crops(img: Image.Image, boxes_x1y1x2y2: List[Tuple[int, int, int, int]], patch_size: int, output_dir: str, base_name: str, object_tag: Optional[str] = None, start_index: int = 0) -> Tuple[int, List[Tuple[int, int, int, int]]]:
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    final_boxes: List[Tuple[int, int, int, int]] = []
    for i, (x1, y1, x2, y2) in enumerate(boxes_x1y1x2y2 or []):
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img.crop((x1, y1, x2, y2)).convert('RGB')
        crop = crop.resize((patch_size, patch_size), Image.LANCZOS)
        tag_part = f"_{object_tag}" if object_tag else ""
        crop.save(os.path.join(output_dir, f"{base_name}{tag_part}_obj_{start_index + i}.png"))
        saved += 1
        final_boxes.append((x1, y1, x2, y2))
    return saved, final_boxes

def create_grid_patches(image_path, patch_size, output_dir, resize_size=None, object_bboxes: Optional[List[Tuple[int, int, int, int]]] = None, object_tag: Optional[str] = None):
    img = Image.open(image_path).convert('RGB')
    img_name = Path(image_path).stem
    orig_w, orig_h = img.size
    # First resize by width
    img = _resize_keep_aspect(img, resize_size)
    w, h = img.size
    s1 = w / float(orig_w)
    boxes_x1y1x2y2 = _xywh_to_x1y1x2y2(object_bboxes or [])
    boxes_scaled = _scale_boxes_x1y1x2y2(boxes_x1y1x2y2, s1)
    # If too small, upscale to fit patch
    if w < patch_size or h < patch_size:
        new_w = max(w, patch_size)
        s2 = new_w / float(w)
        new_h = max(h, int(round(h * s2)))
        if new_h < patch_size:
            new_h = patch_size
        img = img.resize((new_w, new_h), Image.LANCZOS)
        boxes_scaled = _scale_boxes_x1y1x2y2(boxes_scaled, s2)
        w, h = img.size
    # Clip boxes to current image size, then save object crops
    boxes_scaled = _clip_boxes_x1y1x2y2(boxes_scaled, w, h)
    os.makedirs(output_dir, exist_ok=True)
    _save_object_crops(img, boxes_scaled, patch_size, output_dir, img_name, object_tag, start_index=0)
    # Proceed with grid patches
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

def create_random_patches(image_path, patch_size, output_dir, P, max_overlap_ratio=0.25, max_attempts_per_patch=100, resize_size=None, object_bboxes: Optional[List[Tuple[int, int, int, int]]] = None, object_tag: Optional[str] = None):
    img = Image.open(image_path).convert('RGB')
    name = Path(image_path).stem
    orig_w, orig_h = img.size
    # Resize by width
    img = _resize_keep_aspect(img, resize_size)
    w, h = img.size
    s1 = w / float(orig_w)
    boxes_x1y1x2y2 = _xywh_to_x1y1x2y2(object_bboxes or [])
    boxes_scaled = _scale_boxes_x1y1x2y2(boxes_x1y1x2y2, s1)
    # Ensure minimum size for patching
    if w < patch_size or h < patch_size:
        scale_w = patch_size / float(w) if w < patch_size else 1.0
        scale_h = patch_size / float(h) if h < patch_size else 1.0
        s2 = max(scale_w, scale_h)
        if s2 > 1.0:
            img = img.resize((int(round(w * s2)), int(round(h * s2))), Image.LANCZOS)
            boxes_scaled = _scale_boxes_x1y1x2y2(boxes_scaled, s2)
            w, h = img.size
    os.makedirs(output_dir, exist_ok=True)
    # Save object crops first, treat them as existing patches for overlap
    boxes_scaled = _clip_boxes_x1y1x2y2(boxes_scaled, w, h)
    _, obj_boxes_final = _save_object_crops(img, boxes_scaled, patch_size, output_dir, name, object_tag, start_index=0)
    saved = list(obj_boxes_final)
    if w == patch_size and h == patch_size and P > 0:
        img.convert('RGB').save(os.path.join(output_dir, f"{name}_patch_0.png"))
        return
    created = 0
    attempts = 0
    max_attempts_total = P * max_attempts_per_patch
    while created < P and attempts < max_attempts_total:
        attempts += 1
        left = random.randint(0, max(0, w - patch_size))
        top = random.randint(0, max(0, h - patch_size))
        box = (left, top, left + patch_size, top + patch_size)
        if any(calculate_iou(box, b) > max_overlap_ratio for b in saved):
            continue
        img.crop(box).convert('RGB').save(os.path.join(output_dir, f"{name}_patch_{created}.png"))
        saved.append(box)
        created += 1
    if created < P:
        print(f"Info: only generated {created}/{P} patches for {image_path} (overlap constraints).")

def process_folder_structure(root_input, root_output, patch_size=128, P=10, max_overlap=0.25, resize_size=None, grid=False, object_detection: bool = False, batch_size: int = 8):
    # Build per-tag image lists (tag = immediate subfolder name of the image path)
    tag_to_paths: Dict[str, List[str]] = {}
    for subdir, _, files in os.walk(root_input):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                p = os.path.join(subdir, f)
                tag = Path(subdir).name
                tag_to_paths.setdefault(tag, []).append(p)
    # Run LangSAM per tag group to use the tag as the text prompt
    boxes_map: Dict[str, List[Tuple[int, int, int, int]]] = {}
    if object_detection:
        for tag, paths in tag_to_paths.items():
            if not paths:
                continue
            boxes_list = run_langsam_batched(paths, tag=tag, batch_size=batch_size)
            boxes_map.update({p: b for p, b in zip(paths, boxes_list)})
    # Process folders and pass through boxes
    for subdir, _, files in os.walk(root_input):
        rel = os.path.relpath(subdir, root_input)
        out_sub = os.path.join(root_output, rel)
        os.makedirs(out_sub, exist_ok=True)
        current_tag = Path(subdir).name
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(subdir, f)
                bboxes = boxes_map.get(path)
                if grid:
                    create_grid_patches(path, patch_size, out_sub, resize_size, object_bboxes=bboxes, object_tag=current_tag if object_detection else None)
                else:
                    create_random_patches(path, patch_size, out_sub, P, max_overlap_ratio=max_overlap, resize_size=resize_size, object_bboxes=bboxes, object_tag=current_tag if object_detection else None)

def process_json_mapping(json_file, input_root, output_root, patch_size=128, P=10, max_overlap=0.25, resize_size=None, grid=False, min_images_per_tag=30, max_images_per_tag=0, object_detection: bool = False, batch_size: int = 8):
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
        # Prepare absolute paths and run batch detection for this group if enabled
        abs_paths: List[str] = []
        for rel in rels:
            p = os.path.join(input_root, rel)
            if os.path.isfile(p):
                abs_paths.append(p)
            else:
                print(f"Warning: missing image {p}")
        boxes_map: Dict[str, List[Tuple[int, int, int, int]]] = {}
        # Use the mapping key 'tag' as the detection prompt
        tag_to_use = tag
        if object_detection and abs_paths:
            boxes_list = run_langsam_batched(abs_paths, tag=tag_to_use, batch_size=batch_size)
            boxes_map = {p: b for p, b in zip(abs_paths, boxes_list)}
        for rel in rels:
            img_path = os.path.join(input_root, rel)
            if not os.path.isfile(img_path):
                continue
            bboxes = boxes_map.get(img_path)
            if grid:
                create_grid_patches(img_path, patch_size, out_dir, resize_size, object_bboxes=bboxes, object_tag=tag_to_use if object_detection else None)
            else:
                create_random_patches(img_path, patch_size, out_dir, P, max_overlap_ratio=max_overlap, resize_size=resize_size, object_bboxes=bboxes, object_tag=tag_to_use if object_detection else None)

def main():
    parser = argparse.ArgumentParser(description="Extract image patches (random, grid, or concept-focused) with optional JSON tag mapping.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--patch_size", type=int, default=200, help="Size for square patches / concept crops")
    parser.add_argument("--patches_per_image", type=int, default=18, help="Random patches per image (random mode)")
    parser.add_argument("--max_overlap", type=float, default=0.5, help="Max IoU overlap (random mode)")
    parser.add_argument("--resize", type=int, default=500, help="Target width resize (keep aspect)")
    parser.add_argument("--grid", action='store_true', help="Enable grid mode")
    parser.add_argument("--json_mapping", type=str, default=None, help="Tag -> [relative paths] JSON")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--concept_mode", action='store_true', help="Enable concept cropping (min images threshold)")
    parser.add_argument("--concept_crops_per_image", type=int, default=3, help="Crops per image in concept mode")
    parser.add_argument("--min_images_per_tag", type=int, default=30, help="Minimum images required for a tag to be processed")
    parser.add_argument("--max_images_per_tag", type=int, default=0, help="Cap on number of images sampled per tag (0 = no cap)")
    # Object detection (single simple flag)
    parser.add_argument("--object_detection", action='store_true', help="Enable LangSAM object detection before cropping")
    # Single batch size flag
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for LangSAM detection")
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    # No external object_tag required: JSON mode uses mapping keys; folder mode uses subfolder name
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
            object_detection=args.object_detection,
            batch_size=args.batch_size,
        )
        return
    if args.json_mapping:
        process_json_mapping(
            args.json_mapping, args.input_root, args.output_root,
            args.patch_size, args.patches_per_image, args.max_overlap,
            args.resize, args.grid,
            min_images_per_tag=args.min_images_per_tag,
            max_images_per_tag=args.max_images_per_tag,
                object_detection=args.object_detection,
                batch_size=args.batch_size,
        )
    else:
        process_folder_structure(
            args.input_root, args.output_root, args.patch_size,
            args.patches_per_image, args.max_overlap, args.resize, args.grid,
                object_detection=args.object_detection,
                batch_size=args.batch_size,
        )

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
   --input_root  /mnt/abka03/Projects/xl-vlms/data  \
   --output_root  /mnt/abka03/Projects/xl-vlms/crops/train  \
   --json_mapping /mnt/abka03/Projects/xl-vlms/data/coco_10_concept_image_mapping.json \
   --concept_mode --concept_crops_per_image 100 --patch_size 128 --resize 512 --seed 123 --min_images_per_tag 10 --max_images_per_tag 300
With object tag

python preprocessing/random_crops.py \
  --input_root /mnt/abka03/Projects/xl-vlms/data \
  --output_root /mnt/abka03/Projects/xl-vlms/output_patches \
  --json_mapping /mnt/abka03/Projects/xl-vlms/data/coco_10_concept_image_mapping.json \
  --patch_size 128 --patches_per_image 16 --resize 512 \
  --object_detection
   

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
