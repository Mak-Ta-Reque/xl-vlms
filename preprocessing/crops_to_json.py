import os
import sys
import json
import random
import argparse
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp

# Avoid hard dependency at import time; read image size lazily


# ------------------------
# Utilities and helpers
# ------------------------

def _ensure_repo_root_on_sys_path():
    # Repo root is two levels up from this file (xl-vlms/)
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.append(repo_root)


def _get_image_size(path: str) -> Optional[Tuple[int, int]]:
    """Return (width, height) using Pillow if available; None on failure.
    We avoid importing PIL at module import time to keep this script lightweight.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    try:
        with Image.open(path) as img:
            w, h = img.size
            return int(w), int(h)
    except Exception:
        return None


def _pillow_available() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except Exception:
        return False


def _atomic_write_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _already_done(result: Dict[str, dict], tag: str, rel_path: str) -> bool:
    tag_bucket = result.get(tag)
    if not isinstance(tag_bucket, dict):
        return False
    return rel_path in tag_bucket


def _cleanup_gpu_memory():
    """Release GPU memory after detection batches to prevent leakage."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# No resizing is performed in this script.


_LANGSAM_MODEL = None


def _load_langsam_model(device: Optional[str] = None):
    global _LANGSAM_MODEL
    if _LANGSAM_MODEL is not None:
        return _LANGSAM_MODEL
    _ensure_repo_root_on_sys_path()
    try:
        from src.langsam_utils import load_langsam
    except Exception as e:
        raise RuntimeError(f"Could not import src.langsam_utils.load_langsam: {e}")
    # Forward device hint to loader (best-effort device selection handled inside)
    _LANGSAM_MODEL = load_langsam(device=device)
    return _LANGSAM_MODEL


_SAM3_MODEL = None


def _load_sam3_model(device: Optional[str] = None, confidence_threshold: float = 0.5):
    global _SAM3_MODEL
    if _SAM3_MODEL is not None:
        return _SAM3_MODEL
    _ensure_repo_root_on_sys_path()
    try:
        from src.sam3_utils import load_sam3
    except Exception as e:
        raise RuntimeError(f"Could not import src.sam3_utils.load_sam3: {e}")
    _SAM3_MODEL = load_sam3(device=device, confidence_threshold=confidence_threshold)
    return _SAM3_MODEL


def run_langsam_batched(
    images: List[Any],
    tag: str,
    batch_size: int = 8,
    model: Optional[Any] = None,
    topn: int = 10,
) -> List[List[Tuple[int, int, int, int]]]:
    """Run LangSAM batched detection for a text tag.
    Returns a list parallel to images where each element is a list of up to topn (x,y,w,h) boxes.
    """
    # Load model only if not provided; callers can pass a preloaded instance to avoid reloads.
    if model is None:
        model = _load_langsam_model()
    try:
        from src.langsam_utils import predict_bboxes_for_tag_batched
    except Exception as e:
        raise RuntimeError(f"Could not import src.langsam_utils.predict_bboxes_for_tag_batched: {e}")
    # Run predictions (accepts paths or PIL images)
    boxes_per_image = predict_bboxes_for_tag_batched(model, images, tag=tag, batch_size=batch_size, topn=topn)
    # Limit to topn per image (defensive: handle bad topn gracefully)
    n = max(0, int(topn))
    if n == 0:
        return [[] for _ in images]
    return [b[:n] if isinstance(b, list) else [] for b in boxes_per_image]


def run_sam3_batched(
    images: List[Any],
    tag: str,
    batch_size: int = 8,
    model: Optional[Any] = None,
    topn: int = 10,
) -> List[List[Tuple[int, int, int, int]]]:
    """Run SAM3 batched detection for a text tag.
    Returns a list parallel to images where each element is a list of up to topn (x,y,w,h) boxes.
    """
    if model is None:
        model = _load_sam3_model()
    try:
        from src.sam3_utils import predict_bboxes_for_tag_sam3_batched
    except Exception as e:
        raise RuntimeError(f"Could not import src.sam3_utils.predict_bboxes_for_tag_sam3_batched: {e}")
    # Run predictions (accepts paths or PIL images)
    boxes_per_image = predict_bboxes_for_tag_sam3_batched(model, images, tag=tag, batch_size=batch_size, topn=topn)
    n = max(0, int(topn))
    if n == 0:
        return [[] for _ in images]
    return [b[:n] if isinstance(b, list) else [] for b in boxes_per_image]


def run_detector_batched(
    images: List[Any],
    tag: str,
    detector: str,
    batch_size: int = 8,
    model: Optional[Any] = None,
    topn: int = 10,
) -> List[List[Tuple[int, int, int, int]]]:
    """Run object detection for a text tag using the specified detector.
    
    Args:
        images: List of image file paths OR PIL Images.
        tag: Text prompt for the object to detect.
        detector: One of 'langsam' or 'sam3'.
        batch_size: Batch size for detection.
        model: Preloaded model instance (optional).
        topn: Maximum number of boxes per image.
    
    Returns:
        List parallel to images, each element is a list of (x,y,w,h) boxes.
    """
    if detector == "langsam":
        return run_langsam_batched(images, tag, batch_size=batch_size, model=model, topn=topn)
    elif detector == "sam3":
        return run_sam3_batched(images, tag, batch_size=batch_size, model=model, topn=topn)
    else:
        raise ValueError(f"Unknown detector: {detector}. Use 'langsam' or 'sam3'.")


def load_mapping(json_file: str) -> Dict[str, List[str]]:
    with open(json_file, 'r') as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if isinstance(v, list)}


def _xywh_to_x1y1x2y2(boxes: List[Tuple[float, float, float, float]]):
    return [(x, y, x + w, y + h) for (x, y, w, h) in boxes]


def _compute_virtual_resize(
    orig_w: int, orig_h: int, image_size_width: Optional[int]
) -> Tuple[int, int, float]:
    """Compute a resized size using reference width and aspect ratio.

    Returns:
        (new_w, new_h, scale)
    """
    if image_size_width is None:
        return orig_w, orig_h, 1.0
    target_w = int(image_size_width)
    if target_w <= 0 or orig_w <= 0 or orig_h <= 0:
        return orig_w, orig_h, 1.0
    scale = target_w / float(orig_w)
    new_h = max(1, int(round(orig_h * scale)))
    return target_w, new_h, float(scale)


def _load_and_resize_image(
    abs_image_path: str,
    image_size_width: Optional[int],
) -> Tuple[Any, Tuple[int, int], float]:
    """Load an image and optionally resize it in memory.

    Returns:
        (PIL_image_or_path, (w, h), scale)
        If resize not needed, returns the path string and original size.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        # No Pillow, return path and let detector handle it
        return abs_image_path, (0, 0), 1.0

    try:
        img = Image.open(abs_image_path).convert("RGB")
        orig_w, orig_h = img.size
    except Exception:
        return abs_image_path, (0, 0), 1.0

    new_w, new_h, scale = _compute_virtual_resize(orig_w, orig_h, image_size_width)
    if image_size_width is None or (new_w, new_h) == (orig_w, orig_h):
        return img, (orig_w, orig_h), 1.0

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    img_resized = img.resize((new_w, new_h), resample=resample)
    return img_resized, (new_w, new_h), scale


def _clip_boxes_x1y1x2y2(
    boxes: List[Tuple[float, float, float, float]], w: int, h: int
) -> List[Tuple[int, int, int, int]]:
    clipped: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes:
        ix1 = max(0, min(int(round(x1)), w))
        iy1 = max(0, min(int(round(y1)), h))
        ix2 = max(0, min(int(round(x2)), w))
        iy2 = max(0, min(int(round(y2)), h))
        if ix2 > ix1 and iy2 > iy1:
            clipped.append((ix1, iy1, ix2, iy2))
    return clipped


def _centered_square_box_around(
    x1: float, y1: float, x2: float, y2: float, patch_size: int, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """Return a patch_size x patch_size box centered on the given bbox, clamped to image bounds."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half = patch_size / 2.0
    left = int(round(cx - half))
    top = int(round(cy - half))
    max_left = max(0, img_w - patch_size)
    max_top = max(0, img_h - patch_size)
    left = min(max(left, 0), max_left)
    top = min(max(top, 0), max_top)
    right = left + patch_size
    bottom = top + patch_size
    return left, top, right, bottom


def _crop_detection_to_patch(
    bbox: Tuple[int, int, int, int], patch_size: int, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """Create a patch_size x patch_size box centered on the detection bbox, clamped to bounds."""
    x1, y1, x2, y2 = bbox
    cx1, cy1, cx2, cy2 = _centered_square_box_around(x1, y1, x2, y2, patch_size, img_w, img_h)
    return cx1, cy1, cx2, cy2


def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    interA = inter_w * inter_h
    if interA == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - interA
    if union <= 0:
        return 0.0
    return interA / union


def _gen_grid_boxes(w: int, h: int, patch_size: int) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    for top in range(0, h - patch_size + 1, patch_size):
        for left in range(0, w - patch_size + 1, patch_size):
            boxes.append((left, top, left + patch_size, top + patch_size))
    return boxes


def _gen_random_boxes(
    w: int,
    h: int,
    patch_size: int,
    count: int,
    existing: List[Tuple[int, int, int, int]],
    max_overlap_ratio: float,
    max_attempts_per_patch: int = 100,
) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    created = 0
    attempts = 0
    max_attempts_total = count * max_attempts_per_patch
    while created < count and attempts < max_attempts_total:
        attempts += 1
        left = random.randint(0, max(0, w - patch_size))
        top = random.randint(0, max(0, h - patch_size))
        box = (left, top, left + patch_size, top + patch_size)
        if any(calculate_iou(box, b) > max_overlap_ratio for b in (existing + boxes)):
            continue
        boxes.append(box)
        created += 1
    return boxes


# ------------------------
# Worker for multiprocessing
# ------------------------

def _process_one_image(task: Tuple[str, str, str, int, int, float, bool, Optional[List[Tuple[int, int, int, int]]], Optional[int], Optional[int]]) -> Optional[Tuple[str, str, dict]]:
    """
    task = (
        tag, image_path, rel_path,
        patch_size, P, max_overlap,
        grid, detections_xywh, concept_k, topn
    )
    Returns (tag, rel_path, entry_dict) or None on failure.
    """
    (
        tag, image_path, rel_path,
        patch_size, P, max_overlap,
        grid, detections_xywh, concept_k, topn,
    ) = task

    size = _get_image_size(image_path)
    if not size:
        return None
    w, h = size

    # Prepare detections (converted to xyxy and clipped)
    det_xyxy: List[Tuple[int, int, int, int]] = []
    if detections_xywh:
        det_xyxy = _clip_boxes_x1y1x2y2(_xywh_to_x1y1x2y2(detections_xywh), w, h)
        if isinstance(topn, int):
            n = max(0, int(topn))
            det_xyxy = det_xyxy[:n] if n > 0 else []

    # Existing boxes for overlap exclusion: detection-centered patches (not saved)
    existing: List[Tuple[int, int, int, int]] = []
    if det_xyxy:
        det_patches = [_crop_detection_to_patch(b, patch_size, w, h) for b in det_xyxy]
        existing = _clip_boxes_x1y1x2y2(det_patches, w, h)

    # Generate random/grid boxes
    random_boxes: List[Tuple[int, int, int, int]] = []
    if grid:
        random_boxes = _gen_grid_boxes(w, h, patch_size) if (w >= patch_size and h >= patch_size) else []
    else:
        if w >= patch_size and h >= patch_size:
            if concept_k is not None:
                num = max(0, int(concept_k) - len(existing))
                random_boxes = _gen_random_boxes(w, h, patch_size, num, existing, max_overlap)
            else:
                random_boxes = _gen_random_boxes(w, h, patch_size, int(P), existing, max_overlap)
        else:
            random_boxes = []

    entry = {
        "meta": {"image_size": [w, h], "patch_size": patch_size},
        "detections_xyxy": [list(b) for b in det_xyxy],
        "random_crops": [list(b) for b in random_boxes],
    }
    return tag, rel_path, entry


# ------------------------
# Core processors that build JSON
# ------------------------

def _prepare_detection_boxes_for_image(
    detections_xywh: Optional[List[Tuple[int, int, int, int]]],
    img_w: int,
    img_h: int,
) -> List[Tuple[int, int, int, int]]:
    boxes_xyxy = _xywh_to_x1y1x2y2(detections_xywh or [])
    boxes_xyxy = _clip_boxes_x1y1x2y2(boxes_xyxy, img_w, img_h)
    return boxes_xyxy


def _record_for_image(
    tag_bucket: Dict[str, dict],
    rel_path: str,
    image_size: Tuple[int, int],
    patch_size: int,
    detections_xyxy: List[Tuple[int, int, int, int]],
    random_boxes: List[Tuple[int, int, int, int]],
):
    tag_bucket[rel_path] = {
        "meta": {
            "image_size": list(image_size),
            "patch_size": patch_size,
        },
        "detections_xyxy": [list(b) for b in detections_xyxy],
        "random_crops": [list(b) for b in random_boxes],
    }


def process_folder_structure_to_json(
    root_input: str,
    patch_size: int = 128,
    P: int = 10,
    max_overlap: float = 0.25,
    grid: bool = False,
    object_detection: bool = False,
    detector: str = "langsam",
    batch_size: int = 8,
    topn: int = 10,
    device: Optional[str] = None,
    image_size_width: Optional[int] = None,
    result: Optional[Dict[str, dict]] = None,
    output_json: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, dict]:
    # Build per-tag image lists (tag = immediate subfolder name of the image path)
    tag_to_paths: Dict[str, List[str]] = {}
    for subdir, _, files in os.walk(root_input):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                p = os.path.join(subdir, f)
                tag = Path(subdir).name
                tag_to_paths.setdefault(tag, []).append(p)

    # Run object detection per tag group (optionally on resized images in memory)
    boxes_map: Dict[str, List[Tuple[int, int, int, int]]] = {}
    if object_detection:
        # Load model once and reuse across all tags
        if detector == "langsam":
            model = _load_langsam_model(device=device)
        elif detector == "sam3":
            model = _load_sam3_model(device=device)
        else:
            model = None
        for tag, paths in tag_to_paths.items():
            if not paths:
                continue
            try:
                # Load and resize images in memory for detection
                det_images: List[Any] = []
                orig_paths: List[str] = []
                for p in paths:
                    img_or_path, _, _ = _load_and_resize_image(p, image_size_width)
                    det_images.append(img_or_path)
                    orig_paths.append(p)

                boxes_list = run_detector_batched(det_images, tag=tag, detector=detector, batch_size=batch_size, model=model, topn=topn)
                boxes_map.update({op: b for op, b in zip(orig_paths, boxes_list)})
                # Free PIL images and GPU cache after each tag
                del det_images
                _cleanup_gpu_memory()
            except Exception as e:
                print(f"Warning: detection failed for tag '{tag}' with detector={detector}, batch_size={batch_size}: {e}")
                # Fallback: no detections for these paths
                for p in paths:
                    boxes_map[p] = []

    result = result or {}

    for tag, paths in tag_to_paths.items():
        if not paths:
            continue
        tag_bucket: Dict[str, dict] = result.setdefault(tag, {})
        # simple counters for diagnostics
        processed = 0
        skipped_existing = 0
        size_fail = 0
        for image_path in paths:
            rel_path = os.path.relpath(image_path, root_input)
            if _already_done(result, tag, rel_path):
                skipped_existing += 1
                continue
            size = _get_image_size(image_path)
            if not size:
                size_fail += 1
                continue
            orig_w, orig_h = size

            # Operate in resized coordinate system (record-only).
            # If detection is enabled, detection ran on a resized copy, so boxes are already in resized coords.
            w, h, scale = _compute_virtual_resize(orig_w, orig_h, image_size_width)

            detections_xywh = boxes_map.get(image_path, []) if object_detection else []
            det_boxes_xyxy = _prepare_detection_boxes_for_image(detections_xywh, w, h)
            det_patch_boxes: List[Tuple[int, int, int, int]] = []
            if det_boxes_xyxy:
                det_patch_boxes = [
                    _crop_detection_to_patch(b, patch_size, w, h) for b in det_boxes_xyxy
                ]
                det_patch_boxes = _clip_boxes_x1y1x2y2(det_patch_boxes, w, h)

            random_boxes: List[Tuple[int, int, int, int]] = []
            if grid:
                if w >= patch_size and h >= patch_size:
                    random_boxes = _gen_grid_boxes(w, h, patch_size)
                else:
                    random_boxes = []
            else:
                if w >= patch_size and h >= patch_size:
                    existing = list(det_patch_boxes)
                    random_boxes = _gen_random_boxes(w, h, patch_size, P, existing, max_overlap)
                else:
                    random_boxes = []

            _record_for_image(
                tag_bucket,
                rel_path,
                (w, h),
                patch_size,
                det_boxes_xyxy,
                random_boxes,
            )
            processed += 1
            if output_json:
                _atomic_write_json(output_json, result)

        if verbose:
            print(f"[folder] tag='{tag}': total_files={len(paths)} processed={processed} size_fail={size_fail} already_done={skipped_existing}")

    return result


def process_json_mapping_to_json(
    json_file: str,
    input_root: str,
    patch_size: int = 128,
    P: int = 10,
    max_overlap: float = 0.25,
    grid: bool = False,
    min_images_per_tag: int = 30,
    max_images_per_tag: int = 0,
    object_detection: bool = False,
    detector: str = "langsam",
    batch_size: int = 8,
    topn: int = 10,
    device: Optional[str] = None,
    image_size_width: Optional[int] = None,
    result: Optional[Dict[str, dict]] = None,
    output_json: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, dict]:
    mapping = load_mapping(json_file)
    rng = random
    result = result or {}

    # Preload detection model once if needed
    detection_model: Optional[Any] = None
    if object_detection:
        if detector == "langsam":
            detection_model = _load_langsam_model(device=device)
        elif detector == "sam3":
            detection_model = _load_sam3_model(device=device)

    for tag, rels in mapping.items():
        if not isinstance(rels, list):
            continue
        if len(rels) < min_images_per_tag:
            continue
        if max_images_per_tag > 0 and len(rels) > max_images_per_tag:
            rels = rng.sample(rels, max_images_per_tag)

        abs_paths: List[str] = []
        for rel in rels:
            p = os.path.join(input_root, rel)
            if os.path.isfile(p):
                abs_paths.append(p)
        if verbose and not abs_paths:
            print(f"[mapping] Warning: tag='{tag}' has {len(rels)} entries but 0 files found under input_root='{input_root}'. Check your --input_root and mapping paths.")

        boxes_map: Dict[str, List[Tuple[int, int, int, int]]] = {}
        if object_detection and abs_paths:
            try:
                # Load and resize images in memory for detection
                det_images: List[Any] = []
                orig_paths: List[str] = []
                for p in abs_paths:
                    img_or_path, _, _ = _load_and_resize_image(p, image_size_width)
                    det_images.append(img_or_path)
                    orig_paths.append(p)
                boxes_list = run_detector_batched(det_images, tag=tag, detector=detector, batch_size=batch_size, model=detection_model, topn=topn)
                boxes_map = {op: b for op, b in zip(orig_paths, boxes_list)}
                # Free PIL images and GPU cache after each tag
                del det_images
                _cleanup_gpu_memory()
            except Exception as e:
                print(f"Warning: detection failed for tag '{tag}' with detector={detector}, batch_size={batch_size}: {e}")
                boxes_map = {p: [] for p in abs_paths}

        tag_bucket: Dict[str, dict] = result.setdefault(tag, {})
        processed = 0
        skipped_existing = 0
        size_fail = 0

        for rel in rels:
            img_path = os.path.join(input_root, rel)
            if not os.path.isfile(img_path):
                continue
            if _already_done(result, tag, rel):
                skipped_existing += 1
                continue
            size = _get_image_size(img_path)
            if not size:
                size_fail += 1
                continue
            orig_w, orig_h = size

            # Operate in resized coordinate system (record-only)
            w, h, scale = _compute_virtual_resize(orig_w, orig_h, image_size_width)

            detections_xywh = boxes_map.get(img_path, []) if object_detection else []
            det_boxes_xyxy = _prepare_detection_boxes_for_image(detections_xywh, w, h)
            det_patch_boxes: List[Tuple[int, int, int, int]] = []
            if det_boxes_xyxy:
                det_patch_boxes = [
                    _crop_detection_to_patch(b, patch_size, w, h) for b in det_boxes_xyxy
                ]
                det_patch_boxes = _clip_boxes_x1y1x2y2(det_patch_boxes, w, h)

            if grid:
                if w >= patch_size and h >= patch_size:
                    random_boxes = _gen_grid_boxes(w, h, patch_size)
                else:
                    random_boxes = []
            else:
                if w >= patch_size and h >= patch_size:
                    existing = list(det_patch_boxes)
                    random_boxes = _gen_random_boxes(w, h, patch_size, P, existing, max_overlap)
                else:
                    random_boxes = []

            _record_for_image(
                tag_bucket,
                rel,
                (w, h),
                patch_size,
                det_boxes_xyxy,
                random_boxes,
            )
            processed += 1
            if output_json:
                _atomic_write_json(output_json, result)

        if verbose:
            print(f"[mapping] tag='{tag}': candidates={len(rels)} found_files={len(abs_paths)} processed={processed} size_fail={size_fail} already_done={skipped_existing}")

    return result


def concept_process_json_mapping_to_json(
    json_file: str,
    input_root: str,
    max_crops_per_image: int,
    patch_size: int,
    min_images_per_tag: int = 30,
    max_images_per_tag: int = 0,
    object_detection: bool = False,
    detector: str = "langsam",
    batch_size: int = 8,
    topn: int = 10,
    device: Optional[str] = None,
    image_size_width: Optional[int] = None,
    result: Optional[Dict[str, dict]] = None,
    output_json: Optional[str] = None,
    max_overlap: float = 0.30,
    verbose: bool = False,
) -> Dict[str, dict]:
    mapping = load_mapping(json_file)
    rng = random
    result = result or {}

    # Preload detection model once if needed
    detection_model: Optional[Any] = None
    if object_detection:
        if detector == "langsam":
            detection_model = _load_langsam_model(device=device)
        elif detector == "sam3":
            detection_model = _load_sam3_model(device=device)

    for tag, rel_paths in mapping.items():
        if not isinstance(rel_paths, list):
            continue
        if len(rel_paths) < min_images_per_tag:
            continue
        if max_images_per_tag > 0 and len(rel_paths) > max_images_per_tag:
            rel_paths = rng.sample(rel_paths, max_images_per_tag)
        
        # Prepare detection per-tag
        boxes_map: Dict[str, List[Tuple[int, int, int, int]]] = {}
        abs_paths: List[str] = []
        if object_detection:
            for rel_path in rel_paths:
                abs_p = os.path.join(input_root, rel_path)
                if os.path.isfile(abs_p):
                    abs_paths.append(abs_p)
            if abs_paths:
                try:
                    # Load and resize images in memory for detection
                    det_images: List[Any] = []
                    orig_paths: List[str] = []
                    for p in abs_paths:
                        img_or_path, _, _ = _load_and_resize_image(p, image_size_width)
                        det_images.append(img_or_path)
                        orig_paths.append(p)
                    boxes_list = run_detector_batched(det_images, tag=tag, detector=detector, batch_size=batch_size, model=detection_model, topn=topn)
                    boxes_map = {op: b for op, b in zip(orig_paths, boxes_list)}
                    # Free PIL images and GPU cache after each tag
                    del det_images
                    _cleanup_gpu_memory()
                except Exception as e:
                    print(f"Warning: detection failed for tag '{tag}' with detector={detector}: {e}")
                    boxes_map = {}

        tag_bucket: Dict[str, dict] = result.setdefault(tag, {})
        processed = 0
        skipped_existing = 0
        size_fail = 0

        for rel_path in rel_paths:
            img_path = os.path.join(input_root, rel_path)
            if not os.path.isfile(img_path):
                continue
            if _already_done(result, tag, rel_path):
                skipped_existing += 1
                continue
            size = _get_image_size(img_path)
            if not size:
                size_fail += 1
                continue
            orig_w, orig_h = size

            # Operate in resized coordinate system (record-only)
            w, h, scale = _compute_virtual_resize(orig_w, orig_h, image_size_width)

            detections_xywh = boxes_map.get(img_path, []) if object_detection else []
            det_boxes_xyxy = _prepare_detection_boxes_for_image(detections_xywh, w, h)

            # Save up to k crops from detections first (centered patches)
            k = int(max_crops_per_image)
            det_patch_boxes: List[Tuple[int, int, int, int]] = []
            if det_boxes_xyxy:
                for b in det_boxes_xyxy[:k]:
                    det_patch_boxes.append(_crop_detection_to_patch(b, patch_size, w, h))
                det_patch_boxes = _clip_boxes_x1y1x2y2(det_patch_boxes, w, h)

            # Fill remaining with random crops
            remaining = max(0, k - len(det_patch_boxes))
            random_boxes: List[Tuple[int, int, int, int]] = []
            if remaining > 0 and w >= patch_size and h >= patch_size:
                random_boxes = _gen_random_boxes(
                    w, h, patch_size, remaining, det_patch_boxes, max_overlap_ratio=max_overlap
                )

            _record_for_image(
                tag_bucket,
                rel_path,
                (w, h),
                patch_size,
                det_boxes_xyxy,
                random_boxes,
            )
            processed += 1
            if output_json:
                _atomic_write_json(output_json, result)

        if verbose:
            print(f"[concept] tag='{tag}': candidates={len(rel_paths)} processed={processed} size_fail={size_fail} already_done={skipped_existing}")

    return result


# ------------------------
# CLI
# ------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect crop and detection coordinates to JSON (folder or JSON mapping). "
            "Supports random/grid/concept modes and optional LangSAM detection."
        )
    )
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_json", required=True, help="Path to write the coordinates JSON")
    parser.add_argument("--patch_size", type=int, default=200, help="Square patch size")
    parser.add_argument("--patches_per_image", type=int, default=18, help="Random patches per image (random mode)")
    parser.add_argument("--max_overlap", type=float, default=0.50, help="Max IoU overlap among patches (random mode)")
    # No resize performed: different crop sizes allowed
    parser.add_argument("--grid", action="store_true", default=False, help="Enable grid mode instead of random")
    parser.add_argument("--json_mapping", type=str, default=None, help="Tag -> [relative paths] JSON")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--image_size_width",
        type=int,
        default=None,
        help=(
            "Reference width for resizing (record-only). For each image, compute resized size "
            "(image_size_width, round(orig_h * image_size_width / orig_w)). Random crops are generated in this resized "
            "coordinate system and saved under meta.image_size. If an object detector is enabled, detection runs on a "
            "cached resized copy so detector boxes are produced directly in resized coordinates (no scaling)."
        ),
    )

    # Concept-focused parameters
    parser.add_argument("--concept_mode", action="store_true", help="Enable concept-focused cropping logic")
    parser.add_argument("--concept_crops_per_image", type=int, default=3, help="Crops per image in concept mode")
    parser.add_argument("--min_images_per_tag", type=int, default=30, help="Minimum images required per tag (JSON mapping modes)")
    parser.add_argument("--max_images_per_tag", type=int, default=0, help="Cap images per tag (0 = no cap)")
    parser.add_argument("--topn", type=int, default=10, help="Limit detector to top-N boxes per image (default 10)")

    # Optional object detection
    parser.add_argument("--object_detector", type=str, default="none", choices=["none", "langsam", "sam3"],
                        help="Object detector: 'none' (random crops only), 'langsam', or 'sam3' (detector + random)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for object detection")
    parser.add_argument("--verbose", action="store_true", help="Print per-tag diagnostics and hints")
    parser.add_argument("--device", type=str, default=None, help="Device for detection: cpu, cuda or cuda:N")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for SAM3 detection")

    args = parser.parse_args()
    # Check Pillow early to provide actionable feedback
    if not _pillow_available():
        print("Warning: Pillow is not available. Image sizes cannot be read and no coordinates will be recorded. Install it with 'pip install Pillow'.")

    if args.seed is not None:
        random.seed(args.seed)

    # Resolve detector choice
    detector = args.object_detector
    object_detection_enabled = detector != "none"

    # Incremental/resumable: load existing JSON if present
    result: Dict[str, dict] = {}
    if os.path.isfile(args.output_json):
        try:
            with open(args.output_json, 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    result = loaded
        except Exception:
            result = {}

    if args.concept_mode:
        if not args.json_mapping:
            raise ValueError("--concept_mode requires --json_mapping")
        # concept variant processes and we will append/flush per image in-place
        concept_result = concept_process_json_mapping_to_json(
            json_file=args.json_mapping,
            input_root=args.input_root,
            max_crops_per_image=args.concept_crops_per_image,
            patch_size=args.patch_size,
            min_images_per_tag=args.min_images_per_tag,
            max_images_per_tag=args.max_images_per_tag,
            object_detection=object_detection_enabled,
            detector=detector,
            batch_size=args.batch_size,
            topn=args.topn,
            device=args.device,
            result=result,
            output_json=args.output_json,
            max_overlap=args.max_overlap,
            verbose=args.verbose,
            image_size_width=args.image_size_width,
        )
        # Merge and write once (concept path currently gathers then writes)
        for tag, bucket in concept_result.items():
            result.setdefault(tag, {}).update(bucket)
        _atomic_write_json(args.output_json, result)
    else:
        if args.json_mapping:
            # Mapping mode: process and flush per image
            result = process_json_mapping_to_json(
                json_file=args.json_mapping,
                input_root=args.input_root,
                patch_size=args.patch_size,
                P=args.patches_per_image,
                max_overlap=args.max_overlap,
                grid=args.grid,
                min_images_per_tag=args.min_images_per_tag,
                max_images_per_tag=args.max_images_per_tag,
                object_detection=object_detection_enabled,
                detector=detector,
                batch_size=args.batch_size,
                topn=args.topn,
                device=args.device,
                result=result,
                output_json=args.output_json,
                verbose=args.verbose,
                image_size_width=args.image_size_width,
            )
            _atomic_write_json(args.output_json, result)
        else:
            # Folder mode: flush after each image processed
            result = process_folder_structure_to_json(
                root_input=args.input_root,
                patch_size=args.patch_size,
                P=args.patches_per_image,
                max_overlap=args.max_overlap,
                grid=args.grid,
                object_detection=object_detection_enabled,
                detector=detector,
                batch_size=args.batch_size,
                topn=args.topn,
                device=args.device,
                result=result,
                output_json=args.output_json,
                verbose=args.verbose,
                image_size_width=args.image_size_width,
            )
            # Ensure final write
            _atomic_write_json(args.output_json, result)

    # Simple summary
    total_tags = len(result)
    total_images = sum(len(v) for v in result.values())
    print(f"Wrote coordinates for {total_images} images across {total_tags} tags -> {args.output_json}")

    if total_images == 0:
        print("Hints: If you're using --json_mapping, ensure --input_root is the parent of the mapping's relative paths. Also ensure Pillow is installed so image sizes can be read. Use --verbose for per-tag stats.")


if __name__ == "__main__":
    main()
