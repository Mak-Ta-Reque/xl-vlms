import os
import sys
import json
import random
import argparse
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp

# Progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: simple iterator wrapper that does nothing
    def tqdm(iterable, *args, **kwargs):
        return iterable

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


_YOLO_MODEL = None


def _load_yolo_model(device: Optional[str] = None, confidence_threshold: float = 0.25):
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    _ensure_repo_root_on_sys_path()
    try:
        from src.yolo_utils import load_yolo
    except Exception as e:
        raise RuntimeError(f"Could not import src.yolo_utils.load_yolo: {e}")
    _YOLO_MODEL = load_yolo(device=device, confidence_threshold=confidence_threshold)
    return _YOLO_MODEL


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


# ------------------------------------
# Mask-aware detection functions
# ------------------------------------

def _encode_mask_rle(mask):
    """RLE-encode a binary mask for JSON serialization."""
    _ensure_repo_root_on_sys_path()
    from src.yolo_utils import encode_mask_rle
    return encode_mask_rle(mask)


def run_langsam_batched_with_masks(
    images: List[Any],
    tag: str,
    batch_size: int = 8,
    model: Optional[Any] = None,
    topn: int = 10,
) -> List[List[Tuple[List[int], Any]]]:
    """Run LangSAM detection returning (bbox_xywh, mask_or_None) pairs per image."""
    if model is None:
        model = _load_langsam_model()
    try:
        from src.langsam_utils import predict_bboxes_and_masks_for_tag_batched
    except Exception as e:
        raise RuntimeError(f"Could not import predict_bboxes_and_masks_for_tag_batched: {e}")
    pairs_per_image = predict_bboxes_and_masks_for_tag_batched(
        model, images, tag=tag, batch_size=batch_size, topn=topn
    )
    n = max(0, int(topn))
    if n == 0:
        return [[] for _ in images]
    return [p[:n] if isinstance(p, list) else [] for p in pairs_per_image]


def run_sam3_batched_with_masks(
    images: List[Any],
    tag: str,
    batch_size: int = 8,
    model: Optional[Any] = None,
    topn: int = 10,
) -> List[List[Tuple[List[int], Any]]]:
    """Run SAM3 detection returning (bbox_xywh, mask_or_None) pairs per image."""
    if model is None:
        model = _load_sam3_model()
    try:
        from src.sam3_utils import predict_bboxes_and_masks_for_tag_sam3_batched
    except Exception as e:
        raise RuntimeError(f"Could not import predict_bboxes_and_masks_for_tag_sam3_batched: {e}")
    pairs_per_image = predict_bboxes_and_masks_for_tag_sam3_batched(
        model, images, tag=tag, batch_size=batch_size, topn=topn
    )
    n = max(0, int(topn))
    if n == 0:
        return [[] for _ in images]
    return [p[:n] if isinstance(p, list) else [] for p in pairs_per_image]


def run_yolo_all_objects_with_masks(
    images: List[Any],
    batch_size: int = 8,
    model: Optional[Any] = None,
    topn: int = 10,
) -> List[List[Tuple[List[int], Any]]]:
    """Run YOLO instance segmentation (tag-free). Returns all detected objects with masks."""
    if model is None:
        model = _load_yolo_model()
    try:
        from src.yolo_utils import predict_all_objects, detections_to_bbox_mask_pairs
    except Exception as e:
        raise RuntimeError(f"Could not import yolo_utils: {e}")
    all_dets = predict_all_objects(model, images, batch_size=batch_size, topn=topn)
    return [detections_to_bbox_mask_pairs(dets) for dets in all_dets]


def run_detector_batched_with_masks(
    images: List[Any],
    tag: str,
    detector: str,
    batch_size: int = 8,
    model: Optional[Any] = None,
    topn: int = 10,
) -> List[List[Tuple[List[int], Any]]]:
    """Unified mask-aware detection dispatcher.
    
    Returns:
        List parallel to images, each element is a list of (bbox_xywh, mask_or_None) tuples.
        For YOLO, tag is ignored (all objects detected regardless of class).
    """
    if detector == "langsam":
        return run_langsam_batched_with_masks(images, tag, batch_size=batch_size, model=model, topn=topn)
    elif detector == "sam3":
        return run_sam3_batched_with_masks(images, tag, batch_size=batch_size, model=model, topn=topn)
    elif detector == "yolo_seg":
        return run_yolo_all_objects_with_masks(images, batch_size=batch_size, model=model, topn=topn)
    else:
        raise ValueError(f"Unknown detector: {detector}. Use 'langsam', 'sam3', or 'yolo_seg'.")


def _pairs_to_rle_masks(pairs: List[Tuple[List[int], Any]]) -> List[Optional[dict]]:
    """RLE-encode masks from (bbox, mask) pairs for JSON serialization."""
    rle_list = []
    for _, mask in pairs:
        if mask is not None:
            rle_list.append(_encode_mask_rle(mask))
        else:
            rle_list.append(None)
    return rle_list


# ---------------------------------------------------------------------------
# Non-tag (automatic) segmentation wrappers
# ---------------------------------------------------------------------------

def run_langsam_all_masks(
    images: List[Any],
    model: Optional[Any] = None,
    topn: int = 10,
    min_mask_area: int = 100,
) -> List[List[Tuple[List[int], Any]]]:
    """Run LangSAM automatic segmentation (no text prompt) on a list of images.

    Uses SAM2AutomaticMaskGenerator inside LangSAM to segment everything.
    Pass resized images to reduce memory usage.

    Returns:
        List parallel to images, each element is a list of
        (bbox_xywh, mask_np) tuples sorted by area (largest first).
    """
    if model is None:
        model = _load_langsam_model()
    try:
        from src.langsam_utils import predict_all_masks_langsam
    except Exception as e:
        raise RuntimeError(f"Could not import predict_all_masks_langsam: {e}")
    return predict_all_masks_langsam(model, images, topn=topn, min_mask_area=min_mask_area)


def run_sam3_all_masks(
    images: List[Any],
    model: Optional[Any] = None,
    topn: int = 10,
    min_mask_area: int = 100,
    batch_size: int = 8,
    other_tags: Optional[List[str]] = None,
    exclude_tag: Optional[str] = None,
) -> List[List[Tuple[List[int], Any]]]:
    """Run SAM3 non-tag segmentation using other concept tags.

    SAM3 is text-prompted and cannot auto-segment with generic prompts.
    Instead we iterate through all other concept tags (from the mapping)
    and collect their detections.  The image backbone is computed once
    per image and reused across text prompts.

    When *other_tags* is ``None`` the legacy generic-prompt fallback is
    used (rarely produces results).

    Returns:
        List parallel to images, each element is a list of
        (bbox_xywh, mask_np) tuples.
    """
    if model is None:
        model = _load_sam3_model()
    try:
        from src.sam3_utils import predict_all_masks_sam3
    except Exception as e:
        raise RuntimeError(f"Could not import predict_all_masks_sam3: {e}")
    return predict_all_masks_sam3(
        model, images, topn=topn, min_mask_area=min_mask_area,
        batch_size=batch_size, other_tags=other_tags, exclude_tag=exclude_tag,
    )


def run_nontag_segmentation(
    images: List[Any],
    detector: str,
    model: Optional[Any] = None,
    topn: int = 10,
    min_mask_area: int = 100,
    batch_size: int = 8,
    other_tags: Optional[List[str]] = None,
    exclude_tag: Optional[str] = None,
) -> List[List[Tuple[List[int], Any]]]:
    """Unified dispatcher for non-tag (automatic) segmentation.

    For 'langsam': uses SAM2AutomaticMaskGenerator (everything mode).
    For 'sam3': iterates other concept tags (from *other_tags*) to collect
               non-concept masks (SAM3 cannot auto-segment with generic prompts).
    For 'yolo_seg': uses YOLO instance segmentation (tag-free).
    """
    if detector == "langsam":
        return run_langsam_all_masks(images, model=model, topn=topn, min_mask_area=min_mask_area)
    elif detector == "sam3":
        return run_sam3_all_masks(
            images, model=model, topn=topn, min_mask_area=min_mask_area,
            batch_size=batch_size, other_tags=other_tags, exclude_tag=exclude_tag,
        )
    elif detector == "yolo_seg":
        return run_yolo_all_objects_with_masks(images, batch_size=batch_size, model=model, topn=topn)
    else:
        raise ValueError(f"Unknown detector: {detector}. Use 'langsam', 'sam3', or 'yolo_seg'.")


# ---------------------------------------------------------------------------
# Concept + non-tag combined detection framework
# ---------------------------------------------------------------------------

def _subtract_mask(base_mask, subtract_mask_arr):
    """Subtract subtract_mask_arr from base_mask (both numpy bool arrays).

    When the two masks have different resolutions a resize is needed.
    After resizing we dilate the subtraction mask by 1 px so that
    interpolation boundary artefacts cannot leave a thin strip of
    overlap between the concept mask and the trimmed non-tag mask.

    Returns a new numpy bool array.
    """
    import numpy as np
    if base_mask is None:
        return None
    if subtract_mask_arr is None:
        return base_mask
    from PIL import Image as _Img
    # Resize if needed
    if base_mask.shape != subtract_mask_arr.shape:
        h, w = base_mask.shape
        sm = np.array(_Img.fromarray(
            subtract_mask_arr.astype(np.uint8) * 255
        ).resize((w, h), _Img.NEAREST)) > 127
        # Dilate by 1 px to absorb resize-boundary artefacts
        from scipy.ndimage import binary_dilation
        sm = binary_dilation(sm, iterations=1)
    else:
        sm = subtract_mask_arr
    return base_mask & (~sm)


def _mask_area(mask):
    """Return the number of True pixels in a mask."""
    if mask is None:
        return 0
    return int(mask.sum())


def build_concept_and_nontag_detections_for_image(
    img_pil,
    tag: str,
    detector: str,
    model: Any,
    topn_nontag: int = 10,
    min_mask_area: int = 100,
    all_tags: Optional[List[str]] = None,
) -> List[Tuple[List[int], Any]]:
    """Build per-tag detection list: ONE concept mask + non-tag masks (non-overlapping).

    Uses the SAME detector (LangSAM or SAM3) for both:
      1. Tag-based detection to get the concept mask.
      2. Non-tag detection to get all other object masks.
    No YOLO needed — SAM/LangSAM handle both tag and non-tag segmentation.

    For SAM3, *all_tags* is required so the function can iterate through
    other concept tags (SAM3 cannot auto-segment with generic prompts).

    The resized image should be passed to reduce memory usage.

    Args:
        img_pil: PIL Image (already resized to virtual size).
        tag: The concept tag for the tag-based detector.
        detector: 'langsam' or 'sam3'.
        model: Preloaded detector model (same model for both tag and non-tag).
        topn_nontag: Max non-tag detections per image.
        min_mask_area: Minimum pixels for a trimmed mask to be kept.
        all_tags: All concept tags from the mapping (needed for SAM3 multi-tag approach).

    Returns:
        List of (bbox_xywh, mask_np_or_None) tuples.
        First element is the concept detection (may be empty list if no concept found).
    """
    result_pairs: List[Tuple[List[int], Any]] = []

    # --- 1. Concept detection (tag-specific) ---
    concept_pairs: List[Tuple[List[int], Any]] = []
    if detector == "langsam":
        concept_pairs = run_langsam_batched_with_masks(
            [img_pil], tag, batch_size=1, model=model, topn=5
        )[0]
    elif detector == "sam3":
        concept_pairs = run_sam3_batched_with_masks(
            [img_pil], tag, batch_size=1, model=model, topn=5
        )[0]

    # Pick the best concept detection (largest mask area)
    concept_mask_np = None
    if concept_pairs:
        best_idx = 0
        best_area = 0
        for ci, (cbbox, cmask) in enumerate(concept_pairs):
            a = _mask_area(cmask)
            if a > best_area:
                best_area = a
                best_idx = ci
        best_bbox, best_mask = concept_pairs[best_idx]
        concept_mask_np = best_mask
        result_pairs.append((best_bbox, best_mask))

    # --- 2. Non-tag detection (automatic segmentation using same detector) ---
    nontag_pairs = run_nontag_segmentation(
        [img_pil], detector=detector, model=model, topn=topn_nontag,
        min_mask_area=min_mask_area,
        other_tags=all_tags, exclude_tag=tag,
    )
    nontag_list = nontag_pairs[0] if nontag_pairs else []

    # --- 3. Subtract concept mask from each non-tag mask + filter ---
    for nt_bbox, nt_mask in nontag_list:
        if nt_mask is not None and concept_mask_np is not None:
            trimmed = _subtract_mask(nt_mask, concept_mask_np)
            if _mask_area(trimmed) < min_mask_area:
                continue  # discard: nothing left after subtraction
            result_pairs.append((nt_bbox, trimmed))
        elif nt_mask is not None:
            if _mask_area(nt_mask) >= min_mask_area:
                result_pairs.append((nt_bbox, nt_mask))
        else:
            result_pairs.append((nt_bbox, None))

    return result_pairs


def _combine_concept_and_nontag_pairs(
    concept_pairs: List[Tuple[List[int], Any]],
    nontag_pairs: List[Tuple[List[int], Any]],
    min_mask_area: int = 100,
) -> Tuple[List[Tuple[List[int], Any]], bool]:
    """Combine tag-based concept pairs with non-tag segmentation pairs (non-overlapping).

    The best concept detection (largest mask area) takes priority.
    Non-tag masks have the concept area subtracted.

    Returns:
        (combined_pairs, has_concept_mask)
    """
    result_pairs: List[Tuple[List[int], Any]] = []
    concept_mask_np = None

    # Pick the best concept detection (largest mask area)
    if concept_pairs:
        best_idx = 0
        best_area = 0
        for ci, (cbbox, cmask) in enumerate(concept_pairs):
            a = _mask_area(cmask)
            if a > best_area:
                best_area = a
                best_idx = ci
        best_bbox, best_mask = concept_pairs[best_idx]
        concept_mask_np = best_mask
        result_pairs.append((best_bbox, best_mask))

    # Subtract concept mask from each non-tag mask + filter
    for nt_bbox, nt_mask in nontag_pairs:
        if nt_mask is not None and concept_mask_np is not None:
            trimmed = _subtract_mask(nt_mask, concept_mask_np)
            if _mask_area(trimmed) < min_mask_area:
                continue
            result_pairs.append((nt_bbox, trimmed))
        elif nt_mask is not None:
            if _mask_area(nt_mask) >= min_mask_area:
                result_pairs.append((nt_bbox, nt_mask))
        else:
            result_pairs.append((nt_bbox, None))

    return result_pairs, len(concept_pairs) > 0


def build_concept_yolo_detections_for_image(
    img_pil,
    tag: str,
    concept_detector: str,
    concept_model: Any,
    yolo_model: Any,
    batch_size: int = 8,
    topn_yolo: int = 10,
    min_mask_area: int = 100,
) -> List[Tuple[List[int], Any]]:
    """Build per-tag detection list: ONE concept mask + multiple YOLO masks (non-overlapping).

    Only used when detector='yolo_seg' and a separate concept detector is needed.
    For langsam/sam3, use ``build_concept_and_nontag_detections_for_image`` instead.

    Args:
        img_pil: PIL Image (already resized to virtual size).
        tag: The concept tag for the tag detector.
        concept_detector: 'langsam' or 'sam3'.
        concept_model: Preloaded concept detector model.
        yolo_model: Preloaded YOLO model.
        batch_size: Batch size for detection.
        topn_yolo: Max YOLO detections per image.
        min_mask_area: Minimum pixels for a trimmed YOLO mask to be kept.

    Returns:
        List of (bbox_xywh, mask_np_or_None) tuples.
    """
    result_pairs: List[Tuple[List[int], Any]] = []

    # --- 1. Concept detection (tag-specific) ---
    concept_pairs: List[Tuple[List[int], Any]] = []
    if concept_detector == "langsam":
        concept_pairs = run_langsam_batched_with_masks(
            [img_pil], tag, batch_size=1, model=concept_model, topn=5
        )[0]
    elif concept_detector == "sam3":
        concept_pairs = run_sam3_batched_with_masks(
            [img_pil], tag, batch_size=1, model=concept_model, topn=5
        )[0]

    # Pick the best concept detection (largest mask area)
    concept_mask_np = None
    if concept_pairs:
        best_idx = 0
        best_area = 0
        for ci, (cbbox, cmask) in enumerate(concept_pairs):
            a = _mask_area(cmask)
            if a > best_area:
                best_area = a
                best_idx = ci
        best_bbox, best_mask = concept_pairs[best_idx]
        concept_mask_np = best_mask
        result_pairs.append((best_bbox, best_mask))

    # --- 2. YOLO detection (tag-free, all objects) ---
    yolo_pairs_per_img = run_yolo_all_objects_with_masks(
        [img_pil], batch_size=1, model=yolo_model, topn=topn_yolo
    )
    yolo_pairs = yolo_pairs_per_img[0] if yolo_pairs_per_img else []

    # --- 3. Subtract concept mask from each YOLO mask + filter ---
    for yolo_bbox, yolo_mask in yolo_pairs:
        if yolo_mask is not None and concept_mask_np is not None:
            trimmed = _subtract_mask(yolo_mask, concept_mask_np)
            if _mask_area(trimmed) < min_mask_area:
                continue
            result_pairs.append((yolo_bbox, trimmed))
        elif yolo_mask is not None:
            if _mask_area(yolo_mask) >= min_mask_area:
                result_pairs.append((yolo_bbox, yolo_mask))
        else:
            result_pairs.append((yolo_bbox, None))

    return result_pairs


def load_mapping(json_file: str) -> Dict[str, List[str]]:
    with open(json_file, 'r') as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if isinstance(v, list)}


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


# ------------------------
# Core processors that build JSON
# ------------------------

def _record_for_image(
    tag_bucket: Dict[str, dict],
    rel_path: str,
    image_size: Tuple[int, int],
    patch_size: int,
    detections_masks_rle: Optional[List[Optional[dict]]] = None,
    concept_mask_index: Optional[int] = None,
    max_masks_per_image: int = 0,
):
    """Record mask data for one image.

    Primary data: ``masks_rle`` — list of ``{"rle": ..., "is_concept": bool}``.
    Only RLE masks are stored; bounding-box crops are no longer used.

    Args:
        max_masks_per_image: Maximum total masks (including concept mask).
            0 means unlimited. The concept mask is always preserved.
    """
    w, h = image_size
    entry: Dict[str, Any] = {
        "meta": {
            "image_size": list(image_size),
            "patch_size": patch_size,
            "virtual_w": w,
            "virtual_h": h,
        },
    }

    # Build the mask-centric list
    masks_list: List[Dict[str, Any]] = []
    if detections_masks_rle is not None:
        for mi, mask_rle in enumerate(detections_masks_rle):
            masks_list.append({
                "rle": mask_rle,
                "is_concept": (concept_mask_index is not None and mi == concept_mask_index),
            })

    # Enforce max_masks_per_image limit (concept mask always kept)
    if max_masks_per_image > 0 and len(masks_list) > max_masks_per_image:
        concept_entry = None
        non_concept_entries: List[Dict[str, Any]] = []
        for m in masks_list:
            if m.get("is_concept", False):
                concept_entry = m
            else:
                non_concept_entries.append(m)
        if concept_entry is not None:
            # Keep concept mask + fill remaining slots with non-concept masks
            kept = non_concept_entries[: max_masks_per_image - 1]
            masks_list = [concept_entry] + kept
            # Update concept_mask_index to 0 since concept is now first
            concept_mask_index = 0
        else:
            masks_list = masks_list[:max_masks_per_image]

    if concept_mask_index is not None:
        entry["meta"]["concept_mask_index"] = concept_mask_index

    entry["masks_rle"] = masks_list
    tag_bucket[rel_path] = entry


def process_folder_structure_to_json(
    root_input: str,
    patch_size: int = 128,
    object_detection: bool = False,
    detector: str = "langsam",
    batch_size: int = 8,
    topn: int = 10,
    device: Optional[str] = None,
    image_size_width: Optional[int] = None,
    result: Optional[Dict[str, dict]] = None,
    output_json: Optional[str] = None,
    verbose: bool = False,
    show_progress: bool = True,
    max_masks_per_image: int = 0,
) -> Dict[str, dict]:
    # Build per-tag image lists (tag = immediate subfolder name of the image path)
    tag_to_paths: Dict[str, List[str]] = {}
    for subdir, _, files in os.walk(root_input):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                p = os.path.join(subdir, f)
                tag = Path(subdir).name
                tag_to_paths.setdefault(tag, []).append(p)

    # Count total images for progress bar
    total_images = sum(len(paths) for paths in tag_to_paths.values())
    
    # Run object detection per tag group (optionally on resized images in memory)
    # detections_map: image_path -> List[(bbox_xywh, mask_or_None)]
    detections_map: Dict[str, List[Tuple[List[int], Any]]] = {}
    if object_detection:
        # Load model once and reuse across all tags
        if detector == "langsam":
            model = _load_langsam_model(device=device)
        elif detector == "sam3":
            model = _load_sam3_model(device=device)
        elif detector == "yolo_seg":
            model = _load_yolo_model(device=device)
        else:
            model = None

        # For langsam/sam3: pre-compute non-tag segmentation for ALL images
        # Uses SAM's own automatic segmentation (no YOLO needed)
        nontag_detections_map: Dict[str, List[Tuple[List[int], Any]]] = {}
        if detector in ("langsam", "sam3"):
            print(f"[combined] Using {detector} for both tag-based and non-tag segmentation")
            all_paths = sorted({p for paths in tag_to_paths.values() for p in paths})
            if all_paths:
                # Pass resized images to SAM for lower memory usage
                det_images_nt: List[Any] = []
                for p in all_paths:
                    img_or_path, _, _ = _load_and_resize_image(p, image_size_width)
                    det_images_nt.append(img_or_path)
                try:
                    nontag_pairs = run_nontag_segmentation(
                        det_images_nt, detector=detector, model=model,
                        topn=topn, min_mask_area=100, batch_size=batch_size,
                    )
                    nontag_detections_map = {p: pairs for p, pairs in zip(all_paths, nontag_pairs)}
                except Exception as e:
                    print(f"Warning: non-tag segmentation failed: {e}")
                    nontag_detections_map = {p: [] for p in all_paths}
                del det_images_nt
                _cleanup_gpu_memory()

        if detector == "yolo_seg":
            # YOLO is tag-free: detect all objects once across ALL images
            all_paths = sorted({p for paths in tag_to_paths.values() for p in paths})
            if all_paths:
                det_images: List[Any] = []
                for p in all_paths:
                    img_or_path, _, _ = _load_and_resize_image(p, image_size_width)
                    det_images.append(img_or_path)
                try:
                    yolo_pairs = run_yolo_all_objects_with_masks(
                        det_images, batch_size=batch_size, model=model, topn=topn,
                    )
                    detections_map = {p: pairs for p, pairs in zip(all_paths, yolo_pairs)}
                except Exception as e:
                    print(f"Warning: YOLO detection failed: {e}")
                    detections_map = {p: [] for p in all_paths}
                del det_images
                _cleanup_gpu_memory()
        else:
            # LangSAM / SAM3: detect per tag (text-prompted) on RESIZED images
            det_pbar = tqdm(
                tag_to_paths.items(),
                desc=f"Detection ({detector})",
                unit="tag",
                disable=not (show_progress and TQDM_AVAILABLE),
            ) if TQDM_AVAILABLE else tag_to_paths.items()

            for tag, paths in det_pbar:
                if not paths:
                    continue
                if TQDM_AVAILABLE and show_progress:
                    det_pbar.set_postfix(tag=tag[:15], images=len(paths))
                try:
                    det_images: List[Any] = []
                    orig_paths: List[str] = []
                    for p in paths:
                        # Always pass resized image to SAM for lower memory usage
                        img_or_path, _, _ = _load_and_resize_image(p, image_size_width)
                        det_images.append(img_or_path)
                        orig_paths.append(p)

                    concept_pairs_list = run_detector_batched_with_masks(
                        det_images, tag=tag, detector=detector,
                        batch_size=batch_size, model=model, topn=topn,
                    )
                    # Combine concept (tag-based) pairs with non-tag pairs
                    for op, concept_pairs in zip(orig_paths, concept_pairs_list):
                        nontag_for_img = nontag_detections_map.get(op, [])
                        if nontag_for_img:
                            combined, _ = _combine_concept_and_nontag_pairs(
                                concept_pairs, nontag_for_img, min_mask_area=100,
                            )
                            detections_map[op] = combined
                        else:
                            detections_map[op] = concept_pairs
                    del det_images
                    _cleanup_gpu_memory()
                except Exception as e:
                    print(f"Warning: detection failed for tag '{tag}' with detector={detector}, batch_size={batch_size}: {e}")
                    for p in paths:
                        detections_map[p] = []

    result = result or {}

    # Progress bar for crop generation phase
    crop_pbar = tqdm(
        total=total_images,
        desc="Generating crops",
        unit="img",
        disable=not (show_progress and TQDM_AVAILABLE),
    ) if TQDM_AVAILABLE else None

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
            
            # Update progress bar
            if crop_pbar is not None:
                crop_pbar.update(1)
                crop_pbar.set_postfix(tag=tag[:12], det=len(detections_map.get(image_path, [])))
            
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

            detections_pairs = detections_map.get(image_path, []) if object_detection else []
            masks_rle = _pairs_to_rle_masks(detections_pairs) if detections_pairs else None

            _record_for_image(
                tag_bucket,
                rel_path,
                (w, h),
                patch_size,
                detections_masks_rle=masks_rle,
                max_masks_per_image=max_masks_per_image,
            )
            processed += 1
            if output_json:
                _atomic_write_json(output_json, result)

        if verbose:
            print(f"[folder] tag='{tag}': total_files={len(paths)} processed={processed} size_fail={size_fail} already_done={skipped_existing}")

    # Close progress bar
    if crop_pbar is not None:
        crop_pbar.close()

    return result


def process_json_mapping_to_json(
    json_file: str,
    input_root: str,
    patch_size: int = 128,
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
    show_progress: bool = True,
    max_masks_per_image: int = 0,
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
        elif detector == "yolo_seg":
            detection_model = _load_yolo_model(device=device)

    # Pre-compute non-tag segmentation across all unique images
    # For langsam/sam3: uses their own automatic segmentation (no YOLO needed)
    # For yolo_seg: uses YOLO instance segmentation
    nontag_detections_map: Dict[str, List[Tuple[List[int], Any]]] = {}
    if object_detection and detector in ("yolo_seg", "langsam", "sam3"):
        if detector in ("langsam", "sam3"):
            print(f"[combined] Using {detector} for both tag-based and non-tag segmentation")
        all_abs_paths = set()
        for tag_key, rels in mapping.items():
            if not isinstance(rels, list) or len(rels) < min_images_per_tag:
                continue
            for rel in rels:
                p = os.path.join(input_root, rel)
                if os.path.isfile(p):
                    all_abs_paths.add(p)
        all_abs_paths_list = sorted(all_abs_paths)
        if all_abs_paths_list:
            # Always pass resized images for lower memory usage
            det_images_nt: List[Any] = []
            for p in all_abs_paths_list:
                img_or_path, _, _ = _load_and_resize_image(p, image_size_width)
                det_images_nt.append(img_or_path)
            try:
                if detector == "yolo_seg":
                    nt_pairs = run_yolo_all_objects_with_masks(
                        det_images_nt, batch_size=batch_size, model=detection_model, topn=topn,
                    )
                else:
                    nt_pairs = run_nontag_segmentation(
                        det_images_nt, detector=detector, model=detection_model,
                        topn=topn, min_mask_area=100, batch_size=batch_size,
                    )
                nontag_detections_map = {p: pairs for p, pairs in zip(all_abs_paths_list, nt_pairs)}
            except Exception as e:
                print(f"Warning: non-tag segmentation failed: {e}")
                nontag_detections_map = {p: [] for p in all_abs_paths_list}
            del det_images_nt
            _cleanup_gpu_memory()

    # Count total images for progress
    total_images = 0
    valid_tags = []
    for tag, rels in mapping.items():
        if not isinstance(rels, list):
            continue
        if len(rels) < min_images_per_tag:
            continue
        tag_rels = rels
        if max_images_per_tag > 0 and len(rels) > max_images_per_tag:
            tag_rels = rels[:max_images_per_tag]  # Approximation for counting
        total_images += len(tag_rels)
        valid_tags.append(tag)

    # Progress bar for detection phase
    if object_detection and TQDM_AVAILABLE and show_progress:
        det_pbar = tqdm(
            total=len(valid_tags),
            desc=f"Detection ({detector})",
            unit="tag",
        )
    else:
        det_pbar = None

    # Progress bar for crop generation phase
    crop_pbar = tqdm(
        total=total_images,
        desc="Generating crops",
        unit="img",
        disable=not (show_progress and TQDM_AVAILABLE),
    ) if TQDM_AVAILABLE else None

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
        detections_pairs_map: Dict[str, List[Tuple[List[int], Any]]] = {}
        if object_detection and abs_paths:
            if det_pbar is not None:
                det_pbar.update(1)
                det_pbar.set_postfix(tag=tag[:15], images=len(abs_paths))
            if detector == "yolo_seg":
                # YOLO: use pre-computed cache (all objects, tag-free)
                detections_pairs_map = {p: nontag_detections_map.get(p, []) for p in abs_paths}
            else:
                # LangSAM / SAM3: tag-based detection on RESIZED images + non-tag
                try:
                    det_images: List[Any] = []
                    orig_paths: List[str] = []
                    for p in abs_paths:
                        # Always pass resized image to SAM for lower memory usage
                        img_or_path, _, _ = _load_and_resize_image(p, image_size_width)
                        det_images.append(img_or_path)
                        orig_paths.append(p)
                    concept_pairs_list = run_detector_batched_with_masks(
                        det_images, tag=tag, detector=detector,
                        batch_size=batch_size, model=detection_model, topn=topn,
                    )
                    # Combine concept (tag-based) pairs with non-tag pairs
                    for op, concept_pairs in zip(orig_paths, concept_pairs_list):
                        nontag_for_img = nontag_detections_map.get(op, [])
                        if nontag_for_img:
                            combined, _ = _combine_concept_and_nontag_pairs(
                                concept_pairs, nontag_for_img, min_mask_area=100,
                            )
                            detections_pairs_map[op] = combined
                        else:
                            detections_pairs_map[op] = concept_pairs
                    del det_images
                    _cleanup_gpu_memory()
                except Exception as e:
                    print(f"Warning: detection failed for tag '{tag}' with detector={detector}, batch_size={batch_size}: {e}")
                    detections_pairs_map = {p: [] for p in abs_paths}

        tag_bucket: Dict[str, dict] = result.setdefault(tag, {})
        processed = 0
        skipped_existing = 0
        size_fail = 0

        for rel in rels:
            img_path = os.path.join(input_root, rel)
            
            # Update progress bar
            if crop_pbar is not None:
                crop_pbar.update(1)
                crop_pbar.set_postfix(tag=tag[:12], det=len(detections_pairs_map.get(img_path, [])))
            
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

            detections_pairs = detections_pairs_map.get(img_path, []) if object_detection else []
            masks_rle = _pairs_to_rle_masks(detections_pairs) if detections_pairs else None

            _record_for_image(
                tag_bucket,
                rel,
                (w, h),
                patch_size,
                detections_masks_rle=masks_rle,
                max_masks_per_image=max_masks_per_image,
            )
            processed += 1
            if output_json:
                _atomic_write_json(output_json, result)

        if verbose:
            print(f"[mapping] tag='{tag}': candidates={len(rels)} found_files={len(abs_paths)} processed={processed} size_fail={size_fail} already_done={skipped_existing}")

    # Close progress bars
    if det_pbar is not None:
        det_pbar.close()
    if crop_pbar is not None:
        crop_pbar.close()

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
    verbose: bool = False,
    show_progress: bool = True,
    max_masks_per_image: int = 0,
) -> Dict[str, dict]:
    """Build per-tag bags of masks/crops.

    Combined framework (all detectors: ``yolo_seg``, ``langsam``, ``sam3``):
      For each (tag, image) kept:
        1. Generate exactly ONE concept mask using the tag detector (LangSAM/SAM3)
        2. Generate non-tag masks using the SAME detector's automatic segmentation
        3. Enforce strict NON-OVERLAP across ALL final masks (concept mask has priority)
        4. Save output in strict JSON structure

    For langsam/sam3: both tag-based and non-tag segmentation use the same model.
    No YOLO needed — SAM/LangSAM can do segmentation without a prior tag.
    Resized images are always passed to SAM/LangSAM for lower memory usage.
    """
    mapping = load_mapping(json_file)
    rng = random
    result = result or {}

    # --- Determine which concept detector to use ---
    # When detector is langsam/sam3, use that directly (same model for tag + non-tag).
    # When detector is yolo_seg, read CONCEPT_DETECTOR env var (default: langsam).
    if detector in ("langsam", "sam3"):
        concept_detector_name: str = detector
    else:
        concept_detector_name: str = os.environ.get("CONCEPT_DETECTOR", "langsam")
    use_combined = object_detection and detector in ("yolo_seg", "langsam", "sam3")

    # --- Preload models ---
    concept_model: Optional[Any] = None
    yolo_model: Optional[Any] = None
    detection_model: Optional[Any] = None

    if object_detection:
        if use_combined:
            # Load the concept detector (same model handles both tag + non-tag)
            if concept_detector_name == "langsam":
                concept_model = _load_langsam_model(device=device)
            elif concept_detector_name == "sam3":
                concept_model = _load_sam3_model(device=device)
            else:
                concept_model = _load_langsam_model(device=device)
                concept_detector_name = "langsam"
            # Only load YOLO if detector is yolo_seg
            if detector == "yolo_seg":
                yolo_model = _load_yolo_model(device=device)
            print(f"[concept_combined] Using {concept_detector_name} for tag + non-tag segmentation")

    # --- Count total images for progress ---
    total_images = 0
    valid_tags = []
    for tag, rel_paths in mapping.items():
        if not isinstance(rel_paths, list):
            continue
        if len(rel_paths) < min_images_per_tag:
            continue
        tag_rels = rel_paths
        if max_images_per_tag > 0 and len(rel_paths) > max_images_per_tag:
            tag_rels = rel_paths[:max_images_per_tag]
        total_images += len(tag_rels)
        valid_tags.append(tag)

    # Progress bars
    det_pbar = None
    if object_detection and TQDM_AVAILABLE and show_progress:
        label = f"Detection ({detector}+{concept_detector_name})" if use_combined else f"Detection ({detector})"
        det_pbar = tqdm(total=len(valid_tags), desc=label, unit="tag")

    crop_pbar = tqdm(
        total=total_images,
        desc="Generating crops",
        unit="img",
        disable=not (show_progress and TQDM_AVAILABLE),
    ) if TQDM_AVAILABLE else None

    for tag, rel_paths in mapping.items():
        if not isinstance(rel_paths, list):
            continue
        if len(rel_paths) < min_images_per_tag:
            continue
        if max_images_per_tag > 0 and len(rel_paths) > max_images_per_tag:
            rel_paths = rng.sample(rel_paths, max_images_per_tag)

        # --- Build detection pairs for this tag ---
        detections_pairs_map: Dict[str, List[Tuple[List[int], Any]]] = {}
        # Track which images have a concept mask (index 0 in the pairs list)
        concept_mask_flags: Dict[str, bool] = {}
        abs_paths: List[str] = []

        if object_detection:
            for rel_path in rel_paths:
                abs_p = os.path.join(input_root, rel_path)
                if os.path.isfile(abs_p):
                    abs_paths.append(abs_p)

            if abs_paths:
                if det_pbar is not None:
                    det_pbar.update(1)
                    det_pbar.set_postfix(tag=tag[:15], images=len(abs_paths))

                if use_combined:
                    # ====== COMBINED: concept mask + non-tag masks (same detector) ======
                    for p in abs_paths:
                        # Pass resized image for lower memory usage
                        img_pil, _, _ = _load_and_resize_image(p, image_size_width)
                        try:
                            if detector == "yolo_seg":
                                # YOLO + concept detector
                                pairs = build_concept_yolo_detections_for_image(
                                    img_pil=img_pil,
                                    tag=tag,
                                    concept_detector=concept_detector_name,
                                    concept_model=concept_model,
                                    yolo_model=yolo_model,
                                    batch_size=batch_size,
                                    topn_yolo=topn,
                                    min_mask_area=100,
                                )
                            else:
                                # LangSAM/SAM3: same model for tag + non-tag
                                pairs = build_concept_and_nontag_detections_for_image(
                                    img_pil=img_pil,
                                    tag=tag,
                                    detector=concept_detector_name,
                                    model=concept_model,
                                    topn_nontag=topn,
                                    min_mask_area=100,
                                    all_tags=valid_tags,
                                )
                            detections_pairs_map[p] = pairs
                            # First pair is the concept mask if it exists
                            concept_mask_flags[p] = len(pairs) > 0
                        except Exception as e:
                            print(f"Warning: combined detection failed for tag='{tag}', img='{p}': {e}")
                            detections_pairs_map[p] = []
                            concept_mask_flags[p] = False
                    _cleanup_gpu_memory()
                else:
                    # ====== Single detector mode (langsam / sam3) ======
                    try:
                        det_images: List[Any] = []
                        orig_paths: List[str] = []
                        for p in abs_paths:
                            img_or_path, _, _ = _load_and_resize_image(p, image_size_width)
                            det_images.append(img_or_path)
                            orig_paths.append(p)
                        pairs_list = run_detector_batched_with_masks(
                            det_images, tag=tag, detector=detector,
                            batch_size=batch_size, model=detection_model, topn=topn,
                        )
                        detections_pairs_map = {op: pairs for op, pairs in zip(orig_paths, pairs_list)}
                        del det_images
                        _cleanup_gpu_memory()
                    except Exception as e:
                        print(f"Warning: detection failed for tag '{tag}' with detector={detector}: {e}")
                        detections_pairs_map = {}

        # --- Generate crop records per image ---
        tag_bucket: Dict[str, dict] = result.setdefault(tag, {})
        processed = 0
        skipped_existing = 0
        size_fail = 0

        for rel_path in rel_paths:
            img_path = os.path.join(input_root, rel_path)

            if crop_pbar is not None:
                crop_pbar.update(1)
                crop_pbar.set_postfix(tag=tag[:12], det=len(detections_pairs_map.get(img_path, [])))

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

            w, h, scale = _compute_virtual_resize(orig_w, orig_h, image_size_width)

            detections_pairs = detections_pairs_map.get(img_path, []) if object_detection else []
            masks_rle = _pairs_to_rle_masks(detections_pairs) if detections_pairs else None

            # Mark concept mask index (0 if concept detector found it)
            cmi = 0 if (use_combined and concept_mask_flags.get(img_path, False)) else None

            _record_for_image(
                tag_bucket,
                rel_path,
                (w, h),
                patch_size,
                detections_masks_rle=masks_rle,
                concept_mask_index=cmi,
                max_masks_per_image=max_masks_per_image,
            )
            processed += 1
            if output_json:
                _atomic_write_json(output_json, result)

        if verbose:
            print(f"[concept] tag='{tag}': candidates={len(rel_paths)} processed={processed} size_fail={size_fail} already_done={skipped_existing}")

    if det_pbar is not None:
        det_pbar.close()
    if crop_pbar is not None:
        crop_pbar.close()

    return result


# ------------------------
# CLI
# ------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect detection mask coordinates to JSON (folder or JSON mapping). "
            "Uses only RLE masks from object detectors for feature extraction."
        )
    )
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_json", required=True, help="Path to write the coordinates JSON")
    parser.add_argument("--patch_size", type=int, default=200, help="Square patch size")
    parser.add_argument("--json_mapping", type=str, default=None, help="Tag -> [relative paths] JSON")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--image_size_width",
        type=int,
        default=None,
        help=(
            "Reference width for resizing (record-only). For each image, compute resized size "
            "(image_size_width, round(orig_h * image_size_width / orig_w)). If an object detector is enabled, detection runs on a "
            "cached resized copy so detector masks are produced directly in resized coordinates."
        ),
    )

    # Concept-focused parameters
    parser.add_argument("--concept_mode", action="store_true", help="Enable concept-focused cropping logic")
    parser.add_argument("--concept_crops_per_image", type=int, default=3, help="Crops per image in concept mode")
    parser.add_argument("--min_images_per_tag", type=int, default=30, help="Minimum images required per tag (JSON mapping modes)")
    parser.add_argument("--max_images_per_tag", type=int, default=0, help="Cap images per tag (0 = no cap)")
    parser.add_argument("--topn", type=int, default=10, help="Limit detector to top-N boxes per image (default 10)")

    # Optional object detection
    parser.add_argument("--object_detector", type=str, default="none", choices=["none", "langsam", "sam3", "yolo_seg"],
                        help="Object detector: 'none' (random crops only), 'langsam', 'sam3', or 'yolo_seg' (tag-free YOLO instance seg)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for object detection")
    parser.add_argument("--verbose", action="store_true", help="Print per-tag diagnostics and hints")
    parser.add_argument("--device", type=str, default=None, help="Device for detection: cpu, cuda or cuda:N")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for SAM3 detection")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    parser.add_argument("--max_masks_per_image", type=int, default=0,
                        help="Maximum number of segmentation masks (including concept mask) per image. 0 = unlimited.")

    args = parser.parse_args()
    
    # Check tqdm availability
    if not TQDM_AVAILABLE and not args.no_progress:
        print("Note: tqdm not installed. Install with 'pip install tqdm' for progress bars.")
    
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
            verbose=args.verbose,
            image_size_width=args.image_size_width,
            show_progress=not args.no_progress,
            max_masks_per_image=args.max_masks_per_image,
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
                show_progress=not args.no_progress,
                max_masks_per_image=args.max_masks_per_image,
            )
            _atomic_write_json(args.output_json, result)
        else:
            # Folder mode: flush after each image processed
            result = process_folder_structure_to_json(
                root_input=args.input_root,
                patch_size=args.patch_size,
                object_detection=object_detection_enabled,
                detector=detector,
                batch_size=args.batch_size,
                topn=args.topn,
                device=args.device,
                result=result,
                output_json=args.output_json,
                verbose=args.verbose,
                image_size_width=args.image_size_width,
                show_progress=not args.no_progress,
                max_masks_per_image=args.max_masks_per_image,
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
