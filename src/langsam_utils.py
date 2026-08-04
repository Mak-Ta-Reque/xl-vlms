"""
LangSAM utilities for extracting bounding boxes by text tag.

Public API:
- load_langsam(device: Optional[str] = None, **kwargs) -> LangSAM
- predict_bboxes_for_tag(model, images, tag, prefer_masks=True) -> List[List[List[int]]]
- predict_bboxes_for_tag_batched(model, images, tag, prefer_masks=True, batch_size=8) -> List[List[List[int]]]

Each item of the returned outer list corresponds to an input image.
Each inner list contains zero or more [x, y, w, h] integer boxes for that image.
"""

from __future__ import annotations

from typing import List, Union, Optional, Sequence

import numpy as np
from PIL import Image

try:  # Optional import; module works without torch
    import torch  # type: ignore
except Exception:  # pragma: no cover - runtime convenience
    torch = None  # type: ignore

try:
    from lang_sam import LangSAM  # type: ignore
except Exception as e:  # pragma: no cover - clearer error if missing
    raise ImportError(
        "lang_sam package is required. Install via: pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git"
    ) from e


def load_langsam(device: Optional[str] = None, **kwargs) -> LangSAM:
    """
    Create and return a LangSAM model instance.

    Args:
        device: Optional device spec (e.g., "cuda", "cpu"). If unsupported, it's ignored.
        **kwargs: Forwarded to LangSAM initializer (kept for future flexibility).

    Returns:
        LangSAM model instance.
    """
    # Best-effort device control:
    # - If a specific cuda:N is requested, prefer setting the active device
    #   without breaking constructors that don't accept a 'device' kwarg.
    # - When no device is passed, DEVICE from .env is the source of truth
    #   (get_device_config(None) reads the env) instead of cuda:0.
    if not device:
        try:
            from device_utils import get_device_config
            device = str(get_device_config(None).primary_device)
        except Exception:
            device = None
    try:
        import os as _os  # local import to avoid polluting module scope
        if device:
            dev = str(device).lower()
            if dev.startswith("cuda") and torch is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
                # Try to parse index from cuda:N
                idx = None
                if ":" in dev:
                    try:
                        idx = int(dev.split(":", 1)[1])
                    except Exception:
                        idx = None
                if idx is not None and idx >= 0:
                    # Option A: set current cuda device
                    try:
                        torch.cuda.set_device(idx)  # type: ignore[attr-defined]
                    except Exception:
                        # Option B: narrow visibility to the requested index
                        _os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(idx))
            elif dev == "cpu":
                # Force CPU path where possible
                _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    except Exception:
        # Non-fatal; fall back to library defaults
        pass

    return LangSAM(**kwargs)


def _to_pil(img: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def _to_numpy(a) -> np.ndarray:
    """Safely convert tensor/array/list to numpy array."""
    if torch is not None and hasattr(a, "detach") and hasattr(a, "cpu"):  # type: ignore[attr-defined]
        return a.detach().cpu().numpy()
    return np.array(a)


def _binarize_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if mask.dtype == bool:
        return mask
    if np.issubdtype(mask.dtype, np.floating):
        return mask > threshold
    return mask > 0


def _clamp_bbox_xywh(x: float, y: float, w: float, h: float, W: int, H: int) -> List[int]:
    # Convert to inclusive right/bottom to clamp safely, then back to width/height
    x1 = x + w
    y1 = y + h
    x0 = float(np.clip(x, 0, max(W - 1, 0)))
    y0 = float(np.clip(y, 0, max(H - 1, 0)))
    xr = float(np.clip(x1, 0, W))
    yb = float(np.clip(y1, 0, H))

    left = int(np.floor(x0))
    top = int(np.floor(y0))
    right = int(np.ceil(xr))
    bottom = int(np.ceil(yb))

    # Ensure minimum size of 1x1
    if right <= left:
        right = min(W, left + 1)
    if bottom <= top:
        bottom = min(H, top + 1)

    return [left, top, right - left, bottom - top]


def _bboxes_from_masks(
    masks: Union[np.ndarray, Sequence], image_size: tuple, threshold: float = 0.5
) -> List[List[int]]:
    """
    Compute tight XYWH bboxes for each mask.

    Args:
        masks: 2D mask, stack of masks (N,H,W), list/tuple of masks, or tensor equivalents.
        image_size: (W, H) of the source image.
        threshold: Threshold for binarizing float masks.

    Returns:
        List of [x, y, w, h] ints, one per non-empty mask.
    """
    W, H = image_size

    # Normalize to a list of 2D numpy arrays
    masks_list: List[np.ndarray] = []
    if isinstance(masks, (list, tuple)):
        for m in masks:
            m_np = _to_numpy(m)
            if m_np.ndim > 2:
                m_np = np.squeeze(m_np)
            masks_list.append(m_np)
    else:
        m_np = _to_numpy(masks)
        if m_np.ndim == 3:  # (N,H,W)
            for k in range(m_np.shape[0]):
                masks_list.append(m_np[k])
        else:  # (H,W)
            masks_list.append(np.squeeze(m_np))

    bboxes: List[List[int]] = []
    for m in masks_list:
        m_bin = _binarize_mask(m, threshold)
        if not m_bin.any():
            continue
        ys, xs = np.where(m_bin)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x = float(x_min)
        y = float(y_min)
        w = float(x_max - x_min + 1)
        h = float(y_max - y_min + 1)
        bboxes.append(_clamp_bbox_xywh(x, y, w, h, W, H))
    return bboxes


def _bboxes_from_xywh_boxes(boxes: Union[np.ndarray, Sequence], image_size: tuple) -> List[List[int]]:
    """Normalize incoming xywh boxes to integer [x,y,w,h] clamped to image bounds."""
    W, H = image_size
    arr = _to_numpy(boxes)
    arr = np.asarray(arr).reshape(-1, 4)
    out: List[List[int]] = []
    for x, y, w, h in arr:
        out.append(_clamp_bbox_xywh(float(x), float(y), float(w), float(h), W, H))
    return out


def _bboxes_from_xyxy_boxes(boxes: Union[np.ndarray, Sequence], image_size: tuple) -> List[List[int]]:
    """Normalize incoming xyxy boxes to integer [x,y,w,h] clamped to image bounds."""
    W, H = image_size
    arr = _to_numpy(boxes)
    arr = np.asarray(arr).reshape(-1, 4)
    out: List[List[int]] = []
    for x1, y1, x2, y2 in arr:
        x = float(x1)
        y = float(y1)
        w = float(x2 - x1)
        h = float(y2 - y1)
        out.append(_clamp_bbox_xywh(x, y, w, h, W, H))
    return out


def _extract_bboxes_and_masks_from_result(
    result: dict, image_size: tuple, prefer_masks: bool = False
) -> List[tuple]:
    """Extract (bbox_xywh, binary_mask) tuples from a LangSAM result dict.

    Returns:
        List of (bbox_xywh: List[int], mask: np.ndarray|None) tuples.
    """
    boxes = result.get("boxes")
    masks = result.get("masks")
    scores = result.get("scores")

    # Convert to numpy arrays to ensure consistent handling
    if boxes is not None:
        boxes = _to_numpy(boxes)
    if masks is not None:
        masks = _to_numpy(masks)
    if scores is not None:
        scores = _to_numpy(scores).reshape(-1)  # ensure 1D

    if scores is not None and len(scores) > 0:
        idx = np.argsort(-scores, kind="stable")

        if boxes is not None and len(boxes) == len(scores):
            boxes = boxes[idx]

        if masks is not None and len(masks) == len(scores):
            masks = masks[idx]

    W, H = image_size

    # Build per-detection masks list (binarized at original resolution)
    mask_list: List[Optional[np.ndarray]] = []
    if masks is not None:
        # masks shape: (N, H, W) or list of 2D arrays
        masks_arr = masks if isinstance(masks, np.ndarray) and masks.ndim == 3 else None
        if masks_arr is not None:
            for k in range(masks_arr.shape[0]):
                m = _binarize_mask(masks_arr[k])
                mask_list.append(m)
        else:
            # list / tuple of 2D masks
            for m in masks:
                m_np = _to_numpy(m)
                if m_np.ndim > 2:
                    m_np = np.squeeze(m_np)
                mask_list.append(_binarize_mask(m_np))

    if prefer_masks and mask_list:
        bbs = _bboxes_from_masks(
            [m for m in mask_list if m is not None], image_size
        )
        if bbs:
            # Pair each tight bbox with its mask
            pairs = []
            for bb, m in zip(bbs, mask_list):
                pairs.append((bb, m))
            return pairs

    if boxes is not None:
        # LangSAM returns boxes in xyxy format (x1, y1, x2, y2)
        bbs = _bboxes_from_xyxy_boxes(boxes, image_size)
        # Pair bboxes with masks (if available)
        pairs = []
        for i, bb in enumerate(bbs):
            m = mask_list[i] if i < len(mask_list) else None
            pairs.append((bb, m))
        return pairs

    return []


def _extract_bboxes_from_result(result: dict, image_size: tuple, prefer_masks: bool = False) -> List[List[int]]:
    """Legacy wrapper: returns only bboxes (backward compat)."""
    pairs = _extract_bboxes_and_masks_from_result(result, image_size, prefer_masks)
    return [bb for bb, _m in pairs]


def predict_bboxes_and_masks_for_tag(
    model: LangSAM,
    images: Sequence[Union[str, Image.Image, np.ndarray]],
    tag: str,
    prefer_masks: bool = True,
) -> List[List[tuple]]:
    """
    Predict bounding boxes AND segmentation masks for a tag over a list of images.

    Returns:
        A list of length len(images). Each element is a list of
        (bbox_xywh: List[int], mask: np.ndarray|None) tuples.
    """
    images_pil = [_to_pil(im) for im in images]

    all_pairs: List[List[tuple]] = []
    for img in images_pil:
        results = model.predict([img], [tag])
        img_pairs: List[tuple] = []
        if isinstance(results, dict):
            results = [results]
        for res in results:
            img_pairs.extend(_extract_bboxes_and_masks_from_result(res, img.size, prefer_masks))
        all_pairs.append(img_pairs)

    return all_pairs


def predict_bboxes_for_tag(
    model: LangSAM,
    images: Sequence[Union[str, Image.Image, np.ndarray]],
    tag: str,
    prefer_masks: bool = True,
) -> List[List[List[int]]]:
    """
    Predict bounding boxes for a tag over a list of images (legacy API).
    """
    all_pairs = predict_bboxes_and_masks_for_tag(model, images, tag, prefer_masks)
    return [[bb for bb, _m in pairs] for pairs in all_pairs]


def predict_bboxes_and_masks_for_tag_batched(
    model: LangSAM,
    images: Sequence[Union[str, Image.Image, np.ndarray]],
    tag: str,
    prefer_masks: bool = True,
    batch_size: int = 8,
    topn: int = 2,
) -> List[List[tuple]]:
    """
    Batched: predict (bbox, mask) pairs for a tag over a list of images.

    Returns:
        List of length len(images). Each element is a list of
        (bbox_xywh: List[int], mask: np.ndarray|None) tuples.
    """
    images_pil = [_to_pil(im) for im in images]
    N = len(images_pil)
    all_pairs: List[List[tuple]] = [[] for _ in range(N)]

    def chunked(seq, n):
        for i in range(0, len(seq), n):
            yield i, seq[i : i + n]

    for start, imgs_chunk in chunked(images_pil, batch_size):
        tags_chunk = [tag] * len(imgs_chunk)
        try:
            results = model.predict(imgs_chunk, tags_chunk)
        except Exception:
            print(f"Probably batch size {batch_size} overflowed; using single image prediction.")
            for idx, im in enumerate(imgs_chunk, start=start):
                try:
                    single_res = model.predict([im], [tag])
                except Exception as single_err:
                    print(f"WARNING: LangSAM failed on image {idx} for tag '{tag}': {single_err}; skipping image.")
                    all_pairs[idx] = []
                    continue
                img_pairs: List[tuple] = []
                if isinstance(single_res, dict):
                    single_res = [single_res]
                for res in single_res:
                    img_pairs.extend(_extract_bboxes_and_masks_from_result(res, im.size, prefer_masks))
                all_pairs[idx] = img_pairs
            continue

        if isinstance(results, list) and len(results) == len(imgs_chunk):
            for off, (im, res_i) in enumerate(zip(imgs_chunk, results)):
                img_pairs: List[tuple] = []
                if isinstance(res_i, dict):
                    img_pairs.extend(_extract_bboxes_and_masks_from_result(res_i, im.size, prefer_masks))
                elif isinstance(res_i, (list, tuple)):
                    for r in res_i:
                        if isinstance(r, dict):
                            img_pairs.extend(_extract_bboxes_and_masks_from_result(r, im.size, prefer_masks))
                all_pairs[start + off] = img_pairs
        else:
            for idx, im in enumerate(imgs_chunk, start=start):
                try:
                    single_res = model.predict([im], [tag])
                except Exception as single_err:
                    print(f"WARNING: LangSAM failed on image {idx} for tag '{tag}': {single_err}; skipping image.")
                    all_pairs[idx] = []
                    continue
                img_pairs: List[tuple] = []
                if isinstance(single_res, dict):
                    single_res = [single_res]
                for res in single_res:
                    img_pairs.extend(_extract_bboxes_and_masks_from_result(res, im.size, prefer_masks))
                all_pairs[idx] = img_pairs

    return all_pairs


def predict_bboxes_for_tag_batched(
    model: LangSAM,
    images: Sequence[Union[str, Image.Image, np.ndarray]],
    tag: str,
    prefer_masks: bool = True,
    batch_size: int = 8,
    topn: int = 2,
) -> List[List[List[int]]]:
    """Legacy batched API: returns only bboxes."""
    all_pairs = predict_bboxes_and_masks_for_tag_batched(
        model, images, tag, prefer_masks, batch_size, topn
    )
    return [[bb for bb, _m in pairs] for pairs in all_pairs]


# ---------------------------------------------------------------------------
# Automatic (non-tag) segmentation via SAM2AutomaticMaskGenerator
# ---------------------------------------------------------------------------

def predict_all_masks_langsam(
    model: LangSAM,
    images: Sequence[Union[str, Image.Image, np.ndarray]],
    topn: int = 10,
    min_mask_area: int = 100,
) -> List[List[tuple]]:
    """Run automatic segmentation (no text prompt) using LangSAM's inner SAM2.

    Uses SAM2AutomaticMaskGenerator to segment *everything* in the image.

    Args:
        model: LangSAM model instance (from load_langsam).
        images: List of image paths, PIL Images, or numpy arrays.
        topn: Maximum number of masks per image (sorted by area, largest first).
        min_mask_area: Discard masks with fewer pixels than this.

    Returns:
        List of length len(images). Each element is a list of
        (bbox_xywh: List[int], mask: np.ndarray bool H×W) tuples.
    """
    images_pil = [_to_pil(im) for im in images]
    all_pairs: List[List[tuple]] = []

    for img in images_pil:
        W, H = img.size
        img_np = np.asarray(img)  # HWC uint8 RGB

        try:
            # model.sam is the inner SAM object which has .generate()
            raw_masks = model.sam.generate(img_np)
        except Exception as e:
            print(f"LangSAM automatic segmentation failed: {e}")
            all_pairs.append([])
            continue

        # raw_masks is a list of dicts with keys: segmentation, bbox, area, ...
        # Sort by area descending (largest first)
        raw_masks.sort(key=lambda m: m.get("area", 0), reverse=True)

        pairs: List[tuple] = []
        for m in raw_masks:
            seg = m.get("segmentation")
            if seg is None:
                continue
            seg_np = _to_numpy(seg).astype(bool)
            if seg_np.ndim > 2:
                seg_np = seg_np.squeeze()
            area = int(seg_np.sum())
            if area < min_mask_area:
                continue
            # bbox from SAM2 is [x, y, w, h]
            bbox = m.get("bbox")
            if bbox is not None:
                bbox_xywh = _clamp_bbox_xywh(
                    float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), W, H
                )
            else:
                # Derive bbox from mask
                ys, xs = np.where(seg_np)
                bbox_xywh = _clamp_bbox_xywh(
                    float(xs.min()), float(ys.min()),
                    float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1),
                    W, H,
                )
            pairs.append((bbox_xywh, seg_np))
            if len(pairs) >= topn:
                break

        all_pairs.append(pairs)

    return all_pairs


__all__ = [
    "load_langsam",
    "predict_bboxes_for_tag",
    "predict_bboxes_for_tag_batched",
    "predict_bboxes_and_masks_for_tag",
    "predict_bboxes_and_masks_for_tag_batched",
    "predict_all_masks_langsam",
]
