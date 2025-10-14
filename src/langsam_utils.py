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
    _ = device  # Placeholder for potential future use
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
    """Normalize incoming boxes to integer [x,y,w,h] clamped to image bounds."""
    W, H = image_size
    arr = _to_numpy(boxes)
    arr = np.asarray(arr).reshape(-1, 4)
    out: List[List[int]] = []
    for x, y, w, h in arr:
        out.append(_clamp_bbox_xywh(float(x), float(y), float(w), float(h), W, H))
    return out


def _extract_bboxes_from_result(result: dict, image_size: tuple, prefer_masks: bool = False) -> List[List[int]]:
    boxes = result.get("boxes")
    masks = result.get("masks")
    scores = result.get("scores")

    if scores is not None:
        #scores = _to_numpy(scores).reshape(-1)  # ensure 1D
        idx = np.argsort(-scores, kind="stable")

        if boxes is not None:
            #boxes_np = _to_numpy(boxes)
            if boxes.shape[0] == scores.shape[0]:
                boxes = boxes[idx]

        if masks is not None:
            #masks_np = _to_numpy(masks)
            # Expect masks with shape (N, H, W) or (N, ...)
            if masks.shape[0] == scores.shape[0]:
                masks = masks[idx]

    if prefer_masks and masks is not None:
        bbs = _bboxes_from_masks(masks, image_size)
        if bbs:  # only fall back if masks empty
            return bbs

    if boxes is not None:
        return _bboxes_from_xywh_boxes(boxes, image_size)

    return []


def predict_bboxes_for_tag(
    model: LangSAM,
    images: Sequence[Union[str, Image.Image, np.ndarray]],
    tag: str,
    prefer_masks: bool = True,
) -> List[List[List[int]]]:
    """
    Predict bounding boxes for a tag over a list of images.

    Args:
        model: LangSAM model instance.
        images: Sequence of image paths, PIL Images, or numpy arrays.
        tag: Text prompt for the object name (e.g., "apple").
        prefer_masks: If True, compute tight boxes from masks; otherwise rely on model boxes.

    Returns:
        A list of length len(images). Each element is a list of [x, y, w, h] integer bboxes
        for the corresponding image; can be empty if nothing is detected.
    """
    images_pil = [_to_pil(im) for im in images]

    all_bboxes: List[List[List[int]]] = []
    for img in images_pil:
        # Predict for a single image to keep API compatibility across versions
        results = model.predict([img], [tag])
        # Results is expected to be a list (per image). Gather all detections for this image.
        img_bbs: List[List[int]] = []
        if isinstance(results, dict):
            results = [results]
        for res in results:
            img_bbs.extend(_extract_bboxes_from_result(res, img.size, prefer_masks))
        all_bboxes.append(img_bbs)

    return all_bboxes


def predict_bboxes_for_tag_batched(
    model: LangSAM,
    images: Sequence[Union[str, Image.Image, np.ndarray]],
    tag: str,
    prefer_masks: bool = True,
    batch_size: int = 8,
    topn: int = 2,
) -> List[List[List[int]]]:
    """
    Batched version of predict_bboxes_for_tag for faster throughput.

    Processes images in chunks (batch_size) and calls model.predict on each chunk.

    Returns: Same as predict_bboxes_for_tag.
    """
    images_pil = [_to_pil(im) for im in images]
    N = len(images_pil)
    all_bboxes: List[List[List[int]]] = [[] for _ in range(N)]

    def chunked(seq, n):
        for i in range(0, len(seq), n):
            yield i, seq[i : i + n]

    for start, imgs_chunk in chunked(images_pil, batch_size):
        tags_chunk = [tag] * len(imgs_chunk)
        try:
            results = model.predict(imgs_chunk, tags_chunk)
        except Exception:
            # Fallback to per-image if batch call not supported
            # Log the message that batch size or overflowed using single batch
            print(f"Probably batch size {batch_size} overflowed; using single image prediction.")
            for idx, im in enumerate(imgs_chunk, start=start):
                single_res = model.predict([im], [tag])
                img_bbs: List[List[int]] = []
                if isinstance(single_res, dict):
                    single_res = [single_res]
                for res in single_res:
                    img_bbs.extend(_extract_bboxes_from_result(res, im.size, prefer_masks))
                all_bboxes[idx] = img_bbs
            continue

        # Parse batch results. Common case: list of length == len(imgs_chunk)
        if isinstance(results, list) and len(results) == len(imgs_chunk):
            for off, (im, res_i) in enumerate(zip(imgs_chunk, results)):
                img_bbs: List[List[int]] = []
                if isinstance(res_i, dict):
                    img_bbs.extend(_extract_bboxes_from_result(res_i, im.size, prefer_masks))
                elif isinstance(res_i, (list, tuple)):
                    for r in res_i:
                        if isinstance(r, dict):
                            img_bbs.extend(_extract_bboxes_from_result(r, im.size, prefer_masks))
                all_bboxes[start + off] = img_bbs
        else:
            # Unexpected shape; fallback per image for this chunk
            for idx, im in enumerate(imgs_chunk, start=start):
                single_res = model.predict([im], [tag])
                img_bbs: List[List[int]] = []
                if isinstance(single_res, dict):
                    single_res = [single_res]
                for res in single_res:
                    img_bbs.extend(_extract_bboxes_from_result(res, im.size, prefer_masks))
                all_bboxes[idx] = img_bbs

    return all_bboxes


__all__ = [
    "load_langsam",
    "predict_bboxes_for_tag",
    "predict_bboxes_for_tag_batched",
]
