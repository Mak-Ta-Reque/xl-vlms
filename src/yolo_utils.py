"""
YOLO utilities for tag-free object detection with instance segmentation.

Public API:
- load_yolo(device=None, confidence_threshold=0.5) -> Dict[str, Any]
- predict_all_objects(model_dict, images, batch_size=8, topn=10) -> List[List[Dict]]
- encode_mask_rle(mask: np.ndarray) -> Dict[str, Any]
- decode_mask_rle(rle_dict: Dict[str, Any]) -> np.ndarray

YOLO is tag-free: it detects ALL objects in an image regardless of class.
The VLM downstream determines which detections are semantically relevant.

Each detection dict contains:
    class_name: str          - COCO class name (e.g., "dog", "car")
    bbox_xywh: [x, y, w, h] - integer bounding box
    mask: np.ndarray         - binary mask (H, W), dtype bool
    confidence: float        - detection confidence score
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
except Exception:
    torch = None  # type: ignore

try:
    import pycocotools.mask as mask_util
except ImportError:
    mask_util = None  # type: ignore


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_yolo(
    device: Optional[str] = None,
    confidence_threshold: float = 0.5,
    model_name: str = "yolo11x-seg",
) -> Dict[str, Any]:
    """
    Load a YOLOv11 instance-segmentation model.

    Args:
        device: Device spec (e.g., "cuda", "cuda:0", "cpu"). None = auto.
        confidence_threshold: Minimum confidence for detections.
        model_name: Ultralytics model name. Default is the largest YOLOv11-seg.

    Returns:
        Dict with model instance and configuration.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics package is required. Install via: pip install ultralytics>=8.3"
        ) from e

    model = YOLO(model_name)

    # Move to device if specified
    target_device = device
    if target_device is None:
        if torch is not None and torch.cuda.is_available():
            target_device = "cuda"
        else:
            target_device = "cpu"

    return {
        "model": model,
        "device": target_device,
        "confidence_threshold": confidence_threshold,
        "model_name": model_name,
    }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _to_pil(img: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """Convert various image formats to PIL Image."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def _clamp_bbox_xywh(
    x: float, y: float, w: float, h: float, W: int, H: int
) -> List[int]:
    """Clamp a bbox to image bounds and return integer [x, y, w, h]."""
    x1 = float(np.clip(x, 0, max(W - 1, 0)))
    y1 = float(np.clip(y, 0, max(H - 1, 0)))
    x2 = float(np.clip(x + w, 0, W))
    y2 = float(np.clip(y + h, 0, H))

    left = int(np.floor(x1))
    top = int(np.floor(y1))
    right = int(np.ceil(x2))
    bottom = int(np.ceil(y2))

    if right <= left:
        right = min(W, left + 1)
    if bottom <= top:
        bottom = min(H, top + 1)

    return [left, top, right - left, bottom - top]


# ---------------------------------------------------------------------------
# RLE encoding / decoding (pycocotools format)
# ---------------------------------------------------------------------------

def encode_mask_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Encode a binary mask to pycocotools RLE format.

    Args:
        mask: Binary numpy array of shape (H, W), dtype bool or uint8.

    Returns:
        Dict with 'size' ([H, W]) and 'counts' (bytes→str for JSON).
    """
    if mask_util is None:
        raise ImportError("pycocotools is required for RLE encoding. pip install pycocotools")

    # pycocotools expects Fortran-order uint8 array
    mask_f = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_util.encode(mask_f)
    # Convert bytes counts to string for JSON serialization
    rle["counts"] = rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]
    rle["size"] = list(rle["size"])
    return rle


def decode_mask_rle(rle_dict: Dict[str, Any]) -> np.ndarray:
    """
    Decode a pycocotools RLE dict back to a binary mask.

    Args:
        rle_dict: Dict with 'size' and 'counts' keys.

    Returns:
        Binary numpy array of shape (H, W), dtype bool.
    """
    if mask_util is None:
        raise ImportError("pycocotools is required for RLE decoding. pip install pycocotools")

    rle = dict(rle_dict)
    # Ensure counts is bytes for pycocotools
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    mask = mask_util.decode(rle)
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def predict_all_objects(
    model_dict: Dict[str, Any],
    images: Sequence[Union[str, Image.Image, np.ndarray]],
    batch_size: int = 8,
    topn: int = 50,
) -> List[List[Dict[str, Any]]]:
    """
    Run YOLO instance segmentation on images. Tag-free: returns ALL detected objects.

    Args:
        model_dict: Dict returned by load_yolo().
        images: List of image paths, PIL Images, or numpy arrays.
        batch_size: Batch size for inference.
        topn: Maximum number of detections per image (sorted by confidence).

    Returns:
        List (one per image) of lists of detection dicts:
            {
                "class_name": str,
                "bbox_xywh": [x, y, w, h],
                "mask": np.ndarray(H, W, bool),
                "confidence": float,
            }
    """
    model = model_dict["model"]
    device = model_dict["device"]
    conf_threshold = model_dict["confidence_threshold"]

    # Convert all images to PIL for consistent handling
    pil_images = [_to_pil(img) for img in images]

    all_detections: List[List[Dict[str, Any]]] = []

    # Process in batches
    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start : start + batch_size]

        # Run YOLO inference
        results = model.predict(
            source=batch,
            conf=conf_threshold,
            device=device,
            verbose=False,
            retina_masks=True,  # High-quality masks at original resolution
        )

        for i, result in enumerate(results):
            img_w, img_h = batch[i].size
            detections: List[Dict[str, Any]] = []

            if result.boxes is None or len(result.boxes) == 0:
                all_detections.append(detections)
                continue

            # Get boxes, scores, class ids
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
            scores = result.boxes.conf.cpu().numpy()       # (N,)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

            # Get masks if available
            masks = None
            if result.masks is not None and result.masks.data is not None:
                masks = result.masks.data.cpu().numpy()  # (N, mask_h, mask_w)

            # Sort by confidence (descending) and take topn
            sorted_idx = np.argsort(-scores)
            if topn > 0:
                sorted_idx = sorted_idx[:topn]

            for j in sorted_idx:
                x1, y1, x2, y2 = boxes_xyxy[j]
                bbox_xywh = _clamp_bbox_xywh(
                    float(x1), float(y1), float(x2 - x1), float(y2 - y1),
                    img_w, img_h,
                )

                # Extract binary mask for this detection
                det_mask = None
                if masks is not None and j < len(masks):
                    raw_mask = masks[j]
                    # Resize mask to original image size if needed
                    if raw_mask.shape != (img_h, img_w):
                        from PIL import Image as _PILImage
                        mask_pil = _PILImage.fromarray(
                            (raw_mask * 255).astype(np.uint8)
                        ).resize((img_w, img_h), resample=_PILImage.NEAREST)
                        det_mask = np.array(mask_pil) > 127
                    else:
                        det_mask = raw_mask > 0.5

                # Get class name from YOLO model names dict
                class_name = result.names.get(class_ids[j], f"class_{class_ids[j]}")

                detections.append({
                    "class_name": class_name,
                    "bbox_xywh": bbox_xywh,
                    "mask": det_mask,
                    "confidence": float(scores[j]),
                })

            all_detections.append(detections)

    return all_detections


# ---------------------------------------------------------------------------
# Convenience: extract just bboxes+masks as tuples (for crops_to_json compat)
# ---------------------------------------------------------------------------

def detections_to_bbox_mask_pairs(
    detections: List[Dict[str, Any]],
) -> List[Tuple[List[int], Optional[np.ndarray]]]:
    """
    Convert detection dicts to a list of (bbox_xywh, mask) tuples.

    Args:
        detections: List of detection dicts from predict_all_objects.

    Returns:
        List of (bbox_xywh, mask_or_None) tuples.
    """
    return [(d["bbox_xywh"], d.get("mask", None)) for d in detections]
