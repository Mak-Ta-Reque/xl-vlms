"""
Mask encoding/decoding utilities (RLE format via pycocotools).

Extracted from yolo_utils.py to be detector-agnostic.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:
    from pycocotools import mask as mask_util  # type: ignore
except ImportError:
    mask_util = None


def encode_mask_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Encode a binary mask to pycocotools RLE format.

    Args:
        mask: Binary numpy array of shape (H, W), dtype bool or uint8.

    Returns:
        Dict with 'size' ([H, W]) and 'counts' (str, JSON-safe).
    """
    if mask_util is None:
        raise ImportError("pycocotools is required for RLE encoding. pip install pycocotools")

    mask_f = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_util.encode(mask_f)
    rle["counts"] = rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]
    rle["size"] = list(rle["size"])
    return rle


def decode_mask_rle(rle_dict: Dict[str, Any]) -> np.ndarray:
    """Decode a pycocotools RLE dict back to a binary mask.

    Args:
        rle_dict: Dict with 'size' and 'counts' keys.

    Returns:
        Binary numpy array of shape (H, W), dtype bool.
    """
    if mask_util is None:
        raise ImportError("pycocotools is required for RLE decoding. pip install pycocotools")

    rle_copy = dict(rle_dict)
    if isinstance(rle_copy.get("counts"), str):
        rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
    return mask_util.decode(rle_copy).astype(bool)
