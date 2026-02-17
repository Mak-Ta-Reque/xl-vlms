"""Visualization helpers for concept prototype rendering.

Extracted from notebooks/explain_binary.ipynb so they can be reused in the
Streamlit interactive demo and other tools.

Dependencies: Pillow, numpy, pycocotools (for RLE mask decoding).
OpenCV is **not** required — mask overlays are pure PIL/NumPy.
"""

from __future__ import annotations

import io
import os
import base64
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ────────────────────────────── bbox helpers ──────────────────────────────


def pick_first_bbox(b) -> Optional[List[float]]:
    """Return a single [x1, y1, x2, y2] bbox if present; else ``None``."""
    if b is None:
        return None
    if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b):
        return list(map(float, b))
    if isinstance(b, (list, tuple)) and len(b) > 0 and isinstance(b[0], (list, tuple)):
        return pick_first_bbox(b[0])
    return None


def draw_bbox_on_image(
    img: Image.Image,
    bbox: Optional[Sequence[float]],
    color: str = "lime",
    width: int = 3,
    label: Optional[str] = None,
) -> Image.Image:
    """Draw a ``[x1, y1, x2, y2]`` bbox on an image copy, optionally with label."""
    if bbox is None:
        return img
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        label_y = max(0, y1 - th - 4)
        draw.rectangle([x1, label_y, x1 + tw + 6, label_y + th + 4], fill=color)
        draw.text((x1 + 3, label_y + 2), label, fill="black", font=font)
    return img_draw


# ────────────────────────────── resize ────────────────────────────────────


def resize_to_width(img: Image.Image, target_width: int) -> Image.Image:
    """Resize image to *target_width* while maintaining aspect ratio."""
    if target_width <= 0:
        return img
    w, h = img.size
    if w == target_width:
        return img
    ratio = target_width / w
    new_h = max(1, int(h * ratio))
    return img.resize((target_width, new_h), Image.Resampling.LANCZOS)


def add_rounded_corners(img: Image.Image, radius: int = 16) -> Image.Image:
    """Add rounded corners (transparent) to a PIL image."""
    img = img.convert("RGBA")
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, w, h], radius=radius, fill=255)
    img.putalpha(mask)
    return img


# ────────────────────────────── RLE mask helpers ──────────────────────────


def decode_mask_rle(rle_dict) -> Optional[np.ndarray]:
    """Decode an RLE mask dict (pycocotools format) to a boolean numpy array."""
    if rle_dict is None or isinstance(rle_dict, str):
        return None
    if not isinstance(rle_dict, dict) or "counts" not in rle_dict:
        return None
    try:
        import pycocotools.mask as mask_util

        rle = dict(rle_dict)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return mask_util.decode(rle).astype(bool)
    except Exception:
        return None


def overlay_mask_on_image(
    img: Image.Image,
    mask: Optional[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> Image.Image:
    """Overlay a boolean segmentation mask on an image with a semi-transparent colour."""
    if mask is None:
        return img
    img_np = np.array(img).copy()
    h_img, w_img = img_np.shape[:2]
    h_mask, w_mask = mask.shape[:2]

    if (h_mask, w_mask) != (h_img, w_img):
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize((w_img, h_img), Image.Resampling.NEAREST)
        mask = np.array(mask_pil) > 127

    overlay = np.array(color, dtype=np.float32)
    for c in range(3):
        img_np[:, :, c] = np.where(
            mask,
            (1 - alpha) * img_np[:, :, c] + alpha * overlay[c],
            img_np[:, :, c],
        )
    return Image.fromarray(img_np.astype(np.uint8))


# ────────────────────────────── path resolution ──────────────────────────


def resolve_proto_path(p: Union[str, Path], concept_dir: Optional[Path] = None) -> Path:
    """Resolve a prototype image path.

    Resolution order:
    1. Absolute path as-is (works on the host / conda).
    2. Relative to *concept_dir* (fallback for relocated outputs).
    3. ``DATA_ROOT_REMAP`` env-var: when set to ``old_prefix:new_prefix``
       (colon-separated), any stored path starting with *old_prefix* is
       retried under *new_prefix*.  E.g.
       ``DATA_ROOT_REMAP=/mnt/abka03/xlvlm_data:/data`` will translate
       ``/mnt/abka03/xlvlm_data/img.jpg`` → ``/data/img.jpg``.
       Multiple remaps can be separated by ``;``.
    """
    pp = Path(str(p))
    if pp.exists():
        return pp
    if concept_dir is not None:
        alt = concept_dir / pp
        if alt.exists():
            return alt

    # Try environment-variable based path remapping (useful inside Docker)
    remap = os.environ.get("DATA_ROOT_REMAP", "")
    if remap:
        s = str(pp)
        for mapping in remap.split(";"):
            mapping = mapping.strip()
            if ":" not in mapping:
                continue
            old_prefix, new_prefix = mapping.split(":", 1)
            if s.startswith(old_prefix):
                candidate = Path(new_prefix + s[len(old_prefix):])
                if candidate.exists():
                    return candidate

    return pp


# ────────────────────────────── colour palette ───────────────────────────

MASK_COLORS: List[Tuple[int, int, int]] = [
    (56, 189, 248),   # sky blue
    (251, 146, 60),   # orange
    (167, 139, 250),  # violet
    (52, 211, 153),   # emerald
    (251, 113, 133),  # rose
]

MASK_COLORS_HEX: List[str] = [
    "#38bdf8",
    "#fb923c",
    "#a78bfa",
    "#34d399",
    "#fb7185",
]


# ────────────────────────────── high-level render ────────────────────────


def render_prototype(
    concept_info: dict,
    concept_dir: Optional[Path] = None,
    target_width: int = 512,
    mask_color: Tuple[int, int, int] = (56, 189, 248),
    proto_index: int = 0,
    draw_mask: bool = True,
) -> Optional[Image.Image]:
    """Render a single prototype image with bbox + optional mask overlay.

    Returns a PIL Image or ``None`` if the prototype file cannot be found.
    """
    proto_raw = concept_info.get("image_grounding_path")
    proto_list = proto_raw if isinstance(proto_raw, (list, tuple)) else ([proto_raw] if proto_raw else [])
    proto_list = [p for p in proto_list if p]
    if not proto_list or proto_index >= len(proto_list):
        return None

    pp = resolve_proto_path(proto_list[proto_index], concept_dir)
    if not pp.exists():
        return None

    proto_img = Image.open(pp).convert("RGB")
    orig_w = proto_img.size[0]
    proto_img = resize_to_width(proto_img, target_width)

    # bbox
    bbox_raw = concept_info.get("image_grounding_bboxes")
    if bbox_raw is None:
        bbox_list: list = []
    elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) > 0 and isinstance(bbox_raw[0], (list, tuple)):
        bbox_list = list(bbox_raw)
    else:
        bbox_list = [bbox_raw]

    bbox_i = bbox_list[proto_index] if proto_index < len(bbox_list) else None
    bbox_i = pick_first_bbox(bbox_i)
    if bbox_i is not None:
        scale = target_width / orig_w if orig_w > 0 else 1.0
        bbox_i = [coord * scale for coord in bbox_i]

    # mask
    if draw_mask:
        mask_raw = concept_info.get("image_grounding_masks")
        if mask_raw is None:
            mask_list: list = []
        elif isinstance(mask_raw, (list, tuple)):
            mask_list = list(mask_raw)
        else:
            mask_list = [mask_raw]

        mask_i_raw = mask_list[proto_index] if proto_index < len(mask_list) else None
        mask_decoded = decode_mask_rle(mask_i_raw)
        if mask_decoded is not None:
            proto_img = overlay_mask_on_image(proto_img, mask_decoded, color=mask_color, alpha=0.45)

    proto_img = draw_bbox_on_image(proto_img, bbox_i, color="lime", width=3)
    return proto_img


def get_num_prototypes(concept_info: dict) -> int:
    """Return the number of prototype images available for a concept."""
    proto_raw = concept_info.get("image_grounding_path")
    proto_list = proto_raw if isinstance(proto_raw, (list, tuple)) else ([proto_raw] if proto_raw else [])
    return len([p for p in proto_list if p])


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL image to a base64-encoded data URI."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"
