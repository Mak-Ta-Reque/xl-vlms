#!/usr/bin/env python3
"""Visualize image + corresponding bbox entries from a feature .pth file.

Expected keys in the .pth dictionary:
- image: list[str] image file paths
- bbox: list[list|tuple] where each bbox is [x, y, w, h]
Optional keys:
- model_predictions: list[str]
- concept: list[str]
- is_concept: list[bool]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image, ImageDraw


def _to_float_list(x: Any) -> list[float] | None:
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) == 4:
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    return None


def _safe_get(seq: Any, idx: int, default: Any = None) -> Any:
    try:
        return seq[idx]
    except Exception:
        return default


def _iter_indices(total: int, max_items: int) -> Iterable[int]:
    if max_items <= 0:
        yield from range(total)
    else:
        yield from range(min(total, max_items))


def _resize_by_width(img: Image.Image, target_width: int | None) -> Image.Image:
    if target_width is None or target_width <= 0:
        return img
    ow, oh = img.size
    if ow <= 0 or oh <= 0:
        return img
    if ow == target_width:
        return img
    scale = target_width / float(ow)
    nh = max(1, int(round(oh * scale)))
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    return img.resize((target_width, nh), resample=resample)


def visualize_pth(
    pth_path: Path,
    out_dir: Path,
    max_items: int = 0,
    image_size_width: int | None = None,
) -> tuple[int, int]:
    data = torch.load(pth_path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {pth_path}, got {type(data)}")

    images = data.get("image", [])
    bboxes = data.get("bbox", [])
    preds = data.get("model_predictions", [])
    concepts = data.get("concept", [])
    is_concepts = data.get("is_concept", [])

    if not isinstance(images, list) or not isinstance(bboxes, list):
        raise ValueError("Expected list fields: image and bbox")

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    missing = 0

    for i in _iter_indices(len(images), max_items):
        img_path = Path(str(images[i]))
        bbox = _to_float_list(_safe_get(bboxes, i))
        pred = _safe_get(preds, i, "")
        concept = _safe_get(concepts, i, "")
        is_concept = _safe_get(is_concepts, i, None)

        if not img_path.exists():
            missing += 1
            continue

        img = Image.open(img_path).convert("RGB")
        # BBoxes in feature .pth are in resized-image space, so visualize on resized image.
        img = _resize_by_width(img, image_size_width)
        draw = ImageDraw.Draw(img)

        if bbox is not None:
            x, y, w, h = bbox
            x2 = x + w
            y2 = y + h
            draw.rectangle([(x, y), (x2, y2)], outline=(255, 0, 0), width=3)

        meta = f"idx={i}"
        if concept:
            meta += f" | concept={concept}"
        if pred:
            meta += f" | pred={pred}"
        if is_concept is not None:
            meta += f" | is_concept={is_concept}"

        # Draw text banner for quick inspection.
        draw.rectangle([(0, 0), (img.width, 24)], fill=(0, 0, 0))
        draw.text((6, 4), meta, fill=(255, 255, 255))

        out_name = out_dir / f"{i:05d}_{img_path.stem}_bbox.jpg"
        img.save(out_name, quality=92)
        saved += 1

    return saved, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize image+bbox from feature .pth")
    parser.add_argument("--pth", required=True, help="Path to .pth file")
    parser.add_argument("--out_dir", required=True, help="Output directory for annotated images")
    parser.add_argument("--max_items", type=int, default=0, help="Max items to save (0=all)")
    parser.add_argument(
        "--image_size_width",
        type=int,
        default=int(os.environ.get("IMAGE_SIZE_WIDTH", "512")),
        help="Resize each image to this width before drawing bbox (default: IMAGE_SIZE_WIDTH env or 512)",
    )
    args = parser.parse_args()

    pth_path = Path(args.pth)
    out_dir = Path(args.out_dir)

    saved, missing = visualize_pth(
        pth_path,
        out_dir,
        max_items=args.max_items,
        image_size_width=args.image_size_width,
    )
    print(f"Saved {saved} annotated images to: {out_dir}")
    if missing:
        print(f"Skipped {missing} entries with missing image files")


if __name__ == "__main__":
    main()
