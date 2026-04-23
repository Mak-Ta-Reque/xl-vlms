"""
crops_to_json.py — Simplified mask-centric JSON builder.

Given a JSON mapping ``{tag: [relative_image_paths...]}`` and an image root,
produce a crops JSON with per-image segmentation masks in RLE format.

For each (tag, image):
  1. Concept masks   — text-prompted segmentation using *tag* as the prompt.
  2. Extra masks     — prompt-free automatic segmentation (no text prompt).
  3. Enforce limits  — ``MASKS_PER_IMAGE`` total, ``CONCEPT_MASKS_PER_IMAGE``
                       concept masks.  Concept masks are always preserved.

Supported detectors: ``none``, ``langsam``, ``sam3``.
Image resizing (``--image_size_width``) is applied before detection for memory.
When ``detector="none"``, the pipeline skips localization and emits random
bbox-only crops. Set ``RLE=0`` in the environment to skip mask serialization.

Usage::

    python preprocessing/crops_to_json.py \\
        --mapping_json  outputs/bird2/concept_image_mapping.json \\
        --image_root    data/train/bird \\
        --output_json   outputs/bird2/crops.json \\
        --detector      sam3 \\
        --masks_per_image 5 \\
        --concept_masks_per_image 1 \\
        --image_size_width 512

"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable, *args, **kwargs):  # type: ignore[misc]
        return iterable


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _ensure_repo_root_on_sys_path():
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _get_image_size(path: str) -> Optional[Tuple[int, int]]:
    """Return ``(width, height)`` via Pillow; ``None`` on failure."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None


def _compute_virtual_resize(
    orig_w: int, orig_h: int, ref_width: Optional[int],
) -> Tuple[int, int, float]:
    """Compute target ``(w, h, scale)`` keeping aspect ratio."""
    if ref_width is None or ref_width <= 0 or orig_w <= 0 or orig_h <= 0:
        return orig_w, orig_h, 1.0
    scale = ref_width / float(orig_w)
    new_h = max(1, int(round(orig_h * scale)))
    return int(ref_width), new_h, float(scale)


def _load_and_resize(path: str, ref_width: Optional[int]):
    """Load image, optionally resize.  Returns ``(pil_img, (w,h), scale)``."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    new_w, new_h, scale = _compute_virtual_resize(orig_w, orig_h, ref_width)
    if ref_width and (new_w, new_h) != (orig_w, orig_h):
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
        img = img.resize((new_w, new_h), resample=resample)
    return img, (new_w, new_h), scale


def _cleanup_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# RLE encoding
# ---------------------------------------------------------------------------

def _encode_mask_rle(mask):
    """RLE-encode a binary mask for JSON serialisation."""
    _ensure_repo_root_on_sys_path()
    from src.mask_utils import encode_mask_rle
    return encode_mask_rle(mask)


def _random_bbox_for_image(
    image_size: Tuple[int, int],
    rng: random.Random,
    fixed_size: Optional[int] = None,
) -> List[int]:
    """Sample a random crop bbox in ``[x, y, w, h]`` format.

    If ``fixed_size`` is provided and > 0, a square crop of that size is used
    (clamped to image bounds).
    """
    width, height = image_size
    if width <= 1 or height <= 1:
        return [0, 0, max(1, width), max(1, height)]

    if fixed_size is not None and fixed_size > 0:
        crop_width = max(1, min(width, int(fixed_size)))
        crop_height = max(1, min(height, int(fixed_size)))
    else:
        area = width * height
        target_area = rng.uniform(0.15, 0.55) * area
        aspect_ratio = rng.uniform(0.75, 1.33)

        crop_width = int(round((target_area * aspect_ratio) ** 0.5))
        crop_height = int(round((target_area / aspect_ratio) ** 0.5))
        crop_width = max(1, min(width, crop_width))
        crop_height = max(1, min(height, crop_height))

    max_x = max(0, width - crop_width)
    max_y = max(0, height - crop_height)
    x = rng.randint(0, max_x) if max_x > 0 else 0
    y = rng.randint(0, max_y) if max_y > 0 else 0
    return [int(x), int(y), int(crop_width), int(crop_height)]


# ---------------------------------------------------------------------------
# Model loaders (singletons)
# ---------------------------------------------------------------------------

_LANGSAM_MODEL = None
_SAM3_MODEL = None
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_MODEL_NAME = None
_CLIP_MODEL_DEVICE = None


def _load_langsam(device: Optional[str] = None):
    global _LANGSAM_MODEL
    if _LANGSAM_MODEL is not None:
        return _LANGSAM_MODEL
    _ensure_repo_root_on_sys_path()
    from src.langsam_utils import load_langsam
    _LANGSAM_MODEL = load_langsam(device=device)
    return _LANGSAM_MODEL


def _load_sam3(device: Optional[str] = None, confidence_threshold: float = 0.5):
    global _SAM3_MODEL
    if _SAM3_MODEL is not None:
        return _SAM3_MODEL
    _ensure_repo_root_on_sys_path()
    from src.sam3_utils import load_sam3
    _SAM3_MODEL = load_sam3(device=device, confidence_threshold=confidence_threshold)
    return _SAM3_MODEL


def _load_clip_model(model_name: str = "ViT-B/32", device: Optional[str] = None):
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_MODEL_NAME, _CLIP_MODEL_DEVICE
    _ensure_repo_root_on_sys_path()
    try:
        import torch
        import clip
    except Exception as exc:
        raise RuntimeError(f"Could not import CLIP dependencies: {exc}")

    resolved_device = device or os.environ.get("CLIP_DEVICE")
    if resolved_device is None:
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    if (
        _CLIP_MODEL is not None
        and _CLIP_PREPROCESS is not None
        and _CLIP_MODEL_NAME == model_name
        and _CLIP_MODEL_DEVICE == resolved_device
    ):
        return _CLIP_MODEL, _CLIP_PREPROCESS, resolved_device

    _CLIP_MODEL, _CLIP_PREPROCESS = clip.load(model_name, device=resolved_device, jit=False)
    _CLIP_MODEL.eval()
    _CLIP_MODEL_NAME = model_name
    _CLIP_MODEL_DEVICE = resolved_device
    return _CLIP_MODEL, _CLIP_PREPROCESS, resolved_device


def _encode_clip_images(
    images: List[Any],
    model,
    preprocess,
    device: str,
    batch_size: int = 32,
):
    import torch

    if not images:
        return torch.empty((0, 0), dtype=torch.float32)

    embeddings = []
    for start in range(0, len(images), max(1, batch_size)):
        batch = images[start : start + max(1, batch_size)]
        batch_tensor = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        embeddings.append(feats.detach().cpu())
    return torch.cat(embeddings, dim=0)


def _greedy_filter_candidates_by_clip_similarity(
    candidate_pairs: List[Tuple[List[int], Any]],
    candidate_embeddings,
    similarity_threshold: float = 0.5,
    existing_embeddings: Optional[List[List[float]]] = None,
):
    kept_pairs: List[Tuple[List[int], Any]] = []
    kept_embeddings: List[List[float]] = []

    def _to_vector(vec):
        if hasattr(vec, "detach"):
            vec = vec.detach().cpu().tolist()
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        return [float(v) for v in vec]

    def _dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    seen_embeddings: List[List[float]] = []
    if existing_embeddings:
        seen_embeddings.extend(existing_embeddings)

    for (bbox, mask), emb in zip(candidate_pairs, candidate_embeddings):
        emb_np = _to_vector(emb)
        if seen_embeddings:
            max_similarity = max(_dot(prev_emb, emb_np) for prev_emb in seen_embeddings)
            if max_similarity >= similarity_threshold:
                continue
        kept_pairs.append((bbox, mask))
        kept_embeddings.append(emb_np)
        seen_embeddings.append(emb_np)

    return kept_pairs, kept_embeddings


def _build_diverse_random_pairs_for_image(
    image,
    image_size: Tuple[int, int],
    count: int,
    candidate_count: Optional[int],
    fixed_crop_size: Optional[int],
    rng: random.Random,
    clip_model,
    clip_preprocess,
    clip_device: str,
    similarity_threshold: float = 0.5,
    candidate_batch_size: int = 32,
    max_attempts_multiplier: int = 20,
) -> List[Tuple[List[int], Any]]:
    """Build random bbox-only crops and drop near-duplicates with CLIP."""
    if count <= 0:
        count = 1
    if candidate_count is None or candidate_count <= 0:
        candidate_count = count
    candidate_count = max(candidate_count, count)

    accepted_pairs: List[Tuple[List[int], Any]] = []
    accepted_embeddings: List[List[float]] = []
    attempts = 0
    max_attempts = max(candidate_count * max_attempts_multiplier, candidate_batch_size)

    while len(accepted_pairs) < count and attempts < max_attempts:
        batch_size = min(candidate_batch_size, max_attempts - attempts)
        candidate_pairs: List[Tuple[List[int], Any]] = []
        candidate_images = []

        for _ in range(batch_size):
            bbox = _random_bbox_for_image(
                image_size,
                rng,
                fixed_size=fixed_crop_size,
            )
            x, y, w, h = bbox
            candidate_pairs.append((bbox, None))
            candidate_images.append(image.crop((x, y, x + w, y + h)))

        candidate_embeddings = _encode_clip_images(
            candidate_images,
            clip_model,
            clip_preprocess,
            clip_device,
            batch_size=min(candidate_batch_size, len(candidate_images)),
        )

        kept_pairs_batch, kept_embeddings_batch = _greedy_filter_candidates_by_clip_similarity(
            candidate_pairs,
            candidate_embeddings,
            similarity_threshold=similarity_threshold,
            existing_embeddings=accepted_embeddings,
        )

        for pair, emb in zip(kept_pairs_batch, kept_embeddings_batch):
            accepted_pairs.append(pair)
            accepted_embeddings.append(emb)
            if len(accepted_pairs) >= count:
                break

        attempts += batch_size

    return accepted_pairs


def _sliding_window_bboxes_for_image(
    image_size: Tuple[int, int],
    window_size_ratio_range: Tuple[float, float] = (0.15, 0.55),
    stride_ratio: float = 0.3,
) -> List[Tuple[int, int, int, int]]:
    """Generate sliding window crop bboxes in ``[x, y, w, h]`` format.
    
    Args:
        image_size: (width, height) of the image
        window_size_ratio_range: (min_ratio, max_ratio) of window size relative to image area
        stride_ratio: stride as fraction of window width (controls overlap)
    
    Returns:
        List of [x, y, w, h] bboxes covering the image systematically
    """
    width, height = image_size
    if width <= 1 or height <= 1:
        return [[0, 0, max(1, width), max(1, height)]]
    
    # Use midpoint of window size range for consistency
    target_area = ((window_size_ratio_range[0] + window_size_ratio_range[1]) / 2) * (width * height)
    aspect_ratio = 1.0  # Use square windows for sliding
    
    window_width = int(round((target_area * aspect_ratio) ** 0.5))
    window_height = int(round((target_area / aspect_ratio) ** 0.5))
    window_width = max(1, min(width, window_width))
    window_height = max(1, min(height, window_height))
    
    stride = max(1, int(window_width * stride_ratio))
    
    bboxes: List[List[int]] = []
    bbox_set = set()

    def _append_unique_bbox(bbox: List[int]) -> None:
        key = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        if key in bbox_set:
            return
        bbox_set.add(key)
        bboxes.append([key[0], key[1], key[2], key[3]])

    y = 0
    while y + window_height <= height:
        x = 0
        while x + window_width <= width:
            _append_unique_bbox([x, y, window_width, window_height])
            x += stride
        y += stride
    
    # Add edge crops if there's remaining space
    if len(bboxes) == 0:
        _append_unique_bbox([0, 0, max(1, width), max(1, height)])
    else:
        # Right edge
        if x + window_width > width:
            x_edge = max(0, width - window_width)
            y_edge = 0
            while y_edge + window_height <= height:
                bbox = [x_edge, y_edge, window_width, window_height]
                _append_unique_bbox(bbox)
                y_edge += stride
        # Bottom edge
        if y + window_height > height:
            y_edge = max(0, height - window_height)
            x_edge = 0
            while x_edge + window_width <= width:
                bbox = [x_edge, y_edge, window_width, window_height]
                _append_unique_bbox(bbox)
                x_edge += stride
    
    return bboxes


def _build_sliding_window_pairs_for_image(
    image,
    image_size: Tuple[int, int],
    count: int,
    clip_model,
    clip_preprocess,
    clip_device: str,
    similarity_threshold: float = 0.5,
    window_size_ratio_range: Tuple[float, float] = (0.15, 0.55),
    stride_ratio: float = 0.3,
    candidate_batch_size: int = 32,
) -> List[Tuple[List[int], Any]]:
    """Build sliding window bbox-only crops and drop near-duplicates with CLIP.
    
    Args:
        image: PIL Image
        image_size: (width, height) of image
        count: number of crops to return
        clip_model: CLIP model instance
        clip_preprocess: CLIP preprocessing function
        clip_device: device for CLIP inference
        similarity_threshold: skip crops with CLIP cosine similarity >= this value
        window_size_ratio_range: (min_ratio, max_ratio) for window size
        stride_ratio: stride as fraction of window width
        candidate_batch_size: batch size for CLIP encoding
    
    Returns:
        List of (bbox, None) tuples where bbox is [x, y, w, h]
    """
    if count <= 0:
        count = 1
    
    # Generate all sliding window bboxes
    all_bboxes = _sliding_window_bboxes_for_image(
        image_size,
        window_size_ratio_range=window_size_ratio_range,
        stride_ratio=stride_ratio,
    )
    
    if len(all_bboxes) == 0:
        return [([0, 0, image_size[0], image_size[1]], None)]
    
    # Take up to 'count' bboxes if fewer available, otherwise process in batches with CLIP filtering
    if len(all_bboxes) <= count:
        return [(bbox, None) for bbox in all_bboxes]
    
    # Build crops with CLIP filtering
    accepted_pairs: List[Tuple[List[int], Any]] = []
    accepted_embeddings: List[List[float]] = []
    
    for batch_start_idx in range(0, len(all_bboxes), candidate_batch_size):
        batch_end_idx = min(batch_start_idx + candidate_batch_size, len(all_bboxes))
        batch_bboxes = all_bboxes[batch_start_idx:batch_end_idx]
        candidate_images = []
        
        for bbox in batch_bboxes:
            x, y, w, h = bbox
            candidate_images.append(image.crop((x, y, x + w, y + h)))
        
        # Encode candidates with CLIP
        candidate_embeddings = _encode_clip_images(
            candidate_images,
            clip_model,
            clip_preprocess,
            clip_device,
            batch_size=candidate_batch_size,
        )
        
        # Filter using greedy approach
        kept_pairs_batch, kept_embeddings_batch = _greedy_filter_candidates_by_clip_similarity(
            [(bbox, None) for bbox in batch_bboxes],
            candidate_embeddings,
            similarity_threshold=similarity_threshold,
            existing_embeddings=accepted_embeddings,
        )
        
        for pair, emb in zip(kept_pairs_batch, kept_embeddings_batch):
            accepted_pairs.append(pair)
            accepted_embeddings.append(emb)
            if len(accepted_pairs) >= count:
                return accepted_pairs
    
    return accepted_pairs


# ---------------------------------------------------------------------------
# Mask area helper
# ---------------------------------------------------------------------------

def _mask_area(mask) -> int:
    if mask is None:
        return 0
    return int(mask.sum())


def _normalize_mask_polarity(mask, max_fg_ratio: float = 0.50):
    """Invert a mask if it covers more than *max_fg_ratio* of the image.

    SAM3's point-grid auto-mask generation sometimes returns masks whose
    foreground represents the **background** of the scene (e.g. sky, ground)
    rather than a distinct object.  These masks have a very high
    foreground-to-total ratio.  Inverting them turns them into the smaller,
    more meaningful complementary segment.

    The bbox is recomputed after inversion.

    Args:
        mask: Binary numpy array (H, W).
        max_fg_ratio: If the foreground ratio exceeds this value the mask
            is inverted.  Default 0.50.

    Returns:
        The (possibly inverted) mask.
    """
    import numpy as np
    if mask is None:
        return mask
    total = mask.shape[0] * mask.shape[1]
    if total == 0:
        return mask
    fg = int(mask.sum())
    if fg / total > max_fg_ratio:
        return ~mask
    return mask


# ---------------------------------------------------------------------------
# Subtract overlap helper
# ---------------------------------------------------------------------------

def _subtract_mask(base, sub):
    """Remove *sub* pixels from *base*.  Handles resolution mismatch."""
    import numpy as np
    if base is None or sub is None:
        return base
    if base.shape != sub.shape:
        from PIL import Image as _Img
        h, w = base.shape
        sm = np.array(
            _Img.fromarray(sub.astype(np.uint8) * 255).resize((w, h), _Img.NEAREST)
        ) > 127
    else:
        sm = sub
    return base & (~sm)


def _is_mask_solid(mask, max_edge_ratio: float = 0.70) -> bool:
    """Return True if *mask* is a solid filled region, not just edges/outlines.

    A mask is considered edge-like when most of its pixels disappear after
    a mild erosion — i.e. it is made of thin 1–2 px strands rather than a
    filled blob.  Such masks arise when large overlapping masks are
    subtracted from each other, leaving only the boundary residuals.

    Args:
        mask: Binary numpy array (H, W).
        max_edge_ratio: Maximum fraction of pixels that may be "edge-only"
            (removed by erosion).  Masks above this are rejected.
    """
    import numpy as np
    if mask is None:
        return False
    filled = int(mask.sum())
    if filled == 0:
        return False
    try:
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask, iterations=2)
        interior = int(eroded.sum())
    except ImportError:
        return True  # can't check, assume OK
    edge_ratio = 1.0 - (interior / filled)
    return edge_ratio <= max_edge_ratio


# ---------------------------------------------------------------------------
# Concept mask detection (text-prompted, batched)
# ---------------------------------------------------------------------------

def detect_concept_masks(
    images,
    tag: str,
    detector: str,
    model,
    batch_size: int = 2,
    concept_masks_per_image: int = 5,
) -> List[List[Tuple[List[int], Any]]]:
    """Run text-prompted detection for *tag* and return top masks per image.

    Returns:
        List parallel to *images*.  Each element is a list of at most
        *concept_masks_per_image* ``(bbox_xywh, mask_np)`` tuples (sorted
        by area, largest first).
    """
    if detector == "langsam":
        _ensure_repo_root_on_sys_path()
        from src.langsam_utils import predict_bboxes_and_masks_for_tag_batched
        pairs_per_img = predict_bboxes_and_masks_for_tag_batched(
            model, images, tag=tag, batch_size=batch_size,
            topn=max(concept_masks_per_image, 5),
        )
    elif detector == "sam3":
        _ensure_repo_root_on_sys_path()
        from src.sam3_utils import predict_bboxes_and_masks_for_tag_sam3_batched
        pairs_per_img = predict_bboxes_and_masks_for_tag_sam3_batched(
            model, images, tag=tag, batch_size=batch_size,
            topn=max(concept_masks_per_image, 5),
        )
    else:
        raise ValueError(f"Unknown detector: {detector}")

    # Keep only top-K by area
    result = []
    for pairs in pairs_per_img:
        sorted_pairs = sorted(pairs, key=lambda p: _mask_area(p[1]), reverse=True)
        result.append(sorted_pairs[:concept_masks_per_image])
    return result


# ---------------------------------------------------------------------------
# Automatic (prompt-free) mask detection
# ---------------------------------------------------------------------------

def detect_auto_masks(
    images,
    detector: str,
    model,
    topn: int = 10,
    min_mask_area: int = 100,
) -> List[List[Tuple[List[int], Any]]]:
    """Run prompt-free segmentation.

    For LangSAM: uses SAM2AutomaticMaskGenerator.
    For SAM3:    uses point-grid automatic mask generation.

    Returns:
        List parallel to *images*, each a list of ``(bbox_xywh, mask_np)`` tuples.
    """
    if detector == "none":
        return [[] for _ in images]
    if detector == "langsam":
        _ensure_repo_root_on_sys_path()
        from src.langsam_utils import predict_all_masks_langsam
        return predict_all_masks_langsam(model, images, topn=topn, min_mask_area=min_mask_area)
    elif detector == "sam3":
        _ensure_repo_root_on_sys_path()
        from src.sam3_utils import predict_auto_masks_sam3
        return predict_auto_masks_sam3(
            model, images, topn=topn, min_mask_area=min_mask_area,
        )
    else:
        raise ValueError(f"Unknown detector: {detector}")


def _build_random_pairs_for_image(
    image_size: Tuple[int, int],
    count: int,
    rng: random.Random,
) -> List[Tuple[List[int], Any]]:
    """Build random bbox-only crops for detector-less mode."""
    if count <= 0:
        count = 1
    return [(_random_bbox_for_image(image_size, rng), None) for _ in range(count)]


# ---------------------------------------------------------------------------
# Combine concept + auto masks for one image
# ---------------------------------------------------------------------------

def _build_masks_for_image(
    concept_pairs: List[Tuple[List[int], Any]],
    auto_pairs: List[Tuple[List[int], Any]],
    masks_per_image: int,
    concept_masks_per_image: int,
    min_mask_area: int = 100,
) -> Tuple[List[Tuple[List[int], Any]], int]:
    """Merge concept + auto masks, subtract concept from auto, enforce limits.

    When text-prompted concept detection returns nothing (common for
    abstract concepts like colours, textures, or "outdoor"), the largest
    auto mask is **promoted** to concept so that every image always has
    at least one ``is_concept=True`` mask (when ``concept_masks_per_image > 0``).

    Returns:
        ``(final_pairs, n_concept)`` where the first *n_concept* entries
        are concept masks (``is_concept=True``).
    """
    import numpy as np

    # --- Normalise polarity of auto masks ---------------------------------
    # SAM3 point-grid can produce masks where foreground = background of the
    # scene (fg_ratio > 50%).  Invert those so the segment represents the
    # smaller, more distinctive region.  Recompute bbox after inversion.
    normalised_auto: List[Tuple[List[int], Any]] = []
    for bb, m in auto_pairs:
        if m is None:
            normalised_auto.append((bb, m))
            continue
        m_norm = _normalize_mask_polarity(m)
        if m_norm is not m:
            # Mask was inverted — recompute bbox from the new foreground
            ys, xs = np.where(m_norm)
            if len(ys) == 0:
                continue  # inversion produced empty mask
            bb = [int(xs.min()), int(ys.min()),
                  int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        normalised_auto.append((bb, m_norm))
    auto_pairs = normalised_auto

    # --- concept masks (always first) ---
    kept_concept: List[Tuple[List[int], Any]] = concept_pairs[:concept_masks_per_image]
    n_concept = len(kept_concept)

    # --- Fallback: promote largest auto mask(s) when concept detection missed ---
    if n_concept == 0 and concept_masks_per_image > 0 and len(auto_pairs) > 0:
        # Sort auto masks by area (largest first) and promote top-K
        sorted_auto = sorted(
            [(bb, m) for bb, m in auto_pairs if m is not None and _mask_area(m) >= min_mask_area],
            key=lambda p: _mask_area(p[1]),
            reverse=True,
        )
        promote_count = min(concept_masks_per_image, len(sorted_auto))
        kept_concept = sorted_auto[:promote_count]
        n_concept = len(kept_concept)
        # Remove promoted masks from auto_pairs so they aren't duplicated
        promoted_ids = set(id(m) for _, m in kept_concept)
        auto_pairs = [(bb, m) for bb, m in auto_pairs if id(m) not in promoted_ids]

    # Union of concept masks for subtraction
    concept_union = None
    for _, cm in kept_concept:
        if cm is not None:
            if concept_union is None:
                concept_union = cm.copy()
            else:
                if concept_union.shape != cm.shape:
                    from PIL import Image as _Img
                    h, w = concept_union.shape
                    cm_r = np.array(
                        _Img.fromarray(cm.astype(np.uint8) * 255).resize((w, h), _Img.NEAREST)
                    ) > 127
                    concept_union = concept_union | cm_r
                else:
                    concept_union = concept_union | cm

    # --- auto masks (subtract ALL previously placed masks so they are non-overlapping) ---
    # Start the running union from concept masks; each accepted auto mask is added too.
    placed_union = concept_union.copy() if concept_union is not None else None
    remaining_slots = max(0, masks_per_image - n_concept) if masks_per_image > 0 else len(auto_pairs)
    kept_auto: List[Tuple[List[int], Any]] = []
    for bbox, mask in auto_pairs:
        if len(kept_auto) >= remaining_slots:
            break
        if mask is None:
            continue
        if placed_union is not None:
            trimmed = _subtract_mask(mask, placed_union)
            if _mask_area(trimmed) < min_mask_area:
                continue
            # Reject masks that became edge-like after subtraction
            if not _is_mask_solid(trimmed):
                continue
            # Update the running union with the trimmed (non-overlapping) mask
            if placed_union.shape != trimmed.shape:
                from PIL import Image as _Img
                h, w = placed_union.shape
                tr = np.array(
                    _Img.fromarray(trimmed.astype(np.uint8) * 255).resize((w, h), _Img.NEAREST)
                ) > 127
                placed_union = placed_union | tr
            else:
                placed_union = placed_union | trimmed
            kept_auto.append((bbox, trimmed))
        else:
            if _mask_area(mask) >= min_mask_area:
                # First mask with no prior union — start one
                placed_union = mask.copy()
                kept_auto.append((bbox, mask))

    return kept_concept + kept_auto, n_concept


# ---------------------------------------------------------------------------
# JSON output helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _record_for_image(
    tag_bucket: Dict[str, dict],
    rel_path: str,
    image_size: Tuple[int, int],
    patch_size: int,
    pairs: List[Tuple[List[int], Any]],
    n_concept: int,
    store_rle: bool = True,
):
    """Serialize one image's masks into the tag bucket."""
    def _normalize_bbox_to_patch_size(
        bbox: List[int],
        image_wh: Tuple[int, int],
        target_patch: int,
    ) -> List[int]:
        """Center a bbox to a fixed square patch size, clamped to image bounds."""
        iw, ih = int(image_wh[0]), int(image_wh[1])
        if iw <= 0 or ih <= 0:
            return [0, 0, 1, 1]

        x, y, bw, bh = [int(v) for v in bbox]
        if bw <= 0 or bh <= 0:
            bw, bh = 1, 1

        cx = x + (bw / 2.0)
        cy = y + (bh / 2.0)

        side = int(target_patch) if int(target_patch) > 0 else max(1, min(iw, ih))
        side_w = max(1, min(iw, side))
        side_h = max(1, min(ih, side))

        nx = int(round(cx - side_w / 2.0))
        ny = int(round(cy - side_h / 2.0))
        nx = max(0, min(nx, iw - side_w))
        ny = max(0, min(ny, ih - side_h))
        return [int(nx), int(ny), int(side_w), int(side_h)]

    w, h = image_size
    masks_list = []
    for i, (bbox, mask) in enumerate(pairs):
        normalized_bbox = _normalize_bbox_to_patch_size(
            [int(v) for v in bbox],
            (w, h),
            patch_size,
        )
        mask_entry = {
            "bbox": normalized_bbox,
            "is_concept": i < n_concept,
        }
        if store_rle and mask is not None:
            mask_entry["rle"] = _encode_mask_rle(mask)
        masks_list.append(mask_entry)

    tag_bucket[rel_path] = {
        "meta": {
            "image_size": [w, h],
            "patch_size": patch_size,
            "virtual_w": w,
            "virtual_h": h,
        },
        "masks_rle": masks_list,
    }
    if n_concept > 0:
        tag_bucket[rel_path]["meta"]["concept_mask_index"] = 0


def _save_crop_debug_overlays(
    result: Dict[str, dict],
    image_root: str,
    debug_dir: str,
    ref_width: Optional[int] = None,
) -> None:
    """Save one debug overlay per crop entry after crops.json is written."""
    _ensure_repo_root_on_sys_path()
    from PIL import Image, ImageDraw
    import numpy as np

    try:
        from src.mask_utils import decode_mask_rle
    except Exception:
        decode_mask_rle = None

    os.makedirs(debug_dir, exist_ok=True)

    for tag, file_map in result.items():
        if not isinstance(file_map, dict):
            continue

        tag_dir = os.path.join(debug_dir, str(tag))
        os.makedirs(tag_dir, exist_ok=True)

        for rel_path, record in file_map.items():
            if not isinstance(record, dict):
                continue

            meta = record.get("meta", {}) if isinstance(record.get("meta", {}), dict) else {}
            masks_rle = record.get("masks_rle", [])
            image_size = meta.get("image_size", None)

            abs_path = os.path.join(image_root, rel_path)
            if not os.path.isfile(abs_path):
                continue

            try:
                img = Image.open(abs_path).convert("RGB")
            except Exception:
                continue

            source_size = img.size
            resized_w = resized_h = None

            # Prefer the same width-based resize rule used when building crops.
            if ref_width is not None and ref_width > 0:
                try:
                    resized_w, resized_h, _ = _compute_virtual_resize(
                        int(source_size[0]), int(source_size[1]), int(ref_width),
                    )
                except Exception:
                    resized_w = resized_h = None

            # Fallback to stored metadata when no explicit ref width is available.
            if (resized_w is None or resized_h is None) and isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
                try:
                    resized_w, resized_h = int(image_size[0]), int(image_size[1])
                except Exception:
                    resized_w = resized_h = None

            if resized_w and resized_h and resized_w > 0 and resized_h > 0 and img.size != (resized_w, resized_h):
                resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
                img = img.resize((resized_w, resized_h), resample=resample)

            resized_size = img.size
            scale_x = (resized_size[0] / float(source_size[0])) if source_size and source_size[0] else 1.0
            scale_y = (resized_size[1] / float(source_size[1])) if source_size and source_size[1] else 1.0

            stem = Path(rel_path).stem
            for crop_idx, mask_entry in enumerate(masks_rle if isinstance(masks_rle, list) else []):
                if not isinstance(mask_entry, dict):
                    continue

                bbox = mask_entry.get("bbox", None)
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue

                x, y, w, h = [int(v) for v in bbox]
                x2 = x + w
                y2 = y + h
                role = "fg" if mask_entry.get("is_concept", False) else "bg"
                crop_id = f"{stem}_idx{crop_idx}_{role}_{x}_{y}_{x2}_{y2}"

                overlay = img.copy()
                draw = ImageDraw.Draw(overlay)
                line_w = max(2, min(6, int(round(min(resized_size) * 0.004))))
                draw.rectangle([x, y, x2, y2], outline=(255, 0, 0), width=line_w)
                overlay.save(os.path.join(tag_dir, f"{crop_id}_crop_overlay.jpg"), quality=90)

                try:
                    with open(os.path.join(tag_dir, f"{crop_id}_crop_overlay.txt"), "w") as f:
                        f.write(f"source_size={source_size}\n")
                        f.write(f"resized_size={resized_size}\n")
                        f.write(f"resize_scale_x={scale_x:.6f}\n")
                        f.write(f"resize_scale_y={scale_y:.6f}\n")
                        f.write(f"bbox={(x, y, x2, y2)}\n")
                        f.write(f"is_concept={mask_entry.get('is_concept', False)}\n")
                        f.write(f"has_rle={bool(mask_entry.get('rle'))}\n")
                except Exception:
                    pass

                rle = mask_entry.get("rle", None)
                if rle is None or decode_mask_rle is None:
                    continue

                try:
                    mask_np = decode_mask_rle(rle)
                    if mask_np.shape != (resized_size[1], resized_size[0]):
                        mask_pil = Image.fromarray(mask_np.astype("uint8") * 255, mode="L")
                        mask_pil = mask_pil.resize(resized_size, resample=Image.NEAREST)
                        mask_np = np.array(mask_pil) > 127
                    Image.fromarray(mask_np.astype("uint8") * 255, mode="L").save(
                        os.path.join(tag_dir, f"{crop_id}_mask_binary.png")
                    )
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Load mapping JSON
# ---------------------------------------------------------------------------

def load_mapping(path: str) -> Dict[str, List[str]]:
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if isinstance(v, list)}


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_mapping(
    mapping_json: str,
    image_root: str,
    output_json: str,
    *,
    detector: str = "sam3",
    batch_size: int = 2,
    masks_per_image: int = 5,
    concept_masks_per_image: int = 1,
    min_images_per_tag: int = 5,
    max_images_per_tag: int = 0,
    image_size_width: Optional[int] = 512,
    patch_size: int = 200,
    device: Optional[str] = None,
    confidence_threshold: float = 0.5,
    seed: Optional[int] = None,
    verbose: bool = False,
    show_progress: bool = True,
    positive_negative_segment: bool = False,
):
    """Process a tag->images mapping and write a crops JSON with RLE masks.

    For each tag and image:
      1. Produce ``concept_masks_per_image`` concept masks (text-prompted
         with the tag string).
      2. Fill remaining slots (up to ``masks_per_image``) with prompt-free
         automatic masks.
      3. Subtract concept mask area from auto masks so they don't overlap.

    When ``positive_negative_segment=True``:
      - All detected concept instances are unioned into a single FG mask.
      - An inverted copy becomes the "background" mask.
      - Exactly 2 masks per image: [concept (fg), inverted (bg)].
      - ``masks_per_image`` is forced to 2.
    """
    if seed is not None:
        random.seed(seed)
    detector_none = detector == "none"
    store_rle = os.environ.get("RLE", "1") != "0"
    clip_similarity_threshold = float(os.environ.get("CLIP_SIMILARITY_THRESHOLD", "0.5"))
    crop_mode = os.environ.get("CROP_MODE", "random").lower()
    sliding_window_stride_ratio = float(os.environ.get("SLIDING_WINDOW_STRIDE_RATIO", "0.3"))
    random_crop_per_image = int(os.environ.get("CROP_PER_IMAGE", "0"))

    if detector_none and crop_mode not in {"random", "sliding_window"}:
        # Detector-free mode supports CLIP-filtered random/sliding-window crops only.
        crop_mode = "random"

    # --- Override masks_per_image for binary segmentation mode ---
    # In pos/neg mode the output is always 2 masks: 1 fg (union of all
    # instances) + 1 bg (inverse).  concept_masks_per_image stays 1
    # (= "1 foreground segment"); a separate _detect_topn controls how
    # many instances we fetch from the detector so the union can merge
    # all of them.
    _detect_topn: int = concept_masks_per_image     # default: same as output
    if positive_negative_segment:
        masks_per_image = 2
        concept_masks_per_image = 1
        _detect_topn = 10   # fetch up to 10 instances for union
        print(f"[positive_negative_segment] Binary fg/bg mode: "
              f"masks_per_image forced to {masks_per_image}, "
              f"concept_masks_per_image={concept_masks_per_image} "
              f"(detect_topn={_detect_topn}, all instances will be unioned)")

    mapping = load_mapping(mapping_json)

    # --- Load model once ---
    if detector_none:
        model = None
    elif detector == "langsam":
        model = _load_langsam(device=device)
    elif detector == "sam3":
        model = _load_sam3(device=device, confidence_threshold=confidence_threshold)
    else:
        raise ValueError(f"Unknown detector: {detector}. Use 'none', 'langsam' or 'sam3'.")

    clip_model = clip_preprocess = clip_device = None
    if detector_none:
        clip_model, clip_preprocess, clip_device = _load_clip_model(device=device)

    # --- Resume from existing JSON if present ---
    result: Dict[str, dict] = {}
    if os.path.isfile(output_json):
        try:
            with open(output_json) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                result = loaded
        except Exception:
            pass

    # --- Filter valid tags ---
    valid_tags: List[Tuple[str, List[str]]] = []
    for tag, rels in mapping.items():
        if not isinstance(rels, list) or len(rels) < min_images_per_tag:
            if verbose:
                print(f"[skip] tag='{tag}': only {len(rels) if isinstance(rels, list) else 0} images (need {min_images_per_tag})")
            continue
        if max_images_per_tag > 0 and len(rels) > max_images_per_tag:
            rels = random.sample(rels, max_images_per_tag)
        valid_tags.append((tag, rels))

    total_images = sum(len(rels) for _, rels in valid_tags)
    print(f"Processing {total_images} images across {len(valid_tags)} tags "
          f"(detector={detector}, masks_per_image={masks_per_image}, concept={concept_masks_per_image})")

    pbar = tqdm(total=total_images, desc="Segmenting", unit="img",
                disable=not (show_progress and TQDM_AVAILABLE),
                file=sys.stderr) if TQDM_AVAILABLE else None

    # Tag-level progress bar (concepts segmented / remaining)
    tag_pbar = tqdm(total=len(valid_tags), desc="Tags", unit="tag",
                    disable=not (show_progress and TQDM_AVAILABLE),
                    file=sys.stderr, leave=True) if TQDM_AVAILABLE else None

    for tag_idx, (tag, rels) in enumerate(valid_tags):
        if tag_pbar:
            tag_pbar.set_postfix(current=tag[:20], remaining=len(valid_tags) - tag_idx)
        tag_bucket: Dict[str, dict] = result.setdefault(tag, {})

        # Resolve absolute paths and filter already-done
        todo: List[Tuple[str, str]] = []  # (rel, abs)
        for rel in rels:
            abs_p = os.path.join(image_root, rel)
            if not os.path.isfile(abs_p):
                continue
            # Skip only if this image was already processed AND has valid
            # concept masks (when concept detection is requested).  This
            # prevents stale entries from a broken/partial earlier run from
            # being silently kept.
            if rel in tag_bucket:
                existing = tag_bucket[rel]
                has_concept = any(
                    m.get("is_concept", False) for m in existing.get("masks_rle", [])
                )
                if has_concept or concept_masks_per_image == 0:
                    if pbar:
                        pbar.update(1)
                    continue
                # Stale entry without concept masks → re-process
                del tag_bucket[rel]
            todo.append((rel, abs_p))

        if not todo:
            continue

        # Process in small chunks to limit peak GPU/CPU memory.
        # Each chunk: load images → concept detect → auto detect → combine → free.
        _CHUNK_SIZE = max(batch_size, 2)  # process this many images at a time

        for chunk_start in range(0, len(todo), _CHUNK_SIZE):
            chunk = todo[chunk_start : chunk_start + _CHUNK_SIZE]

            # --- Load & resize images for this chunk ---
            pil_images = []
            sizes = []
            for rel, abs_p in chunk:
                img, (w, h), _ = _load_and_resize(abs_p, image_size_width)
                pil_images.append(img)
                sizes.append((w, h))

            # --- 1. Concept masks (text-prompted, batched) ---
            # In pos/neg mode _detect_topn > concept_masks_per_image so
            # all instances are returned for the union step.
            if detector_none:
                concept_per_img = [[] for _ in pil_images]
            else:
                try:
                    concept_per_img = detect_concept_masks(
                        pil_images, tag=tag, detector=detector, model=model,
                        batch_size=batch_size,
                        concept_masks_per_image=_detect_topn,
                    )
                except Exception as e:
                    print(f"Warning: concept detection failed for tag='{tag}': {e}")
                    concept_per_img = [[] for _ in pil_images]

            _cleanup_gpu()
            # --- 2. Auto masks or binary fg/bg split ---
            if detector_none:
                random_count = masks_per_image if masks_per_image > 0 else 1
                for i, (rel, _) in enumerate(chunk):
                    if crop_mode == "sliding_window":
                        pairs = _build_sliding_window_pairs_for_image(
                            pil_images[i],
                            sizes[i],
                            random_count,
                            clip_model,
                            clip_preprocess,
                            clip_device,
                            similarity_threshold=clip_similarity_threshold,
                            stride_ratio=sliding_window_stride_ratio,
                        )
                    else:
                        pairs = _build_diverse_random_pairs_for_image(
                            pil_images[i],
                            sizes[i],
                            random_count,
                            random_crop_per_image if random_crop_per_image > 0 else None,
                            patch_size,
                            random,
                            clip_model,
                            clip_preprocess,
                            clip_device,
                            similarity_threshold=clip_similarity_threshold,
                        )
                    _record_for_image(
                        tag_bucket, rel, sizes[i], patch_size, pairs, 0,
                        store_rle=store_rle,
                    )
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix(tag=tag[:12], masks=len(pairs), concept=0, mode=crop_mode)

            elif positive_negative_segment:
                # Binary mode: no auto masks needed.  For each image,
                # union all concept instances into one mask, then create
                # an inverted copy as the "background" mask.
                # Dilation/erosion via MASK_BOUNDARY_PIXELS is applied
                # later at inference time (save_features.py) so the
                # stored masks stay canonical.
                import numpy as np
                auto_per_img = [[] for _ in pil_images]  # unused

                for i, (rel, _) in enumerate(chunk):
                    cp = concept_per_img[i] if i < len(concept_per_img) else []

                    if len(cp) == 0:
                        # No concept mask detected — record empty
                        _record_for_image(
                            tag_bucket, rel, sizes[i], patch_size, [], 0,
                            store_rle=store_rle,
                        )
                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix(tag=tag[:12], masks=0, concept=0, mode="pos_neg")
                        continue

                    # --- Union all detected concept instances into one mask ---
                    union_mask = None
                    for _bbox_c, _mask_c in cp:
                        if _mask_c is None:
                            continue
                        if union_mask is None:
                            union_mask = _mask_c.copy()
                        else:
                            # Handle potential shape mismatch between instances
                            if union_mask.shape != _mask_c.shape:
                                from PIL import Image as _Img
                                h, w = union_mask.shape
                                _mask_c = np.array(
                                    _Img.fromarray(_mask_c.astype(np.uint8) * 255).resize((w, h), _Img.NEAREST)
                                ) > 127
                            union_mask = union_mask | _mask_c

                    if union_mask is None:
                        _record_for_image(
                            tag_bucket, rel, sizes[i], patch_size, [], 0,
                            store_rle=store_rle,
                        )
                        if pbar:
                            pbar.update(1)
                        continue

                    mask_fg = union_mask

                    # Compute FG bbox from union mask
                    ys_fg, xs_fg = np.where(mask_fg)
                    if len(ys_fg) == 0:
                        _record_for_image(
                            tag_bucket, rel, sizes[i], patch_size, [], 0,
                            store_rle=store_rle,
                        )
                        if pbar:
                            pbar.update(1)
                        continue
                    bbox_fg = [
                        int(xs_fg.min()), int(ys_fg.min()),
                        int(xs_fg.max() - xs_fg.min() + 1),
                        int(ys_fg.max() - ys_fg.min() + 1),
                    ]

                    # Create inverted mask (background)
                    mask_bg = ~mask_fg
                    ys_bg, xs_bg = np.where(mask_bg)
                    if len(ys_bg) == 0:
                        # Concept mask covers entire image — only fg
                        pairs = [(bbox_fg, mask_fg)]
                        n_concept = 1
                    else:
                        bbox_bg = [
                            int(xs_bg.min()), int(ys_bg.min()),
                            int(xs_bg.max() - xs_bg.min() + 1),
                            int(ys_bg.max() - ys_bg.min() + 1),
                        ]
                        # pairs: [foreground (concept), background (inverted)]
                        pairs = [(bbox_fg, mask_fg), (bbox_bg, mask_bg)]
                        n_concept = 1

                    _record_for_image(
                        tag_bucket, rel, sizes[i], patch_size, pairs, n_concept,
                        store_rle=store_rle,
                    )

                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix(tag=tag[:12], masks=len(pairs), concept=n_concept, mode="pos_neg")
            else:
                # --- Standard multi-mask mode ---
                auto_topn = max(masks_per_image * 2, 20)  # fetch extra, will be filtered
                try:
                    auto_per_img = detect_auto_masks(
                        pil_images, detector=detector, model=model,
                        topn=auto_topn, min_mask_area=100,
                    )
                except Exception as e:
                    print(f"Warning: auto segmentation failed for tag='{tag}': {e}")
                    auto_per_img = [[] for _ in pil_images]

                _cleanup_gpu()

                # --- 3. Combine, subtract, record ---
                for i, (rel, _) in enumerate(chunk):
                    cp = concept_per_img[i] if i < len(concept_per_img) else []
                    ap = auto_per_img[i] if i < len(auto_per_img) else []

                    pairs, n_concept = _build_masks_for_image(
                        cp, ap,
                        masks_per_image=masks_per_image,
                        concept_masks_per_image=concept_masks_per_image,
                    )

                    _record_for_image(
                        tag_bucket, rel, sizes[i], patch_size, pairs, n_concept,
                    )

                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix(tag=tag[:12], masks=len(pairs), concept=n_concept)

            # Free chunk images + results before next chunk
            del pil_images, concept_per_img
            try:
                del auto_per_img
            except NameError:
                pass
            _cleanup_gpu()

        # Flush after each tag
        _atomic_write_json(output_json, result)

        if tag_pbar:
            tag_pbar.update(1)

    if pbar:
        pbar.close()
    if tag_pbar:
        tag_pbar.close()

    if os.environ.get("DEBUG_SAVE_VLM_INPUTS", "0") == "1":
        debug_dir = os.path.join(os.path.dirname(output_json), "debug_crop_overlays")
        _save_crop_debug_overlays(result, image_root, debug_dir, ref_width=image_size_width)

    # Final summary
    total_tags_out = len(result)
    total_imgs_out = sum(len(v) for v in result.values() if isinstance(v, dict))
    print(f"Wrote {total_imgs_out} images across {total_tags_out} tags -> {output_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build mask-centric crops JSON from a tag->image mapping.",
    )
    parser.add_argument("--mapping_json", required=True,
                        help="JSON file: {tag: [relative_paths...]}")
    parser.add_argument("--image_root", required=True,
                        help="Root directory that relative paths are resolved against")
    parser.add_argument("--output_json", required=True,
                        help="Path to write the output crops JSON")
    parser.add_argument("--detector", type=str, default="sam3",
                        choices=["none", "langsam", "sam3"],
                        help="Segmentation model (default: sam3)")
    parser.add_argument("--masks_per_image", type=int, default=5,
                        help="Total masks per image (concept + auto)")
    parser.add_argument("--concept_masks_per_image", type=int, default=5,
                        help="Number of concept (text-prompted) masks per image")
    parser.add_argument("--min_images_per_tag", type=int, default=5,
                        help="Skip tags with fewer images than this")
    parser.add_argument("--max_images_per_tag", type=int, default=0,
                        help="Cap images per tag (0 = no cap)")
    parser.add_argument("--image_size_width", type=int, default=512,
                        help="Resize width for detection (height auto by aspect ratio)")
    parser.add_argument("--patch_size", type=int, default=200,
                        help="Patch size metadata")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for text-prompted detection")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu, cuda, or cuda:N")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bars")
    parser.add_argument("--positive_negative_segment", type=int, default=0,
                        help="1 = binary fg/bg masks (concept + inverted), "
                             "0 = multi-mask pipeline (default: 0, reads "
                             "POSITIVE_NEGATIVE_SEGMENT env var as fallback)")

    args = parser.parse_args()

    # Resolve positive_negative_segment: CLI wins, then env var
    pos_neg = args.positive_negative_segment
    if pos_neg == 0:
        pos_neg = int(os.environ.get(
            "POSITIVE_NEGATIVE_SEGMENT",
            os.environ.get("POSITIVE_NEGATVE_SEGMENT", "0"),
        ))

    process_mapping(
        mapping_json=args.mapping_json,
        image_root=args.image_root,
        output_json=args.output_json,
        detector=args.detector,
        batch_size=args.batch_size,
        masks_per_image=args.masks_per_image,
        concept_masks_per_image=args.concept_masks_per_image,
        min_images_per_tag=args.min_images_per_tag,
        max_images_per_tag=args.max_images_per_tag,
        image_size_width=args.image_size_width,
        patch_size=args.patch_size,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        seed=args.seed,
        verbose=args.verbose,
        show_progress=not getattr(args, "no_progress", False),
        positive_negative_segment=bool(pos_neg),
    )


if __name__ == "__main__":
    main()
