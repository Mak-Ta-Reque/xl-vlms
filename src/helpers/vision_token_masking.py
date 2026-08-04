"""Vision-token masking: pass the FULL image to the VLM and zero out the
vision-encoder output tokens that fall outside a crop bbox / segmentation
mask, instead of cropping / blurring the input pixels.

Enabled in ``save_features`` via ``VISION_TOKEN_MASKING=1``.

Designed around Qwen2.5-VL's vision tower: its merged output tokens form a
raster grid of ``(grid_h / merge_size) x (grid_w / merge_size)`` per image,
with the images of a batch concatenated along the token dimension, so a
pixel-space mask maps onto tokens by resizing it to the merged-token grid.
"""

from typing import Any, Callable, List, Optional

import numpy as np
import torch

__all__ = [
    "compute_token_keep_mask",
    "build_full_keep",
    "find_vision_module",
    "VisionTokenMasker",
]


def _grid_to_ints(grid_thw: Any) -> tuple:
    """Accept a (3,) tensor / list / tuple and return (t, grid_h, grid_w)."""
    if torch.is_tensor(grid_thw):
        grid_thw = grid_thw.flatten().tolist()
    t, gh, gw = (int(v) for v in grid_thw)
    return t, gh, gw


def compute_token_keep_mask(
    mask_np: np.ndarray,
    grid_thw: Any,
    merge_size: int = 2,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Map a pixel-space foreground mask to a per-vision-token keep mask.

    Args:
        mask_np:    Boolean (H, W) array in the space of the image handed to
                    the processor. True = keep (foreground).
        grid_thw:   (t, grid_h, grid_w) patch grid of this image as returned
                    by the processor (``image_grid_thw`` row), i.e. BEFORE
                    spatial merging.
        merge_size: Spatial merge factor of the vision tower (2 for Qwen2.5-VL).
        threshold:  Minimum fraction of a merged token's receptive patch that
                    must be foreground for the token to be kept.

    Returns:
        Bool tensor of shape (t * grid_h * grid_w / merge_size**2,) in the
        raster order of the merged tokens. Guaranteed to keep at least one
        token: if nothing clears the threshold the best-covered tokens are
        kept, and a fully-empty mask keeps everything (no-op).
    """
    from PIL import Image

    t, gh, gw = _grid_to_ints(grid_thw)
    th, tw = gh // merge_size, gw // merge_size

    mask_img = Image.fromarray((mask_np > 0).astype("uint8") * 255, mode="L")
    frac = np.asarray(mask_img.resize((tw, th), Image.BILINEAR), dtype=np.float32) / 255.0

    keep = frac >= threshold
    if not keep.any():
        peak = float(frac.max())
        if peak > 0.0:
            keep = frac >= peak
        else:
            keep = np.ones_like(keep, dtype=bool)

    keep = np.tile(keep.reshape(1, th, tw), (t, 1, 1)).reshape(-1)
    return torch.from_numpy(keep)


def build_full_keep(
    per_sample_keeps: List[Optional[torch.Tensor]],
    image_grid_thw: Optional[torch.Tensor],
    merge_size: int = 2,
    logger: Callable = None,
) -> Optional[torch.Tensor]:
    """Concatenate per-sample keep masks into one vector aligned with the
    vision tower's batched output. Samples without a mask keep all tokens."""
    if image_grid_thw is None or all(k is None for k in per_sample_keeps):
        return None
    if len(per_sample_keeps) != image_grid_thw.shape[0]:
        if logger:
            logger.warning(
                f"[vision-token-mask] {len(per_sample_keeps)} keep masks vs "
                f"{image_grid_thw.shape[0]} images — skipping masking for this batch"
            )
        return None

    parts = []
    for keep, row in zip(per_sample_keeps, image_grid_thw):
        n_tokens = int(torch.prod(row).item()) // (merge_size**2)
        if keep is None:
            keep = torch.ones(n_tokens, dtype=torch.bool)
        elif keep.numel() != n_tokens:
            if logger:
                logger.warning(
                    f"[vision-token-mask] keep mask has {keep.numel()} entries "
                    f"but image has {n_tokens} tokens — keeping all tokens"
                )
            keep = torch.ones(n_tokens, dtype=torch.bool)
        parts.append(keep)
    return torch.cat(parts)


def find_vision_module(model: Callable, module_name: str = None) -> Optional[Callable]:
    """Locate the vision tower submodule, by explicit name or by convention
    (``visual`` / ``vision_tower`` / ``vision_model``)."""
    named = dict(model.named_modules())
    if module_name:
        return named.get(module_name)
    candidates = [
        name
        for name in named
        if name and name.rsplit(".", 1)[-1] in ("visual", "vision_tower", "vision_model")
    ]
    if not candidates:
        return None
    # Outermost match (shortest path) is the whole tower, not a nested block.
    return named[min(candidates, key=len)]


class VisionTokenMasker:
    """Forward hook on the vision tower that multiplies the merged vision
    tokens by a 0/1 keep mask set per batch via :meth:`set_keep`.

    Handles the tower returning a bare ``(N, D)`` tensor (older transformers),
    a ``BaseModelOutputWithPooling`` whose ``pooler_output`` holds the merged
    tokens (transformers >= 5.x), or a tuple containing the token tensor.
    """

    def __init__(self, module: Callable, logger: Callable = None) -> None:
        self._keep: Optional[torch.Tensor] = None
        self._logger = logger
        self._warned = False
        self._handle = module.register_forward_hook(self._hook)

    def set_keep(self, keep: Optional[torch.Tensor]) -> None:
        self._keep = keep
        self._warned = False

    def clear(self) -> None:
        self._keep = None

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _warn_mismatch(self, shape: Any) -> None:
        if self._warned:
            return
        self._warned = True
        if self._logger:
            self._logger.warning(
                f"[vision-token-mask] keep mask ({self._keep.numel()} tokens) does "
                f"not match vision output {tuple(shape)} — masking skipped"
            )

    def _mask_rows(self, tokens: torch.Tensor) -> torch.Tensor:
        keep = self._keep.to(device=tokens.device, dtype=tokens.dtype)
        return tokens * keep.unsqueeze(-1)

    def _hook(self, module: Callable, args: Any, output: Any) -> Any:
        keep = self._keep
        if keep is None:
            return output
        n = keep.numel()

        if torch.is_tensor(output):
            if output.ndim == 2 and output.shape[0] == n:
                return self._mask_rows(output)
            self._warn_mismatch(output.shape)
            return output

        pooled = getattr(output, "pooler_output", None)
        if torch.is_tensor(pooled):
            if pooled.ndim == 2 and pooled.shape[0] == n:
                output.pooler_output = self._mask_rows(pooled)
            else:
                self._warn_mismatch(pooled.shape)
            return output

        if isinstance(output, (tuple, list)):
            out = list(output)
            for i, el in enumerate(out):
                if torch.is_tensor(el) and el.ndim == 2 and el.shape[0] == n:
                    out[i] = self._mask_rows(el)
                    return type(output)(out)
            self._warn_mismatch(getattr(out[0], "shape", ()))
        return output
