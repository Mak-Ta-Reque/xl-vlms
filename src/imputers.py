"""
Noisy Linear Imputer for background masking.

Based on the ROAD framework (Rong et al.) — replaces masked-out pixels via
a sparse linear system that interpolates from nearest known neighbours,
optionally adding a small amount of noise to break regularity.

This module exposes:
  - ``NoisyLinearImputer``  — the core imputer (torch tensors)
  - ``noisy_linear_inpaint`` — convenience wrapper for PIL Image + bool mask

Dependencies: torch, numpy, scipy (sparse solver).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


# ── Neighbour offsets & weights ──────────────────────────────────────────

# 8-connected neighbourhood with distance-based weights
neighbors_weights: List[Tuple[Tuple[int, int], float]] = [
    ((-1, -1), 1 / 2),
    ((-1,  0), 1.0),
    ((-1,  1), 1 / 2),
    (( 0, -1), 1.0),
    (( 0,  1), 1.0),
    (( 1, -1), 1 / 2),
    (( 1,  0), 1.0),
    (( 1,  1), 1 / 2),
]


# ── NoisyLinearImputer ──────────────────────────────────────────────────

class NoisyLinearImputer:
    """Noisy linear imputation.

    Solves a sparse linear system to fill in unknown (masked-out) pixels
    from their known neighbours.

    Parameters
    ----------
    noise : float
        Magnitude of additive Gaussian noise (absolute).  Set to 0 for
        deterministic imputation.
    weighting : list[tuple[tuple[int,int], float]]
        Neighbour offsets and their weights.
    """

    def __init__(
        self,
        noise: float = 0.01,
        weighting: List[Tuple[Tuple[int, int], float]] | None = None,
    ):
        self.noise = noise
        self.weighting = weighting if weighting is not None else neighbors_weights

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def add_offset_to_indices(
        indices: np.ndarray,
        offset: Tuple[int, int],
        mask_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add *offset* to flat *indices*.  Returns ``(valid, new_indices)``."""
        cord1 = indices % mask_shape[1]
        cord0 = indices // mask_shape[1]
        cord0 = cord0 + offset[0]
        cord1 = cord1 + offset[1]
        invalid = (
            (cord0 < 0)
            | (cord1 < 0)
            | (cord0 >= mask_shape[0])
            | (cord1 >= mask_shape[1])
        )
        valid = ~invalid
        new_coords = indices + offset[0] * mask_shape[1] + offset[1]
        return valid, new_coords

    @staticmethod
    def setup_sparse_system(
        mask: np.ndarray,
        img: np.ndarray,
        weighting: List[Tuple[Tuple[int, int], float]],
    ) -> Tuple[lil_matrix, np.ndarray]:
        """Build the sparse equation system ``A x = b``.

        Parameters
        ----------
        mask : (H, W) float/int array — 1 = known, 0 = to-impute.
        img  : (C, H, W) float array — channel-first image.
        weighting : neighbour offsets & weights.

        Returns
        -------
        A : (N, N) sparse system matrix.
        b : (N, C) right-hand-side vectors (one per channel).
        """
        maskflt = mask.flatten()
        imgflat = img.reshape((img.shape[0], -1))

        # Indices of *unknown* pixels (maskflt == 0)
        indices = np.argwhere(maskflt == 0).flatten()
        coords_to_vidx = np.zeros(len(maskflt), dtype=int)
        coords_to_vidx[indices] = np.arange(len(indices))

        num_eq = len(indices)
        A = lil_matrix((num_eq, num_eq))
        b = np.zeros((num_eq, img.shape[0]))
        sum_neighbors = np.ones(num_eq)

        for n in weighting:
            offset, weight = n[0], n[1]
            valid, new_coords = NoisyLinearImputer.add_offset_to_indices(
                indices, offset, mask.shape,
            )

            valid_coords = new_coords[valid]
            valid_ids = np.argwhere(valid).flatten()

            # Known-pixel neighbours → move to RHS
            has_values_sel = maskflt[valid_coords] > 0.5
            has_values_coords = valid_coords[has_values_sel]
            has_values_ids = valid_ids[has_values_sel]
            b[has_values_ids, :] -= weight * imgflat[:, has_values_coords].T

            # Unknown-pixel neighbours → add to LHS
            has_no_values_sel = maskflt[valid_coords] < 0.5
            has_no_values = valid_coords[has_no_values_sel]
            variable_ids = coords_to_vidx[has_no_values]
            has_no_values_ids = valid_ids[has_no_values_sel]
            A[has_no_values_ids, variable_ids] = weight

            # Reduce weight for out-of-bounds neighbours
            sum_neighbors[np.argwhere(~valid).flatten()] -= weight

        A[np.arange(num_eq), np.arange(num_eq)] = -sum_neighbors
        return A, b

    # ── __call__ ─────────────────────────────────────────────────────────

    def __call__(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Impute unknown pixels.

        Parameters
        ----------
        img  : (C, H, W) float tensor — original image.
        mask : (H, W) float/int tensor — 1 = keep, 0 = impute.

        Returns
        -------
        (C, H, W) float tensor with imputed values.
        """
        imgflt = img.reshape(img.shape[0], -1)
        maskflt = mask.reshape(-1)
        indices_linear = np.argwhere(maskflt.numpy() == 0).flatten()

        if len(indices_linear) == 0:
            return img.clone()

        A, b = NoisyLinearImputer.setup_sparse_system(
            mask.numpy(), img.numpy(), self.weighting,
        )
        res = torch.tensor(spsolve(csc_matrix(A), b), dtype=torch.float)

        img_infill = imgflt.clone()
        img_infill[:, indices_linear] = res.t() + self.noise * torch.randn_like(res.t())
        return img_infill.reshape_as(img)

    def batched_call(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Process a batch of images sequentially."""
        in_device = img.device
        res_list = []
        for i in range(len(img)):
            res_list.append(self.__call__(img[i].cpu(), mask[i].cpu()))
        return torch.stack(res_list).to(in_device)


# ── PIL convenience wrapper ──────────────────────────────────────────────

# Module-level singleton to avoid re-creating the imputer every call.
_default_imputer: NoisyLinearImputer | None = None


def noisy_linear_inpaint(
    crop_img,  # PIL.Image.Image (RGB)
    crop_mask_np: np.ndarray,  # bool (H, W) — True = keep sharp
    noise: float = 0.01,
):
    """Convenience wrapper: PIL Image + bool mask → inpainted PIL Image.

    ``crop_mask_np`` follows the convention of ``_apply_background_masking``:
    ``True`` = foreground (keep sharp), ``False`` = background (impute).
    """
    from PIL import Image as _PILImage

    global _default_imputer
    if _default_imputer is None or _default_imputer.noise != noise:
        _default_imputer = NoisyLinearImputer(noise=noise)

    img_np = np.array(crop_img).astype(np.float32) / 255.0  # (H, W, 3)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)

    # Imputer convention: 1 = known, 0 = to-impute
    mask_t = torch.from_numpy(crop_mask_np.astype(np.float32))  # (H, W)

    result_t = _default_imputer(img_t, mask_t)
    result_np = (result_t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return _PILImage.fromarray(result_np)
