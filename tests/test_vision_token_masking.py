import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from helpers.vision_token_masking import (VisionTokenMasker, build_full_keep,
                                          compute_token_keep_mask,
                                          find_vision_module)


# ---------------------------------------------------------------------------
# compute_token_keep_mask
# ---------------------------------------------------------------------------

def test_keep_mask_left_half():
    # 112x112 image, patch grid 8x8, merge 2 -> 4x4 merged tokens (28px each).
    # Mask covers the left half -> keep exactly the left 2 token columns.
    mask = np.zeros((112, 112), dtype=bool)
    mask[:, :56] = True
    keep = compute_token_keep_mask(mask, (1, 8, 8), merge_size=2, threshold=0.5)
    grid = keep.reshape(4, 4)
    assert grid[:, :2].all()
    assert not grid[:, 2:].any()


def test_keep_mask_empty_mask_keeps_all():
    # Fully empty mask degenerates to a no-op (keep everything) rather than
    # zeroing out the entire image.
    mask = np.zeros((112, 112), dtype=bool)
    keep = compute_token_keep_mask(mask, (1, 8, 8), merge_size=2, threshold=0.5)
    assert keep.all()


def test_keep_mask_tiny_region_keeps_best_token():
    # A blob too small to clear the threshold anywhere still keeps the
    # best-covered token(s) instead of dropping everything.
    mask = np.zeros((112, 112), dtype=bool)
    mask[2:6, 2:6] = True  # 4x4 px inside the top-left 28px token
    keep = compute_token_keep_mask(mask, (1, 8, 8), merge_size=2, threshold=0.5)
    assert keep.any()
    assert not keep.all()
    assert keep.reshape(4, 4)[0, 0]


def test_keep_mask_tensor_grid_and_temporal_tiling():
    mask = np.ones((56, 56), dtype=bool)
    keep = compute_token_keep_mask(
        mask, torch.tensor([2, 4, 4]), merge_size=2, threshold=0.5
    )
    # t=2 tiles the 2x2 merged grid twice
    assert keep.numel() == 2 * 2 * 2
    assert keep.all()


# ---------------------------------------------------------------------------
# build_full_keep
# ---------------------------------------------------------------------------

def test_build_full_keep_mixed_samples():
    grids = torch.tensor([[1, 4, 4], [1, 2, 2]])  # 4 and 1 merged tokens
    keep0 = torch.tensor([True, False, False, True])
    full = build_full_keep([keep0, None], grids, merge_size=2)
    assert full.tolist() == [True, False, False, True, True]


def test_build_full_keep_all_none_returns_none():
    grids = torch.tensor([[1, 4, 4]])
    assert build_full_keep([None], grids, merge_size=2) is None
    assert build_full_keep([torch.ones(4, dtype=torch.bool)], None) is None


def test_build_full_keep_size_mismatch_keeps_all():
    grids = torch.tensor([[1, 4, 4]])  # 4 merged tokens
    bad = torch.ones(7, dtype=torch.bool)
    full = build_full_keep([bad], grids, merge_size=2)
    assert full.numel() == 4 and full.all()


# ---------------------------------------------------------------------------
# find_vision_module / VisionTokenMasker
# ---------------------------------------------------------------------------

class _Visual(torch.nn.Module):
    def forward(self, x):
        return x


class _DummyVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Visual()
        self.lm = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.visual(x)


def test_find_vision_module_by_convention_and_name():
    model = _DummyVLM()
    assert find_vision_module(model) is model.visual
    assert find_vision_module(model, "visual") is model.visual
    assert find_vision_module(model, "nope") is None
    assert find_vision_module(torch.nn.Linear(2, 2)) is None


def test_masker_zeroes_tensor_rows():
    model = _DummyVLM()
    masker = VisionTokenMasker(model.visual)
    keep = torch.tensor([True, False, True, False])
    masker.set_keep(keep)
    out = model(torch.ones(4, 3))
    assert torch.equal(out[0], torch.ones(3))
    assert torch.equal(out[1], torch.zeros(3))
    assert torch.equal(out[3], torch.zeros(3))

    # clear() disables masking again
    masker.clear()
    out = model(torch.ones(4, 3))
    assert torch.equal(out, torch.ones(4, 3))

    # remove() detaches the hook entirely
    masker.set_keep(keep)
    masker.remove()
    out = model(torch.ones(4, 3))
    assert torch.equal(out, torch.ones(4, 3))


def test_masker_handles_pooler_output_object():
    # Mimics transformers >= 5.x where the tower returns an output object
    # whose pooler_output holds the merged vision tokens.
    class _Output:
        def __init__(self, pooled):
            self.last_hidden_state = torch.ones(16, 3)
            self.pooler_output = pooled

    class _VisualObj(torch.nn.Module):
        def forward(self, x):
            return _Output(torch.ones(4, 3))

    mod = _VisualObj()
    masker = VisionTokenMasker(mod)
    masker.set_keep(torch.tensor([False, True, False, True]))
    out = mod(None)
    assert torch.equal(out.pooler_output[0], torch.zeros(3))
    assert torch.equal(out.pooler_output[1], torch.ones(3))
    # last_hidden_state (pre-merge tokens) is untouched
    assert torch.equal(out.last_hidden_state, torch.ones(16, 3))


def test_masker_handles_tuple_output():
    class _VisualTuple(torch.nn.Module):
        def forward(self, x):
            return (torch.ones(16, 3), torch.ones(4, 3))

    mod = _VisualTuple()
    masker = VisionTokenMasker(mod)
    masker.set_keep(torch.tensor([True, True, False, False]))
    out = mod(None)
    assert torch.equal(out[0], torch.ones(16, 3))
    assert torch.equal(out[1][2], torch.zeros(3))
    assert torch.equal(out[1][0], torch.ones(3))


def test_masker_shape_mismatch_leaves_output_untouched():
    model = _DummyVLM()
    masker = VisionTokenMasker(model.visual)
    masker.set_keep(torch.tensor([True, False]))  # 2 vs 4 rows
    out = model(torch.ones(4, 3))
    assert torch.equal(out, torch.ones(4, 3))
