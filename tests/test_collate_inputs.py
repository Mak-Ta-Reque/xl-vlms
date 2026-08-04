import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from save_features import _collate_inputs

PAD = 99


def _sample(seq_len, n_patches, token_offset=0):
    return {
        "input_ids": torch.arange(seq_len).unsqueeze(0) + token_offset,
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        "mm_token_type_ids": torch.full((1, seq_len), token_offset, dtype=torch.long),
        "pixel_values": torch.ones(n_patches, 4) * (token_offset + 1),
        "image_grid_thw": torch.tensor([[1, 2, n_patches // 2]]),
    }


def test_single_sample_passthrough():
    s = _sample(5, 8)
    assert _collate_inputs([s], pad_id=PAD) is s


def test_heterogeneous_lengths_pad_token_aligned_tensors():
    a, b = _sample(5, 8, token_offset=0), _sample(3, 4, token_offset=7)
    out = _collate_inputs([a, b], pad_id=PAD)

    # input_ids left-padded with pad_id to the max length
    assert out["input_ids"].shape == (2, 5)
    assert out["input_ids"][1, :2].tolist() == [PAD, PAD]
    assert out["input_ids"][1, 2:].tolist() == [7, 8, 9]

    # attention_mask left-padded with 0
    assert out["attention_mask"][1].tolist() == [0, 0, 1, 1, 1]

    # mm_token_type_ids must be padded IN LOCKSTEP with input_ids — this is
    # what get_rope_index indexes with attention_mask (the pre-fix code left
    # it as an unpadded list and Qwen generation crashed with an IndexError).
    assert isinstance(out["mm_token_type_ids"], torch.Tensor)
    assert out["mm_token_type_ids"].shape == (2, 5)
    mask = out["attention_mask"][1].bool()
    assert out["mm_token_type_ids"][1][mask].tolist() == [7, 7, 7]

    # non-token tensors are concatenated along dim 0
    assert out["pixel_values"].shape == (12, 4)
    assert out["image_grid_thw"].shape == (2, 3)


def test_homogeneous_lengths_unchanged_behavior():
    a, b = _sample(4, 6), _sample(4, 6, token_offset=1)
    out = _collate_inputs([a, b], pad_id=PAD)
    assert out["input_ids"].shape == (2, 4)
    assert (out["attention_mask"] == 1).all()
    assert out["mm_token_type_ids"].shape == (2, 4)
    assert out["pixel_values"].shape == (12, 4)


def test_uncollatable_values_fall_back_to_list():
    a, b = _sample(4, 6), _sample(4, 6)
    a["weird"] = torch.ones(2, 2, 2)
    b["weird"] = torch.ones(3, 5)
    out = _collate_inputs([a, b], pad_id=PAD)
    assert isinstance(out["weird"], list) and len(out["weird"]) == 2
