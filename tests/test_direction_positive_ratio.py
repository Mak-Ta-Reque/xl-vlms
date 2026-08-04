"""Tests for the positive/negative concept-direction selection ratios.

assigned ratio_j = (#VLM-positive predictions among samples assigned to
                    direction j by argmax activation) / (#assigned samples)
weighted ratio_j = (positive activation mass on j) / (total activation mass on j)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.multimodal_grounding import (
    compute_direction_positive_ratios,
    needs_final_norm,
)


def test_all_positive_direction_scores_one():
    # 30 samples, K=2: direction 0 owns 20 samples (ALL positive) -> ratio 1.0
    acts = torch.zeros(30, 2)
    acts[:20, 0] = 1.0
    acts[20:, 1] = 1.0
    preds = [["person"] * 20 + ["No person"] * 10]
    ratios, counts, weighted = compute_direction_positive_ratios(acts, preds)
    assert counts == [20, 10]
    assert ratios[0] == 1.0
    assert ratios[1] == 0.0
    assert weighted[0] == 1.0  # all mass on dir 0 comes from positives
    assert weighted[1] == 0.0


def test_mixed_direction_ratio_is_positives_over_assigned():
    acts = torch.zeros(20, 2)
    acts[:, 0] = 1.0
    preds = [["person"] * 15 + ["No person"] * 5]
    ratios, counts, weighted = compute_direction_positive_ratios(acts, preds)
    assert counts == [20, 0]
    assert ratios[0] == 15 / 20
    assert ratios[1] is None  # no samples assigned -> None -> negative bank
    assert weighted[0] == 15 / 20  # uniform mass -> same as count ratio
    assert weighted[1] is None  # zero mass on dir 1


def test_weighted_ratio_uses_activation_mass_not_counts():
    # 3 samples on dir 0: one strong positive (mass 8), two weak negatives
    # (mass 1 each). Count ratio = 1/3; weighted ratio = 8/10.
    acts = torch.tensor([[8.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    preds = [["cat", "No cat", "No cat"]]
    ratios, counts, weighted = compute_direction_positive_ratios(acts, preds)
    assert counts == [3, 0]
    assert abs(ratios[0] - 1 / 3) < 1e-9
    assert abs(weighted[0] - 0.8) < 1e-6


def test_margin_sample_splits_mass_between_directions():
    # A borderline positive sample (0.51 vs 0.49) counts fully for dir 0 in
    # the assigned ratio but contributes ~half its mass to each direction in
    # the weighted ratio.
    acts = torch.tensor([[0.51, 0.49], [0.0, 1.0]])
    preds = [["cat", "No cat"]]
    ratios, counts, weighted = compute_direction_positive_ratios(acts, preds)
    assert counts == [1, 1]
    assert ratios[0] == 1.0 and ratios[1] == 0.0
    assert abs(weighted[0] - 1.0) < 1e-6            # only the positive loads dir 0
    assert abs(weighted[1] - 0.49 / 1.49) < 1e-6    # positive mass share on dir 1


def test_all_zero_rows_are_excluded_from_assignment():
    # 2 unexplained samples (all-zero activations) must not be argmax-defaulted
    # to direction 0.
    acts = torch.zeros(4, 2)
    acts[0, 0] = 1.0  # only sample 0 is explained (by dir 0)
    acts[1, 1] = 1.0  # sample 1 explained by dir 1
    preds = [["cat", "No cat", "No cat", "No cat"]]
    ratios, counts, weighted = compute_direction_positive_ratios(acts, preds)
    assert counts == [1, 1]  # zero-rows excluded, not counted for dir 0
    assert ratios[0] == 1.0
    assert ratios[1] == 0.0


def test_batched_prediction_lists_are_flattened():
    acts = torch.zeros(6, 2)
    acts[:3, 0] = 1.0
    acts[3:, 1] = 1.0
    # predictions stored per batch (2 batches of 3)
    preds = [["cat", "cat", "No cat"], ["No cat", "No cat", "No cat"]]
    ratios, counts, weighted = compute_direction_positive_ratios(acts, preds)
    assert counts == [3, 3]
    assert abs(ratios[0] - 2 / 3) < 1e-9
    assert ratios[1] == 0.0


def test_misaligned_predictions_return_none_ratios():
    acts = torch.zeros(10, 2)
    ratios, counts, weighted = compute_direction_positive_ratios(acts, [["x", "y"]])
    assert ratios == [None, None]
    assert counts == [0, 0]
    assert weighted == [None, None]


def test_needs_final_norm_gating():
    assert needs_final_norm("model.language_model.layers.24")
    assert not needs_final_norm("model.language_model.norm")
    assert not needs_final_norm("lm_head")
    assert not needs_final_norm(None)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASSED: {name}")
