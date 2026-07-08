#!/usr/bin/env python3
"""Focused tests for semanticsegments_sam3 mask post-processing."""

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from preprocessing.crops_to_json import _postprocess_semantic_sam3_masks


class TestSemanticSam3Postprocess(unittest.TestCase):
    def test_splits_large_masks_and_filters_small_disconnected_masks(self):
        image_size = (100, 100)

        small_mask = np.zeros((100, 100), dtype=bool)
        small_mask[0:5, 0:5] = True

        large_mask = np.ones((100, 100), dtype=bool)

        disconnected_mask = np.zeros((100, 100), dtype=bool)
        disconnected_mask[10:25, 10:25] = True
        disconnected_mask[60:75, 60:75] = True

        valid_mask = np.zeros((100, 100), dtype=bool)
        valid_mask[30:50, 30:50] = True

        pairs = [
            ([0, 0, 5, 5], small_mask),
            ([0, 0, 100, 100], large_mask),
            ([10, 10, 65, 65], disconnected_mask),
            ([30, 30, 20, 20], valid_mask),
        ]

        filtered = _postprocess_semantic_sam3_masks(
            pairs,
            image_size,
            masks_per_image=4,
        )

        self.assertEqual(len(filtered), 4)
        self.assertTrue(all(mask.shape == (100, 100) for _, mask in filtered))
        self.assertTrue(all(bbox[2] == 32 and bbox[3] == 32 for bbox, _ in filtered))
        self.assertTrue(all(mask.sum() == 1024 for _, mask in filtered))
        self.assertTrue(all(mask.sum() >= 1 for _, mask in filtered))
        self.assertTrue(all(mask.sum() < 7500 for _, mask in filtered))

    def test_unlimited_mode_keeps_all_valid_masks(self):
        image_size = (100, 100)
        mask_one = np.zeros((100, 100), dtype=bool)
        mask_one[10:30, 10:30] = True
        mask_two = np.zeros((100, 100), dtype=bool)
        mask_two[40:60, 40:60] = True

        pairs = [([10, 10, 20, 20], mask_one), ([40, 40, 20, 20], mask_two)]
        filtered = _postprocess_semantic_sam3_masks(pairs, image_size, masks_per_image=0)

        self.assertEqual(len(filtered), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
