#!/usr/bin/env python3
"""Unit tests for semanticsegments_sam3 mode detection in save_features.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.resolve()
sys.path.insert(0, str(ROOT_DIR / "src"))

from save_features import _semanticsegments_sam3_fg_only_mode


class TestSemanticSegmentsSam3Mode(unittest.TestCase):
    def test_fg_only_when_pos_neg_is_zero(self):
        with patch.dict(os.environ, {"CROP_MODE": "semanticsegments_sam3", "POSITIVE_NEGATIVE_SEGMENT": "0"}, clear=False):
            self.assertTrue(_semanticsegments_sam3_fg_only_mode())

    def test_not_fg_only_when_pos_neg_is_one(self):
        with patch.dict(os.environ, {"CROP_MODE": "semanticsegments_sam3", "POSITIVE_NEGATIVE_SEGMENT": "1"}, clear=False):
            self.assertFalse(_semanticsegments_sam3_fg_only_mode())

    def test_not_fg_only_for_other_modes(self):
        with patch.dict(os.environ, {"CROP_MODE": "sam3", "POSITIVE_NEGATIVE_SEGMENT": "0"}, clear=False):
            self.assertFalse(_semanticsegments_sam3_fg_only_mode())


if __name__ == "__main__":
    unittest.main(verbosity=2)