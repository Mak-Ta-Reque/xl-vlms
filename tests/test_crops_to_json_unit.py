#!/usr/bin/env python3
"""
Unit tests for crops_to_json.py geometry helpers and crop generators.

These tests are CPU-only and do not require any heavy dependencies (SAM3, LangSAM).
They test the pure geometry/math functions for correctness.

Run with:
    python -m unittest tests.test_crops_to_json_unit -v
    
Or:
    python tests/test_crops_to_json_unit.py
"""

import os
import sys
import unittest
import random
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from preprocessing.crops_to_json import (
    calculate_iou,
    _gen_grid_boxes,
    _gen_random_boxes,
    _xywh_to_x1y1x2y2,
    _clip_boxes_x1y1x2y2,
    _compute_virtual_resize,
    _centered_square_box_around,
    _crop_detection_to_patch,
)


class TestCalculateIoU(unittest.TestCase):
    """Test IoU calculation between bounding boxes."""
    
    def test_identical_boxes(self):
        """Identical boxes should have IoU of 1.0."""
        box = (10, 10, 50, 50)
        self.assertAlmostEqual(calculate_iou(box, box), 1.0)
    
    def test_no_overlap(self):
        """Non-overlapping boxes should have IoU of 0.0."""
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        self.assertAlmostEqual(calculate_iou(box1, box2), 0.0)
    
    def test_partial_overlap(self):
        """Test partial overlap calculation."""
        box1 = (0, 0, 20, 20)   # Area = 400
        box2 = (10, 10, 30, 30) # Area = 400
        # Intersection: (10,10) to (20,20) = 100
        # Union: 400 + 400 - 100 = 700
        expected_iou = 100 / 700
        self.assertAlmostEqual(calculate_iou(box1, box2), expected_iou, places=5)
    
    def test_one_inside_other(self):
        """Test when one box is inside another."""
        outer = (0, 0, 100, 100)  # Area = 10000
        inner = (25, 25, 75, 75)  # Area = 2500
        # Intersection = 2500 (the inner box)
        # Union = 10000 (just the outer)
        expected_iou = 2500 / 10000
        self.assertAlmostEqual(calculate_iou(outer, inner), expected_iou, places=5)
    
    def test_touching_edge(self):
        """Boxes touching at edge should have IoU of 0."""
        box1 = (0, 0, 10, 10)
        box2 = (10, 0, 20, 10)  # Touches at x=10
        self.assertAlmostEqual(calculate_iou(box1, box2), 0.0)
    
    def test_zero_area_box(self):
        """Box with zero area should return 0 IoU."""
        box1 = (10, 10, 10, 10)  # Zero area (point)
        box2 = (0, 0, 20, 20)
        self.assertAlmostEqual(calculate_iou(box1, box2), 0.0)


class TestGenGridBoxes(unittest.TestCase):
    """Test non-overlapping grid box generation."""
    
    def test_exact_fit(self):
        """Grid should fit exactly when dimensions are multiples of patch_size."""
        boxes = _gen_grid_boxes(200, 200, 100)
        self.assertEqual(len(boxes), 4)  # 2x2 grid
        
        # Check all boxes are correct size
        for x1, y1, x2, y2 in boxes:
            self.assertEqual(x2 - x1, 100)
            self.assertEqual(y2 - y1, 100)
    
    def test_partial_fit(self):
        """Grid should not include partial patches at edges."""
        boxes = _gen_grid_boxes(250, 250, 100)
        # Only 2x2 = 4 boxes fit (not 3x3 because 250 < 300)
        self.assertEqual(len(boxes), 4)
    
    def test_image_too_small(self):
        """Should return empty list if image is smaller than patch_size."""
        boxes = _gen_grid_boxes(50, 50, 100)
        self.assertEqual(len(boxes), 0)
    
    def test_one_dimension_too_small(self):
        """Should return empty if either dimension is too small."""
        boxes = _gen_grid_boxes(200, 50, 100)
        self.assertEqual(len(boxes), 0)
    
    def test_non_overlapping(self):
        """Grid boxes should not overlap."""
        boxes = _gen_grid_boxes(400, 300, 100)
        for i, box1 in enumerate(boxes):
            for j, box2 in enumerate(boxes):
                if i != j:
                    iou = calculate_iou(box1, box2)
                    self.assertAlmostEqual(iou, 0.0, places=5)
    
    def test_within_bounds(self):
        """All boxes should be within image bounds."""
        w, h, patch = 350, 280, 100
        boxes = _gen_grid_boxes(w, h, patch)
        for x1, y1, x2, y2 in boxes:
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, w)
            self.assertLessEqual(y2, h)


class TestGenRandomBoxes(unittest.TestCase):
    """Test random box generation with overlap constraints."""
    
    def setUp(self):
        """Set seed for reproducibility."""
        random.seed(42)
    
    def test_generates_requested_count(self):
        """Should generate the requested number of boxes (when possible)."""
        boxes = _gen_random_boxes(500, 500, 100, count=5, existing=[], max_overlap_ratio=0.25)
        self.assertEqual(len(boxes), 5)
    
    def test_respects_max_overlap(self):
        """Generated boxes should respect max overlap ratio."""
        boxes = _gen_random_boxes(500, 500, 100, count=10, existing=[], max_overlap_ratio=0.25)
        for i, box1 in enumerate(boxes):
            for j, box2 in enumerate(boxes):
                if i != j:
                    iou = calculate_iou(box1, box2)
                    self.assertLessEqual(iou, 0.25 + 0.01)  # Small tolerance
    
    def test_avoids_existing_boxes(self):
        """New boxes should avoid overlap with existing boxes."""
        existing = [(100, 100, 200, 200)]
        boxes = _gen_random_boxes(500, 500, 100, count=5, existing=existing, max_overlap_ratio=0.1)
        for box in boxes:
            iou = calculate_iou(box, existing[0])
            self.assertLessEqual(iou, 0.1 + 0.01)
    
    def test_within_bounds(self):
        """All random boxes should be within image bounds."""
        w, h, patch = 300, 250, 100
        boxes = _gen_random_boxes(w, h, patch, count=10, existing=[], max_overlap_ratio=0.5)
        for x1, y1, x2, y2 in boxes:
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, w)
            self.assertLessEqual(y2, h)
    
    def test_correct_size(self):
        """All boxes should be exactly patch_size x patch_size."""
        patch = 80
        boxes = _gen_random_boxes(400, 400, patch, count=5, existing=[], max_overlap_ratio=0.25)
        for x1, y1, x2, y2 in boxes:
            self.assertEqual(x2 - x1, patch)
            self.assertEqual(y2 - y1, patch)
    
    def test_image_too_small(self):
        """When image is smaller than patch, boxes extend outside (caller should check bounds).
        
        Note: The actual crops_to_json code checks w >= patch_size before calling this function.
        This function itself doesn't validate bounds - it generates boxes starting at (0,0).
        """
        boxes = _gen_random_boxes(50, 50, 100, count=5, existing=[], max_overlap_ratio=0.25)
        # Function still creates boxes (but they extend outside image bounds)
        # The caller is responsible for checking w >= patch_size && h >= patch_size
        self.assertGreaterEqual(len(boxes), 0)
        if len(boxes) > 0:
            # Box starts at (0,0) but extends to (100, 100) - outside 50x50 image
            x1, y1, x2, y2 = boxes[0]
            self.assertEqual(x1, 0)
            self.assertEqual(y1, 0)
            self.assertEqual(x2, 100)  # Extends past image boundary
            self.assertEqual(y2, 100)
    
    def test_deterministic_with_seed(self):
        """Same seed should produce same boxes."""
        random.seed(123)
        boxes1 = _gen_random_boxes(400, 400, 100, count=5, existing=[], max_overlap_ratio=0.25)
        
        random.seed(123)
        boxes2 = _gen_random_boxes(400, 400, 100, count=5, existing=[], max_overlap_ratio=0.25)
        
        self.assertEqual(boxes1, boxes2)
    
    def test_may_produce_fewer_than_requested(self):
        """When space is tight, may produce fewer boxes than requested."""
        # Small image with many boxes requested and low overlap tolerance
        boxes = _gen_random_boxes(200, 200, 100, count=100, existing=[], max_overlap_ratio=0.0)
        # At most 4 non-overlapping 100x100 boxes fit in 200x200
        self.assertLessEqual(len(boxes), 4)


class TestXYWHToX1Y1X2Y2(unittest.TestCase):
    """Test coordinate format conversion."""
    
    def test_basic_conversion(self):
        """Test basic xywh to xyxy conversion."""
        xywh = [(10, 20, 30, 40)]  # x, y, width, height
        xyxy = _xywh_to_x1y1x2y2(xywh)
        self.assertEqual(xyxy, [(10, 20, 40, 60)])  # x1, y1, x2, y2
    
    def test_multiple_boxes(self):
        """Test conversion of multiple boxes."""
        xywh = [(0, 0, 10, 10), (100, 100, 50, 50)]
        xyxy = _xywh_to_x1y1x2y2(xywh)
        self.assertEqual(xyxy, [(0, 0, 10, 10), (100, 100, 150, 150)])
    
    def test_empty_list(self):
        """Empty input should return empty output."""
        self.assertEqual(_xywh_to_x1y1x2y2([]), [])


class TestClipBoxes(unittest.TestCase):
    """Test box clipping to image bounds."""
    
    def test_no_clipping_needed(self):
        """Boxes within bounds should not change."""
        boxes = [(10, 10, 90, 90)]
        clipped = _clip_boxes_x1y1x2y2(boxes, 100, 100)
        self.assertEqual(clipped, [(10, 10, 90, 90)])
    
    def test_clips_to_bounds(self):
        """Boxes extending beyond bounds should be clipped."""
        boxes = [(-10, -10, 110, 110)]
        clipped = _clip_boxes_x1y1x2y2(boxes, 100, 100)
        self.assertEqual(clipped, [(0, 0, 100, 100)])
    
    def test_removes_invalid_boxes(self):
        """Boxes that become zero-area after clipping should be removed."""
        boxes = [(200, 200, 300, 300)]  # Completely outside 100x100 image
        clipped = _clip_boxes_x1y1x2y2(boxes, 100, 100)
        self.assertEqual(clipped, [])
    
    def test_preserves_valid_boxes(self):
        """Mix of valid and invalid boxes."""
        boxes = [
            (10, 10, 50, 50),    # Valid
            (200, 200, 300, 300),  # Invalid (outside)
            (80, 80, 120, 120),   # Partially outside -> clipped
        ]
        clipped = _clip_boxes_x1y1x2y2(boxes, 100, 100)
        self.assertEqual(len(clipped), 2)
        self.assertEqual(clipped[0], (10, 10, 50, 50))
        self.assertEqual(clipped[1], (80, 80, 100, 100))


class TestComputeVirtualResize(unittest.TestCase):
    """Test virtual resize computation."""
    
    def test_no_resize_when_none(self):
        """Should return original size when image_size_width is None."""
        w, h, scale = _compute_virtual_resize(800, 600, None)
        self.assertEqual((w, h), (800, 600))
        self.assertAlmostEqual(scale, 1.0)
    
    def test_downscale(self):
        """Test downscaling to target width."""
        w, h, scale = _compute_virtual_resize(1000, 500, 500)
        self.assertEqual(w, 500)
        self.assertEqual(h, 250)
        self.assertAlmostEqual(scale, 0.5)
    
    def test_upscale(self):
        """Test upscaling to target width."""
        w, h, scale = _compute_virtual_resize(200, 100, 400)
        self.assertEqual(w, 400)
        self.assertEqual(h, 200)
        self.assertAlmostEqual(scale, 2.0)
    
    def test_preserves_aspect_ratio(self):
        """Aspect ratio should be preserved."""
        orig_w, orig_h = 1920, 1080
        new_w, new_h, scale = _compute_virtual_resize(orig_w, orig_h, 640)
        orig_ratio = orig_w / orig_h
        new_ratio = new_w / new_h
        self.assertAlmostEqual(orig_ratio, new_ratio, places=2)
    
    def test_zero_width_returns_original(self):
        """Zero target width should return original."""
        w, h, scale = _compute_virtual_resize(800, 600, 0)
        self.assertEqual((w, h), (800, 600))


class TestCenteredSquareBoxAround(unittest.TestCase):
    """Test centered square box generation around detections."""
    
    def test_centered_on_detection(self):
        """Box should be centered on the detection."""
        # Detection centered at (50, 50)
        x1, y1, x2, y2 = _centered_square_box_around(40, 40, 60, 60, 100, 200, 200)
        # Center of result should be near (50, 50)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        self.assertAlmostEqual(cx, 50, delta=1)
        self.assertAlmostEqual(cy, 50, delta=1)
    
    def test_correct_size(self):
        """Result should be patch_size x patch_size."""
        x1, y1, x2, y2 = _centered_square_box_around(50, 50, 70, 70, 100, 200, 200)
        self.assertEqual(x2 - x1, 100)
        self.assertEqual(y2 - y1, 100)
    
    def test_clamped_to_left_edge(self):
        """Box should be clamped when detection is near left edge."""
        x1, y1, x2, y2 = _centered_square_box_around(5, 50, 15, 70, 100, 200, 200)
        self.assertEqual(x1, 0)
        self.assertEqual(x2, 100)
    
    def test_clamped_to_right_edge(self):
        """Box should be clamped when detection is near right edge."""
        x1, y1, x2, y2 = _centered_square_box_around(185, 50, 195, 70, 100, 200, 200)
        self.assertEqual(x1, 100)
        self.assertEqual(x2, 200)
    
    def test_clamped_to_top_edge(self):
        """Box should be clamped when detection is near top edge."""
        x1, y1, x2, y2 = _centered_square_box_around(50, 5, 70, 15, 100, 200, 200)
        self.assertEqual(y1, 0)
        self.assertEqual(y2, 100)
    
    def test_clamped_to_bottom_edge(self):
        """Box should be clamped when detection is near bottom edge."""
        x1, y1, x2, y2 = _centered_square_box_around(50, 185, 70, 195, 100, 200, 200)
        self.assertEqual(y1, 100)
        self.assertEqual(y2, 200)


class TestCropDetectionToPatch(unittest.TestCase):
    """Test conversion of detection bbox to centered patch."""
    
    def test_creates_patch_around_detection(self):
        """Should create a patch centered on the detection."""
        detection = (30, 30, 70, 70)  # 40x40 detection centered at (50, 50)
        patch = _crop_detection_to_patch(detection, 100, 200, 200)
        
        # Patch should be 100x100
        x1, y1, x2, y2 = patch
        self.assertEqual(x2 - x1, 100)
        self.assertEqual(y2 - y1, 100)
        
        # Patch should be within bounds
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 200)
        self.assertLessEqual(y2, 200)


class TestDeterminism(unittest.TestCase):
    """Test that operations are deterministic when seeded."""
    
    def test_full_pipeline_deterministic(self):
        """Full crop generation should be deterministic with same seed."""
        def generate_crops(seed):
            random.seed(seed)
            existing = [(50, 50, 150, 150)]
            return _gen_random_boxes(500, 500, 100, count=10, existing=existing, max_overlap_ratio=0.25)
        
        crops1 = generate_crops(999)
        crops2 = generate_crops(999)
        crops3 = generate_crops(888)  # Different seed
        
        self.assertEqual(crops1, crops2, "Same seed should produce same crops")
        self.assertNotEqual(crops1, crops3, "Different seeds should produce different crops")


if __name__ == "__main__":
    unittest.main(verbosity=2)
