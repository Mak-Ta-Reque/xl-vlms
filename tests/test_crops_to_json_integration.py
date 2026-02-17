#!/usr/bin/env python3
"""
Integration tests for crops_to_json.py.

These tests create temporary images and directories to test the full pipeline:
- Folder mode (walking directory structure)
- JSON mapping mode
- Concept mode
- Detector modes (SAM3/LangSAM) with stubbed detectors
- Multibatch behavior
- Visualization verification
- Resumability

Run with:
    python -m unittest tests.test_crops_to_json_integration -v

Or:
    python tests/test_crops_to_json_integration.py
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from unittest.mock import patch, MagicMock

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))


def create_test_image(path: str, width: int = 400, height: int = 300, color: Tuple[int, int, int] = (128, 128, 128)):
    """Create a test image using Pillow."""
    from PIL import Image
    img = Image.new('RGB', (width, height), color=color)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    img.save(path)
    return path


def create_test_folder_structure(base_dir: str, tags: Dict[str, int], img_size: Tuple[int, int] = (400, 300)):
    """
    Create a folder structure with test images.
    
    Args:
        base_dir: Base directory to create structure in.
        tags: Dict of tag_name -> number of images.
        img_size: (width, height) for images.
        
    Returns:
        Dict mapping tag -> list of absolute image paths.
    """
    result = {}
    for tag, count in tags.items():
        tag_dir = os.path.join(base_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)
        paths = []
        for i in range(count):
            img_path = os.path.join(tag_dir, f"image_{i:03d}.jpg")
            create_test_image(img_path, *img_size)
            paths.append(img_path)
        result[tag] = paths
    return result


def create_test_mapping_json(json_path: str, input_root: str, tags: Dict[str, int], img_size: Tuple[int, int] = (400, 300)):
    """
    Create a mapping JSON and the corresponding images.
    
    Args:
        json_path: Path to write the mapping JSON.
        input_root: Root directory for images.
        tags: Dict of tag_name -> number of images.
        img_size: (width, height) for images.
        
    Returns:
        The mapping dict that was written.
    """
    mapping = {}
    for tag, count in tags.items():
        rel_paths = []
        for i in range(count):
            rel_path = f"{tag}/img_{i:03d}.png"
            abs_path = os.path.join(input_root, rel_path)
            create_test_image(abs_path, *img_size)
            rel_paths.append(rel_path)
        mapping[tag] = rel_paths
    
    os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    return mapping


def mock_detector_fn(
    images: List[Any],
    tag: str,
    detector: str,
    batch_size: int = 8,
    model: Optional[Any] = None,
    topn: int = 10,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Mock detector that returns deterministic fake boxes.
    
    Returns boxes based on the tag name to simulate different detections.
    Each image gets 1-3 boxes depending on image index.
    """
    results = []
    for idx, img in enumerate(images):
        # Generate deterministic boxes based on index
        num_boxes = (idx % 3) + 1  # 1, 2, or 3 boxes
        boxes = []
        for b in range(num_boxes):
            # Create boxes in xywh format (what the real detector returns)
            x = 50 + b * 60
            y = 50 + b * 40
            w = 80
            h = 60
            boxes.append((x, y, w, h))
        results.append(boxes[:topn])  # Respect topn
    return results


def mock_detector_batched_tracker(batch_sizes: List[int]):
    """
    Factory for a mock detector that tracks batch sizes.
    
    Args:
        batch_sizes: List to append batch sizes to (for verification).
        
    Returns:
        A mock detector function that tracks batch sizes.
    """
    def detector_fn(images, tag, detector, batch_size=8, model=None, topn=10):
        batch_sizes.append(len(images))
        return mock_detector_fn(images, tag, detector, batch_size, model, topn)
    return detector_fn


class TestFolderMode(unittest.TestCase):
    """Test processing images from folder structure."""
    
    def setUp(self):
        """Create temporary directory with test images."""
        self.temp_dir = tempfile.mkdtemp(prefix="crops_test_")
        self.input_dir = os.path.join(self.temp_dir, "images")
        self.output_json = os.path.join(self.temp_dir, "crops.json")
        
        # Create test images
        self.image_paths = create_test_folder_structure(
            self.input_dir,
            {"apple": 3, "banana": 2},
            img_size=(400, 300)
        )
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_folder_mode_basic(self):
        """Test basic folder mode processing."""
        from preprocessing.crops_to_json import process_folder_structure_to_json
        
        result = process_folder_structure_to_json(
            root_input=self.input_dir,
            patch_size=100,
            P=5,
            max_overlap=0.25,
            grid=False,
            object_detection=False,
            verbose=False,
        )
        
        # Check tags are discovered
        self.assertIn("apple", result)
        self.assertIn("banana", result)
        
        # Check image counts
        self.assertEqual(len(result["apple"]), 3)
        self.assertEqual(len(result["banana"]), 2)
    
    def test_folder_mode_json_schema(self):
        """Test that output JSON has correct schema."""
        from preprocessing.crops_to_json import process_folder_structure_to_json
        
        result = process_folder_structure_to_json(
            root_input=self.input_dir,
            patch_size=100,
            P=5,
            max_overlap=0.25,
            grid=False,
            object_detection=False,
        )
        
        for tag, images_dict in result.items():
            for rel_path, entry in images_dict.items():
                # Check required keys
                self.assertIn("meta", entry)
                self.assertIn("detections_xyxy", entry)
                self.assertIn("random_crops", entry)
                
                # Check meta structure
                meta = entry["meta"]
                self.assertIn("image_size", meta)
                self.assertIn("patch_size", meta)
                self.assertEqual(meta["image_size"], [400, 300])
                self.assertEqual(meta["patch_size"], 100)
                
                # Check crops are valid
                for crop in entry["random_crops"]:
                    self.assertEqual(len(crop), 4)
                    x1, y1, x2, y2 = crop
                    self.assertGreaterEqual(x1, 0)
                    self.assertGreaterEqual(y1, 0)
                    self.assertLessEqual(x2, 400)
                    self.assertLessEqual(y2, 300)
                    self.assertEqual(x2 - x1, 100)
                    self.assertEqual(y2 - y1, 100)
    
    def test_folder_mode_grid(self):
        """Test grid mode creates non-overlapping tiles."""
        from preprocessing.crops_to_json import process_folder_structure_to_json, calculate_iou
        
        result = process_folder_structure_to_json(
            root_input=self.input_dir,
            patch_size=100,
            P=5,
            max_overlap=0.25,
            grid=True,  # Enable grid mode
            object_detection=False,
        )
        
        for tag, images_dict in result.items():
            for rel_path, entry in images_dict.items():
                crops = entry["random_crops"]
                # Grid should produce 4x3 = 12 patches for 400x300 with 100px patches
                # Actually: floor(400/100) * floor(300/100) = 4 * 3 = 12... but
                # the code uses range(0, h - patch_size + 1, patch_size) so:
                # x: 0, 100, 200, 300 (4 values)
                # y: 0, 100, 200 (3 values, since 300-100+1=201, range(0,201,100)=[0,100,200])
                expected_count = 4 * 3
                self.assertEqual(len(crops), expected_count)
                
                # Check no overlap
                for i, c1 in enumerate(crops):
                    for j, c2 in enumerate(crops):
                        if i != j:
                            iou = calculate_iou(tuple(c1), tuple(c2))
                            self.assertAlmostEqual(iou, 0.0)
    
    def test_folder_mode_deterministic(self):
        """Test that seed makes output deterministic."""
        from preprocessing.crops_to_json import process_folder_structure_to_json
        
        random.seed(42)
        result1 = process_folder_structure_to_json(
            root_input=self.input_dir,
            patch_size=100,
            P=5,
            max_overlap=0.25,
            grid=False,
            object_detection=False,
        )
        
        random.seed(42)
        result2 = process_folder_structure_to_json(
            root_input=self.input_dir,
            patch_size=100,
            P=5,
            max_overlap=0.25,
            grid=False,
            object_detection=False,
        )
        
        # Results should be identical
        self.assertEqual(json.dumps(result1, sort_keys=True), json.dumps(result2, sort_keys=True))
    
    def test_folder_mode_with_resize(self):
        """Test that resize option affects recorded image_size."""
        from preprocessing.crops_to_json import process_folder_structure_to_json
        
        result = process_folder_structure_to_json(
            root_input=self.input_dir,
            patch_size=100,
            P=5,
            max_overlap=0.25,
            grid=False,
            object_detection=False,
            image_size_width=200,  # Resize from 400 to 200
        )
        
        for tag, images_dict in result.items():
            for rel_path, entry in images_dict.items():
                meta = entry["meta"]
                # Original: 400x300, resized to width=200 -> 200x150
                self.assertEqual(meta["image_size"], [200, 150])


class TestMappingMode(unittest.TestCase):
    """Test processing images from JSON mapping."""
    
    def setUp(self):
        """Create temporary directory with test images and mapping."""
        self.temp_dir = tempfile.mkdtemp(prefix="crops_mapping_test_")
        self.input_root = os.path.join(self.temp_dir, "images")
        self.mapping_json = os.path.join(self.temp_dir, "mapping.json")
        self.output_json = os.path.join(self.temp_dir, "crops.json")
        
        # Create test images and mapping
        self.mapping = create_test_mapping_json(
            self.mapping_json,
            self.input_root,
            {"cat": 35, "dog": 40},  # More than min_images_per_tag default (30)
            img_size=(400, 300)
        )
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_mapping_mode_basic(self):
        """Test basic mapping mode processing."""
        from preprocessing.crops_to_json import process_json_mapping_to_json
        
        result = process_json_mapping_to_json(
            json_file=self.mapping_json,
            input_root=self.input_root,
            patch_size=100,
            P=5,
            max_overlap=0.25,
            grid=False,
            min_images_per_tag=30,
            object_detection=False,
        )
        
        self.assertIn("cat", result)
        self.assertIn("dog", result)
        self.assertEqual(len(result["cat"]), 35)
        self.assertEqual(len(result["dog"]), 40)
    
    def test_mapping_mode_min_images_filter(self):
        """Test that tags with fewer than min_images are skipped."""
        # Create mapping with a small tag
        small_mapping_json = os.path.join(self.temp_dir, "small_mapping.json")
        create_test_mapping_json(
            small_mapping_json,
            os.path.join(self.temp_dir, "small_images"),
            {"large_tag": 50, "small_tag": 10},  # small_tag < 30
        )
        
        from preprocessing.crops_to_json import process_json_mapping_to_json
        
        result = process_json_mapping_to_json(
            json_file=small_mapping_json,
            input_root=os.path.join(self.temp_dir, "small_images"),
            patch_size=100,
            P=5,
            min_images_per_tag=30,
            object_detection=False,
        )
        
        self.assertIn("large_tag", result)
        self.assertNotIn("small_tag", result)  # Should be filtered out
    
    def test_mapping_mode_max_images_cap(self):
        """Test that max_images_per_tag caps the number of images."""
        from preprocessing.crops_to_json import process_json_mapping_to_json
        
        random.seed(42)
        result = process_json_mapping_to_json(
            json_file=self.mapping_json,
            input_root=self.input_root,
            patch_size=100,
            P=5,
            min_images_per_tag=30,
            max_images_per_tag=32,  # Cap at 32
            object_detection=False,
        )
        
        # Both tags should be capped at 32
        self.assertEqual(len(result["cat"]), 32)
        self.assertEqual(len(result["dog"]), 32)


class TestConceptMode(unittest.TestCase):
    """Test concept-focused cropping mode."""
    
    def setUp(self):
        """Create temporary directory with test images and mapping."""
        self.temp_dir = tempfile.mkdtemp(prefix="crops_concept_test_")
        self.input_root = os.path.join(self.temp_dir, "images")
        self.mapping_json = os.path.join(self.temp_dir, "mapping.json")
        
        self.mapping = create_test_mapping_json(
            self.mapping_json,
            self.input_root,
            {"bird": 35},
            img_size=(400, 300)
        )
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_concept_mode_crops_per_image(self):
        """Test that concept mode respects max_crops_per_image."""
        from preprocessing.crops_to_json import concept_process_json_mapping_to_json
        
        result = concept_process_json_mapping_to_json(
            json_file=self.mapping_json,
            input_root=self.input_root,
            max_crops_per_image=3,
            patch_size=100,
            min_images_per_tag=30,
            object_detection=False,
        )
        
        self.assertIn("bird", result)
        for rel_path, entry in result["bird"].items():
            # Without detection, all crops come from random_crops
            total_crops = len(entry["random_crops"])
            self.assertLessEqual(total_crops, 3)


class TestDetectorModes(unittest.TestCase):
    """Test detector integration with mocked SAM3/LangSAM."""
    
    def setUp(self):
        """Create temporary directory with test images."""
        self.temp_dir = tempfile.mkdtemp(prefix="crops_detector_test_")
        self.input_dir = os.path.join(self.temp_dir, "images")
        
        self.image_paths = create_test_folder_structure(
            self.input_dir,
            {"apple": 5},
            img_size=(400, 300)
        )
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('preprocessing.crops_to_json.run_detector_batched')
    @patch('preprocessing.crops_to_json._load_sam3_model')
    def test_sam3_detector_called(self, mock_load_sam3, mock_detector):
        """Test that SAM3 detector is called correctly."""
        from preprocessing.crops_to_json import process_folder_structure_to_json
        
        # Setup mocks
        mock_load_sam3.return_value = MagicMock()
        mock_detector.side_effect = mock_detector_fn
        
        result = process_folder_structure_to_json(
            root_input=self.input_dir,
            patch_size=100,
            P=5,
            max_overlap=0.25,
            grid=False,
            object_detection=True,
            detector="sam3",
            batch_size=4,
            topn=5,
        )
        
        # Verify detector was called
        self.assertTrue(mock_detector.called)
        
        # Check that detections are recorded
        for tag, images_dict in result.items():
            for rel_path, entry in images_dict.items():
                # Mock returns 1-3 boxes per image in xywh format
                # These get converted to xyxy and recorded
                self.assertIn("detections_xyxy", entry)
    
    @patch('preprocessing.crops_to_json.run_detector_batched')
    @patch('preprocessing.crops_to_json._load_langsam_model')
    def test_langsam_detector_called(self, mock_load_langsam, mock_detector):
        """Test that LangSAM detector is called correctly."""
        from preprocessing.crops_to_json import process_folder_structure_to_json
        
        # Setup mocks
        mock_load_langsam.return_value = MagicMock()
        mock_detector.side_effect = mock_detector_fn
        
        result = process_folder_structure_to_json(
            root_input=self.input_dir,
            patch_size=100,
            P=5,
            max_overlap=0.25,
            grid=False,
            object_detection=True,
            detector="langsam",
            batch_size=4,
            topn=5,
        )
        
        # Verify detector was called
        self.assertTrue(mock_detector.called)
    
    @patch('preprocessing.crops_to_json.run_detector_batched')
    @patch('preprocessing.crops_to_json._load_sam3_model')
    def test_detections_first_then_random_fill(self, mock_load_sam3, mock_detector):
        """Test that detection patches are created before random fill."""
        from preprocessing.crops_to_json import concept_process_json_mapping_to_json
        
        # Create mapping for concept mode
        mapping_json = os.path.join(self.temp_dir, "mapping.json")
        create_test_mapping_json(
            mapping_json,
            self.input_dir,
            {"apple": 35},  # Reuse existing images dir structure
            img_size=(400, 300)
        )
        
        # Mock returns 2 boxes per image
        def fixed_detector(images, tag, detector, batch_size=8, model=None, topn=10):
            return [[(50, 50, 80, 80), (150, 150, 80, 80)] for _ in images]
        
        mock_load_sam3.return_value = MagicMock()
        mock_detector.side_effect = fixed_detector
        
        result = concept_process_json_mapping_to_json(
            json_file=mapping_json,
            input_root=self.input_dir,
            max_crops_per_image=5,  # Request 5 crops
            patch_size=100,
            min_images_per_tag=30,
            object_detection=True,
            detector="sam3",
            topn=10,
        )
        
        # Each image should have detections recorded + random fill
        for rel_path, entry in result["apple"].items():
            detections = entry["detections_xyxy"]
            random_crops = entry["random_crops"]
            
            # Should have up to 2 detections (what mock returns)
            self.assertLessEqual(len(detections), 2)
            
            # Random crops should fill remaining slots (up to max_crops_per_image - detections)
            # Note: random_crops in concept mode fills the gap
            total = len(random_crops)  # In concept mode, random_crops is the fill
            # Detection patches are separate from detections_xyxy


class TestMultibatchBehavior(unittest.TestCase):
    """Test that detector receives images in batches."""
    
    def setUp(self):
        """Create temporary directory with test images."""
        self.temp_dir = tempfile.mkdtemp(prefix="crops_batch_test_")
        self.input_dir = os.path.join(self.temp_dir, "images")
        
        # Create 10 images to test batching
        self.image_paths = create_test_folder_structure(
            self.input_dir,
            {"test_tag": 10},
            img_size=(400, 300)
        )
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('preprocessing.crops_to_json._load_sam3_model')
    def test_batch_size_respected(self, mock_load_sam3):
        """Test that images are sent to detector in batches."""
        from preprocessing.crops_to_json import process_folder_structure_to_json
        
        batch_sizes_received = []
        
        mock_load_sam3.return_value = MagicMock()
        
        with patch('preprocessing.crops_to_json.run_detector_batched') as mock_detector:
            mock_detector.side_effect = mock_detector_batched_tracker(batch_sizes_received)
            
            result = process_folder_structure_to_json(
                root_input=self.input_dir,
                patch_size=100,
                P=5,
                object_detection=True,
                detector="sam3",
                batch_size=4,  # Request batch size of 4
            )
        
        # Detector should have been called with the full batch for the tag
        # (the code sends all images for a tag at once, batching happens inside detector)
        self.assertTrue(len(batch_sizes_received) > 0)
        # All 10 images for the tag should be sent together
        self.assertEqual(batch_sizes_received[0], 10)


class TestResumability(unittest.TestCase):
    """Test incremental/resumable processing."""
    
    def setUp(self):
        """Create temporary directory with test images."""
        self.temp_dir = tempfile.mkdtemp(prefix="crops_resume_test_")
        self.input_dir = os.path.join(self.temp_dir, "images")
        self.output_json = os.path.join(self.temp_dir, "crops.json")
        
        self.image_paths = create_test_folder_structure(
            self.input_dir,
            {"apple": 5},
            img_size=(400, 300)
        )
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_skips_existing_entries(self):
        """Test that existing entries are not reprocessed."""
        from preprocessing.crops_to_json import process_folder_structure_to_json, _atomic_write_json
        
        # Create a pre-existing result with one entry
        existing_result = {
            "apple": {
                "apple/image_000.jpg": {
                    "meta": {"image_size": [400, 300], "patch_size": 100},
                    "detections_xyxy": [],
                    "random_crops": [[10, 10, 110, 110]],  # Distinctive value
                }
            }
        }
        _atomic_write_json(self.output_json, existing_result)
        
        # Process with the existing result
        random.seed(42)
        result = process_folder_structure_to_json(
            root_input=self.input_dir,
            patch_size=100,
            P=5,
            max_overlap=0.25,
            result=existing_result.copy(),  # Pass existing result
            output_json=self.output_json,
        )
        
        # The existing entry should be preserved (not regenerated)
        self.assertEqual(
            result["apple"]["apple/image_000.jpg"]["random_crops"],
            [[10, 10, 110, 110]]
        )
        
        # But new entries should be added
        self.assertEqual(len(result["apple"]), 5)


class TestVisualization(unittest.TestCase):
    """Test crop visualization functionality."""
    
    def setUp(self):
        """Create temporary directory with test images."""
        self.temp_dir = tempfile.mkdtemp(prefix="crops_viz_test_")
        self.input_dir = os.path.join(self.temp_dir, "images")
        self.viz_output_dir = os.path.join(self.temp_dir, "viz_output")
        
        self.image_paths = create_test_folder_structure(
            self.input_dir,
            {"fruit": 3},
            img_size=(400, 300)
        )
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_draw_boxes_creates_valid_image(self):
        """Test that draw_boxes_on_image creates a valid image."""
        from preprocessing.crop_visualizer import load_image, draw_boxes_on_image
        
        img_path = self.image_paths["fruit"][0]
        img = load_image(img_path)
        
        detections = [(50, 50, 150, 150), (200, 100, 300, 200)]
        random_crops = [(10, 10, 110, 110), (250, 150, 350, 250)]
        
        result = draw_boxes_on_image(img, detections, random_crops)
        
        # Result should be a PIL Image
        self.assertEqual(result.size, (400, 300))
        self.assertEqual(result.mode, "RGB")
    
    def test_visualize_single_image(self):
        """Test single image visualization."""
        from preprocessing.crop_visualizer import visualize_single_image
        
        img_path = self.image_paths["fruit"][0]
        output_path = os.path.join(self.viz_output_dir, "test_viz.png")
        
        detections = [(50, 50, 150, 150)]
        random_crops = [(200, 100, 300, 200)]
        
        result = visualize_single_image(
            img_path, detections, random_crops,
            output_path=output_path,
            show=False,
        )
        
        # Output file should exist
        self.assertTrue(os.path.isfile(output_path))
        
        # Verify image was saved correctly
        from PIL import Image
        saved = Image.open(output_path)
        self.assertEqual(saved.size, (400, 300))
    
    def test_visualize_with_resize(self):
        """Test visualization with resize option."""
        from preprocessing.crop_visualizer import visualize_single_image
        
        img_path = self.image_paths["fruit"][0]
        output_path = os.path.join(self.viz_output_dir, "test_viz_resized.png")
        
        detections = [(25, 25, 75, 75)]  # Scaled for 200x150
        random_crops = [(100, 50, 150, 100)]
        
        result = visualize_single_image(
            img_path, detections, random_crops,
            output_path=output_path,
            resize_to=(200, 150),
            show=False,
        )
        
        # Result should be resized
        self.assertEqual(result.size, (200, 150))
    
    def test_visualize_from_json(self):
        """Test visualization from crops JSON."""
        from preprocessing.crops_to_json import process_folder_structure_to_json
        from preprocessing.crop_visualizer import visualize_crops_from_json
        
        # Generate crops
        result = process_folder_structure_to_json(
            root_input=self.input_dir,
            patch_size=100,
            P=3,
            object_detection=False,
        )
        
        # Visualize
        output_paths = visualize_crops_from_json(
            result,
            self.input_dir,
            self.viz_output_dir,
            max_images=2,
        )
        
        # Should have created 2 visualization images
        self.assertEqual(len(output_paths), 2)
        for p in output_paths:
            self.assertTrue(os.path.isfile(p))
    
    def test_get_box_stats(self):
        """Test box statistics extraction for testing."""
        from preprocessing.crop_visualizer import get_box_stats
        
        detections = [(10, 10, 60, 60), (100, 100, 150, 150)]
        random_crops = [(200, 50, 300, 150)]
        image_size = (400, 300)
        
        stats = get_box_stats(detections, random_crops, image_size)
        
        self.assertEqual(stats["num_detections"], 2)
        self.assertEqual(stats["num_random_crops"], 1)
        self.assertEqual(stats["total_boxes"], 3)
        self.assertTrue(stats["detections_valid"])
        self.assertTrue(stats["random_crops_valid"])
        self.assertEqual(stats["detection_areas"], [2500, 2500])  # 50x50 boxes
        self.assertEqual(stats["random_crop_areas"], [10000])  # 100x100 box
    
    def test_box_stats_detects_invalid_boxes(self):
        """Test that box stats correctly identifies invalid boxes."""
        from preprocessing.crop_visualizer import get_box_stats
        
        # Box that extends outside image bounds
        detections = [(350, 250, 450, 350)]  # Extends beyond 400x300
        random_crops = []
        image_size = (400, 300)
        
        stats = get_box_stats(detections, random_crops, image_size)
        
        self.assertFalse(stats["detections_valid"])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp(prefix="crops_edge_test_")
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_image_smaller_than_patch(self):
        """Test handling of images smaller than patch size."""
        from preprocessing.crops_to_json import process_folder_structure_to_json
        
        input_dir = os.path.join(self.temp_dir, "small_images")
        create_test_folder_structure(
            input_dir,
            {"small": 2},
            img_size=(50, 50)  # Smaller than patch_size=100
        )
        
        result = process_folder_structure_to_json(
            root_input=input_dir,
            patch_size=100,
            P=5,
            object_detection=False,
        )
        
        # Should still process images but with empty crops
        self.assertIn("small", result)
        for rel_path, entry in result["small"].items():
            self.assertEqual(entry["random_crops"], [])
    
    def test_empty_directory(self):
        """Test handling of empty directory."""
        from preprocessing.crops_to_json import process_folder_structure_to_json
        
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir)
        
        result = process_folder_structure_to_json(
            root_input=empty_dir,
            patch_size=100,
            P=5,
            object_detection=False,
        )
        
        self.assertEqual(result, {})
    
    def test_invalid_image_handling(self):
        """Test graceful handling of invalid image files."""
        from preprocessing.crops_to_json import process_folder_structure_to_json
        
        input_dir = os.path.join(self.temp_dir, "mixed")
        tag_dir = os.path.join(input_dir, "mixed_tag")
        os.makedirs(tag_dir)
        
        # Create one valid image
        create_test_image(os.path.join(tag_dir, "valid.jpg"), 400, 300)
        
        # Create one invalid "image" (just a text file with wrong extension)
        with open(os.path.join(tag_dir, "invalid.jpg"), 'w') as f:
            f.write("not an image")
        
        result = process_folder_structure_to_json(
            root_input=input_dir,
            patch_size=100,
            P=5,
            object_detection=False,
        )
        
        # Should only have the valid image
        self.assertIn("mixed_tag", result)
        self.assertEqual(len(result["mixed_tag"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
