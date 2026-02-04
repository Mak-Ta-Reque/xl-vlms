#!/usr/bin/env python3
"""
Test SAM3 batch processing functionality.

This script validates that SAM3 batch inference works correctly and
can detect when batch processing fails or falls back to per-image mode.

Usage:
    python tests/test_sam3_batch.py --input_dir data/train/apple --tag "apple"
    python tests/test_sam3_batch.py --input_dir data/train/cat --tag "cat" --num_images 10
"""

import os
import sys
import json
import time
import argparse
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

# Memory config
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class BatchProcessingStatus:
    """Track batch processing status and failures."""
    
    def __init__(self):
        self.batch_api_available = False
        self.batch_inference_attempted = 0
        self.batch_inference_succeeded = 0
        self.batch_inference_failed = 0
        self.fallback_used = 0
        self.total_images = 0
        self.images_with_detections = 0
        self.total_boxes = 0
        self.errors: List[str] = []
    
    @property
    def batch_success_rate(self) -> float:
        if self.batch_inference_attempted == 0:
            return 0.0
        return self.batch_inference_succeeded / self.batch_inference_attempted
    
    @property
    def is_working(self) -> bool:
        """Check if batch processing is working properly."""
        return (
            self.batch_api_available and 
            self.batch_inference_succeeded > 0 and
            self.total_boxes > 0
        )
    
    def summary(self) -> str:
        lines = [
            f"Batch API Available: {self.batch_api_available}",
            f"Batches Attempted: {self.batch_inference_attempted}",
            f"Batches Succeeded: {self.batch_inference_succeeded}",
            f"Batches Failed: {self.batch_inference_failed}",
            f"Fallback Used: {self.fallback_used} times",
            f"Success Rate: {self.batch_success_rate:.1%}",
            f"Total Images: {self.total_images}",
            f"Images with Detections: {self.images_with_detections}",
            f"Total Boxes: {self.total_boxes}",
        ]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors[:3]:
                lines.append(f"  - {e[:80]}")
        return "\n".join(lines)


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def find_images(input_dir: Path, max_images: int = -1) -> List[Path]:
    """Find images in directory."""
    extensions = ('.jpg', '.jpeg', '.png', '.webp')
    images = []
    for ext in extensions:
        images.extend(input_dir.rglob(f"*{ext}"))
        images.extend(input_dir.rglob(f"*{ext.upper()}"))
    images = sorted(set(images))
    if max_images > 0:
        images = images[:max_images]
    return images


def resize_image(img, max_size: int) -> tuple:
    """Resize image to max_size on longest edge, return (resized_img, scale_factor)."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img, 1.0
    
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), resample=3)  # LANCZOS
    return resized, scale


def test_batch_processing(
    input_dir: Path,
    tag: str,
    num_images: int = 10,
    batch_size: int = 1,
    topn: int = 5,
    confidence: float = 0.5,
    max_size: int = 0,
) -> Tuple[BatchProcessingStatus, List[List[List[int]]]]:
    """
    Test SAM3 batch processing and track status.
    
    Args:
        max_size: If > 0, resize images to this max dimension before processing.
    
    Returns:
        Tuple of (status, results) where results is list of bboxes per image.
    """
    from src.sam3_utils import load_sam3, _predict_bboxes_batched_internal, _predict_bboxes_per_image, _to_pil
    import torch
    
    status = BatchProcessingStatus()
    
    # Find images
    image_paths = find_images(input_dir, max_images=num_images)
    if not image_paths:
        status.errors.append("No images found")
        return status, []
    
    status.total_images = len(image_paths)
    
    # Load model
    print(f"Loading SAM3 model...")
    model_dict = load_sam3(device="cuda", confidence_threshold=confidence)
    status.batch_api_available = model_dict.get("batch_supported", False)
    
    if not status.batch_api_available:
        status.errors.append("Batch API not available in SAM3 installation")
        # Still run per-image as fallback
        images_pil = [_to_pil(str(p)) for p in image_paths]
        results = _predict_bboxes_per_image(model_dict, images_pil, tag, topn)
        status.fallback_used = len(image_paths)
    else:
        # Test batch processing
        images_pil = [_to_pil(str(p)) for p in image_paths]
        original_sizes = [img.size for img in images_pil]  # Store original sizes
        scale_factors = [1.0] * len(images_pil)
        
        # Resize if max_size specified
        if max_size > 0:
            print(f"Resizing images to max {max_size}px...")
            resized_images = []
            for i, img in enumerate(images_pil):
                resized, scale = resize_image(img, max_size)
                resized_images.append(resized)
                scale_factors[i] = scale
            images_pil = resized_images
            print(f"  Original sizes: {original_sizes[:3]}...")
            print(f"  Resized sizes: {[img.size for img in images_pil[:3]]}...")
        
        image_sizes = [img.size for img in images_pil]  # Sizes for inference
        
        results = [[] for _ in range(len(images_pil))]
        
        # Process in batches and track success/failure
        print(f"\nProcessing {len(images_pil)} images in batches of {batch_size}...")
        for batch_start in range(0, len(images_pil), batch_size):
            batch_end = min(batch_start + batch_size, len(images_pil))
            batch_images = images_pil[batch_start:batch_end]
            
            print(f"  Batch {status.batch_inference_attempted + 1}: images {batch_start}-{batch_end} ({len(batch_images)} images)")
            status.batch_inference_attempted += 1
            
            try:
                # Try batch inference
                batch_results = _run_single_batch(
                    model_dict, batch_images, tag, topn, 
                    batch_start, image_sizes[batch_start:batch_end]
                )
                
                # Check if we got results
                batch_boxes = sum(len(b) for b in batch_results)
                if batch_boxes > 0 or len(batch_images) == 1:
                    # Success - got results or single image (may legitimately have no detections)
                    status.batch_inference_succeeded += 1
                    print(f"    ✓ Success: {batch_boxes} boxes detected")
                    for i, boxes in enumerate(batch_results):
                        results[batch_start + i] = boxes
                else:
                    # No boxes for entire batch - suspicious
                    status.batch_inference_failed += 1
                    print(f"    ⚠ No detections in batch")
                    status.errors.append(f"Batch {batch_start}-{batch_end}: No detections (possible failure)")
                    
            except Exception as e:
                status.batch_inference_failed += 1
                error_msg = str(e)
                is_oom = "out of memory" in error_msg.lower()
                print(f"    ✗ Failed: {'OOM' if is_oom else error_msg[:50]}")
                status.errors.append(f"Batch {batch_start}-{batch_end}: {error_msg[:60]}")
                
                # Fallback to per-image
                print(f"  Batch failed, using per-image fallback...")
                for i, img in enumerate(batch_images):
                    try:
                        fallback_results = _predict_bboxes_per_image(model_dict, [img], tag, topn)
                        results[batch_start + i] = fallback_results[0]
                        status.fallback_used += 1
                    except Exception as e2:
                        status.errors.append(f"Fallback failed for image {batch_start + i}: {str(e2)[:40]}")
            
            # Memory cleanup
            clear_gpu_memory()
    
    # Calculate statistics
    for boxes in results:
        if boxes:
            status.images_with_detections += 1
            status.total_boxes += len(boxes)
    
    return status, results


def _run_single_batch(
    model_dict: Dict,
    batch_images: List,
    tag: str,
    topn: int,
    batch_start: int,
    batch_sizes: List[Tuple[int, int]],
) -> List[List[List[int]]]:
    """Run batch inference on a single batch."""
    import torch
    from src.sam3_utils import (
        _create_datapoint_with_text_prompt,
        _extract_bboxes_from_postprocessed_result,
    )
    
    model = model_dict["model"]
    transform = model_dict["transform"]
    postprocessor = model_dict["postprocessor"]
    collate_fn = model_dict["collate_fn"]
    copy_to_device_fn = model_dict["copy_to_device_fn"]
    device = model_dict.get("device", torch.device("cuda"))
    
    # Create datapoints
    datapoints = []
    query_id_to_idx = {}
    
    for i, img in enumerate(batch_images):
        query_id = batch_start + i + 1
        datapoint = _create_datapoint_with_text_prompt(img, tag, query_id)
        datapoint = transform(datapoint)
        datapoints.append(datapoint)
        query_id_to_idx[query_id] = i
    
    # Collate and run
    batch = collate_fn(datapoints, dict_key="dummy")["dummy"]
    batch = copy_to_device_fn(batch, device, non_blocking=True)
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch)
    
    # Post-process
    processed_results = postprocessor.process_results(output, batch.find_metadatas)
    
    # Extract boxes
    results = [[] for _ in range(len(batch_images))]
    for query_id, result in processed_results.items():
        if query_id in query_id_to_idx:
            idx = query_id_to_idx[query_id]
            img_size = batch_sizes[idx]
            boxes = _extract_bboxes_from_postprocessed_result(result, img_size, topn)
            results[idx] = boxes
    
    return results


def save_results(
    image_paths: List[Path],
    results: List[List[List[int]]],
    tag: str,
    input_dir: Path,
    output_path: Path,
):
    """Save results to crops JSON format."""
    crops_data = {tag: {}}
    
    for img_path, boxes in zip(image_paths, results):
        try:
            rel_path = str(img_path.relative_to(input_dir))
        except ValueError:
            rel_path = img_path.name
        
        # Convert xywh to xyxy
        detections_xyxy = [[x, y, x + w, y + h] for x, y, w, h in boxes]
        
        crops_data[tag][rel_path] = {
            "detections_xyxy": detections_xyxy,
            "random_crops": [],
            "meta": {
                "detector": "sam3",
                "tag": tag,
                "num_detections": len(boxes),
            }
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(crops_data, f, indent=2)


def run_visualization(crops_json: Path, input_root: Path, num_images: int = 5):
    """Run visualization script."""
    import subprocess
    
    viz_script = ROOT_DIR / "scripts" / "visualize_crops.py"
    output_dir = ROOT_DIR / "tests" / "viz_output"
    
    cmd = [
        sys.executable, str(viz_script),
        "--crops_json", str(crops_json),
        "--input_root", str(input_root),
        "--num_tags", "1",
        "--num_images", str(num_images),
        "--output_dir", str(output_dir),
    ]
    
    subprocess.run(cmd, capture_output=True)
    print(f"Visualization saved to: {output_dir}")


def print_test_result(status: BatchProcessingStatus, inference_time: float):
    """Print final test result with clear pass/fail status."""
    print(f"\n{'='*60}")
    print("SAM3 BATCH PROCESSING TEST RESULTS")
    print(f"{'='*60}")
    print(status.summary())
    print(f"Inference Time: {inference_time:.2f}s")
    
    print(f"\n{'='*60}")
    
    if not status.batch_api_available:
        print("❌ FAIL: Batch API not available")
        print("   SAM3 batch processing APIs could not be imported.")
        print("   Check SAM3 installation.")
        return False
    
    if status.batch_inference_succeeded == 0:
        print("❌ FAIL: No batch inference succeeded")
        print("   All batch attempts failed or fell back to per-image.")
        if status.errors:
            print(f"   First error: {status.errors[0]}")
        return False
    
    if status.total_boxes == 0:
        print("⚠️  WARNING: No detections found")
        print("   Batch processing ran but found no objects.")
        print("   This may be expected if images don't contain the target.")
        return True  # Not a batch failure
    
    if status.batch_success_rate < 0.5:
        print(f"⚠️  WARNING: Low batch success rate ({status.batch_success_rate:.0%})")
        print(f"   {status.batch_inference_failed} batches failed")
        print("   Consider using batch_size=1 for stability.")
        return True  # Partial success
    
    print(f"✅ PASS: Batch processing working")
    print(f"   {status.total_boxes} boxes detected in {status.images_with_detections} images")
    print(f"   Batch success rate: {status.batch_success_rate:.0%}")
    print(f"{'='*60}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test SAM3 batch processing")
    parser.add_argument("--input_dir", type=str, default=str(ROOT_DIR / "data" / "train" / "apple"))
    parser.add_argument("--tag", type=str, default="apple")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1, max 2 with --max_size 256)")
    parser.add_argument("--max_size", type=int, default=0, help="Resize images to max dimension (use 256 for batch_size>1)")
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--viz_samples", type=int, default=5)
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Directory not found: {input_dir}")
        sys.exit(1)
    
    output_json = Path(args.output_json) if args.output_json else ROOT_DIR / "tests" / "test_sam3_crops.json"
    
    print(f"\n{'='*60}")
    print("SAM3 Batch Processing Test")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Tag: '{args.tag}'")
    print(f"Num Images: {args.num_images}")
    print(f"Batch Size: {args.batch_size}")
    if args.max_size > 0:
        print(f"Max Image Size: {args.max_size}px")
    
    # Run test
    clear_gpu_memory()
    t0 = time.time()
    
    status, results = test_batch_processing(
        input_dir=input_dir,
        tag=args.tag,
        num_images=args.num_images,
        batch_size=args.batch_size,
        topn=args.topn,
        confidence=args.confidence,
        max_size=args.max_size,
    )
    
    inference_time = time.time() - t0
    
    # Save results
    image_paths = find_images(input_dir, max_images=args.num_images)
    save_results(image_paths, results, args.tag, input_dir, output_json)
    print(f"\nSaved results to: {output_json}")
    
    # Print test result
    passed = print_test_result(status, inference_time)
    
    # Visualization
    if args.visualize and not args.no_visualize and status.total_boxes > 0:
        print(f"\nGenerating visualization...")
        run_visualization(output_json, input_dir, args.viz_samples)
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
