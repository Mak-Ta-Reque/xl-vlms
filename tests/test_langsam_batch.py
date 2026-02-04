#!/usr/bin/env python3
"""
Test LangSAM batch processing functionality.

This script tests LangSAM batch inference and determines the maximum
batch size the GPU can handle.

Usage:
    python tests/test_langsam_batch.py --input_dir data/train/apple --tag "apple"
    python tests/test_langsam_batch.py --input_dir data/train/cat --tag "cat" --batch_size 8
    python tests/test_langsam_batch.py --auto_tune  # Find optimal batch size
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
        return self.batch_inference_succeeded > 0 and self.total_boxes > 0
    
    def summary(self) -> str:
        lines = [
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


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - allocated
            return {
                "allocated": allocated,
                "reserved": reserved,
                "total": total,
                "free": free,
            }
    except Exception:
        pass
    return {}


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
    batch_size: int = 8,
    topn: int = 5,
    max_size: int = 0,
    box_threshold: float = 0.5,
) -> Tuple[BatchProcessingStatus, List[List[List[int]]], float]:
    """
    Test LangSAM batch processing and track status.
    
    Returns:
        Tuple of (status, results, inference_time)
    """
    from src.langsam_utils import load_langsam, predict_bboxes_for_tag_batched, _to_pil, _extract_bboxes_from_result
    
    status = BatchProcessingStatus()
    
    # Find images
    image_paths = find_images(input_dir, max_images=num_images)
    if not image_paths:
        status.errors.append("No images found")
        return status, [], 0.0
    
    status.total_images = len(image_paths)
    
    # Load model
    print(f"Loading LangSAM model...")
    t0 = time.time()
    model = load_langsam(device="cuda")
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")
    
    mem = get_gpu_memory_info()
    if mem:
        print(f"GPU Memory after load: {mem['allocated']:.2f}GB / {mem['total']:.2f}GB")
    
    # Load and optionally resize images
    images_pil = [_to_pil(str(p)) for p in image_paths]
    original_sizes = [img.size for img in images_pil]
    
    if max_size > 0:
        print(f"Resizing images to max {max_size}px...")
        resized_images = []
        for img in images_pil:
            resized, _ = resize_image(img, max_size)
            resized_images.append(resized)
        images_pil = resized_images
        print(f"  Original sizes: {original_sizes[:3]}...")
        print(f"  Resized sizes: {[img.size for img in images_pil[:3]]}...")
    
    results = [[] for _ in range(len(images_pil))]
    
    # Process in batches
    print(f"\nProcessing {len(images_pil)} images in batches of {batch_size}...")
    t0 = time.time()
    
    for batch_start in range(0, len(images_pil), batch_size):
        batch_end = min(batch_start + batch_size, len(images_pil))
        batch_images = images_pil[batch_start:batch_end]
        batch_tags = [tag] * len(batch_images)
        
        print(f"  Batch {status.batch_inference_attempted + 1}: images {batch_start}-{batch_end} ({len(batch_images)} images)")
        status.batch_inference_attempted += 1
        
        try:
            batch_results = model.predict(batch_images, batch_tags, box_threshold=box_threshold)
            
            # Parse results
            if isinstance(batch_results, list) and len(batch_results) == len(batch_images):
                batch_boxes = 0
                for off, (img, res) in enumerate(zip(batch_images, batch_results)):
                    img_bbs = []
                    if isinstance(res, dict):
                        img_bbs = _extract_bboxes_from_result(res, img.size, prefer_masks=True)
                    results[batch_start + off] = img_bbs[:topn]
                    batch_boxes += len(img_bbs[:topn])
                
                status.batch_inference_succeeded += 1
                print(f"    ✓ Success: {batch_boxes} boxes detected")
            else:
                raise ValueError(f"Unexpected result format: {type(batch_results)}")
                
        except Exception as e:
            error_msg = str(e)
            is_oom = "out of memory" in error_msg.lower()
            status.batch_inference_failed += 1
            print(f"    ✗ Failed: {'OOM' if is_oom else error_msg[:50]}")
            status.errors.append(f"Batch {batch_start}-{batch_end}: {error_msg[:60]}")
            
            # Fallback to per-image
            print(f"    Using per-image fallback...")
            for i, img in enumerate(batch_images):
                try:
                    single_res = model.predict([img], [tag], box_threshold=box_threshold)
                    img_bbs = []
                    if isinstance(single_res, list) and len(single_res) > 0:
                        res = single_res[0]
                        if isinstance(res, dict):
                            img_bbs = _extract_bboxes_from_result(res, img.size, prefer_masks=True)
                    results[batch_start + i] = img_bbs[:topn]
                    status.fallback_used += 1
                except Exception as e2:
                    status.errors.append(f"Fallback failed: {str(e2)[:40]}")
        
        # Memory cleanup between batches
        clear_gpu_memory()
    
    inference_time = time.time() - t0
    
    # Calculate statistics
    for boxes in results:
        if boxes:
            status.images_with_detections += 1
            status.total_boxes += len(boxes)
    
    return status, results, inference_time


def auto_tune_batch_size(
    input_dir: Path,
    tag: str,
    num_images: int = 10,
    max_size: int = 0,
    box_threshold: float = 0.5,
) -> int:
    """
    Automatically find the maximum batch size that works without OOM.
    
    Returns:
        Maximum working batch size.
    """
    print(f"\n{'='*60}")
    print("Auto-tuning batch size...")
    print(f"{'='*60}")
    
    # Test batch sizes: 1, 2, 4, 8, 16, 32
    batch_sizes_to_test = [1, 2, 4, 8, 16, 32]
    max_working_batch_size = 1
    
    for bs in batch_sizes_to_test:
        print(f"\nTesting batch_size={bs}...")
        clear_gpu_memory()
        
        try:
            status, results, _ = test_batch_processing(
                input_dir=input_dir,
                tag=tag,
                num_images=min(num_images, bs * 2),  # Test with at least 2 batches
                batch_size=bs,
                max_size=max_size,                box_threshold=box_threshold,            )
            
            if status.batch_success_rate >= 1.0:
                max_working_batch_size = bs
                print(f"  ✓ batch_size={bs} works!")
            else:
                print(f"  ✗ batch_size={bs} had failures, stopping")
                break
                
        except Exception as e:
            print(f"  ✗ batch_size={bs} failed: {e}")
            break
        
        clear_gpu_memory()
    
    print(f"\n{'='*60}")
    print(f"Recommended batch_size: {max_working_batch_size}")
    print(f"{'='*60}")
    
    return max_working_batch_size


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
                "detector": "langsam",
                "tag": tag,
                "num_detections": len(boxes),
            }
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(crops_data, f, indent=2)


def print_test_result(status: BatchProcessingStatus, inference_time: float) -> bool:
    """Print final test result with clear pass/fail status."""
    print(f"\n{'='*60}")
    print("LANGSAM BATCH PROCESSING TEST RESULTS")
    print(f"{'='*60}")
    print(status.summary())
    print(f"Inference Time: {inference_time:.2f}s")
    if status.total_images > 0:
        print(f"Throughput: {status.total_images / inference_time:.2f} images/sec")
    
    print(f"\n{'='*60}")
    
    if status.batch_inference_succeeded == 0:
        print("❌ FAIL: No batch inference succeeded")
        if status.errors:
            print(f"   First error: {status.errors[0]}")
        return False
    
    if status.total_boxes == 0:
        print("⚠️  WARNING: No detections found")
        print("   Batch processing ran but found no objects.")
        return True
    
    if status.batch_success_rate < 0.5:
        print(f"⚠️  WARNING: Low batch success rate ({status.batch_success_rate:.0%})")
        print(f"   {status.batch_inference_failed} batches failed")
        return True
    
    print(f"✅ PASS: Batch processing working")
    print(f"   {status.total_boxes} boxes detected in {status.images_with_detections} images")
    print(f"   Batch success rate: {status.batch_success_rate:.0%}")
    print(f"{'='*60}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test LangSAM batch processing")
    parser.add_argument("--input_dir", type=str, default=str(ROOT_DIR / "data" / "train" / "apple"))
    parser.add_argument("--tag", type=str, default="apple")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_size", type=int, default=0, help="Resize images to max dimension (0=no resize)")
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--box_threshold", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--auto_tune", action="store_true", help="Auto-find optimal batch size")
    parser.add_argument("--no_visualize", action="store_true")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Directory not found: {input_dir}")
        sys.exit(1)
    
    output_json = Path(args.output_json) if args.output_json else ROOT_DIR / "tests" / "test_langsam_crops.json"
    
    # Auto-tune mode
    if args.auto_tune:
        optimal_bs = auto_tune_batch_size(
            input_dir=input_dir,
            tag=args.tag,
            num_images=args.num_images,
            max_size=args.max_size,
            box_threshold=args.box_threshold,
        )
        print(f"\nOptimal batch size: {optimal_bs}")
        sys.exit(0)
    
    # Regular test mode
    print(f"\n{'='*60}")
    print("LangSAM Batch Processing Test")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Tag: '{args.tag}'")
    print(f"Num Images: {args.num_images}")
    print(f"Batch Size: {args.batch_size}")
    if args.max_size > 0:
        print(f"Max Image Size: {args.max_size}px")
    
    # Run test
    clear_gpu_memory()
    
    status, results, inference_time = test_batch_processing(
        input_dir=input_dir,
        tag=args.tag,
        num_images=args.num_images,
        batch_size=args.batch_size,
        topn=args.topn,
        max_size=args.max_size,
        box_threshold=args.box_threshold,
    )
    
    # Save results
    image_paths = find_images(input_dir, max_images=args.num_images)
    save_results(image_paths, results, args.tag, input_dir, output_json)
    print(f"\nSaved results to: {output_json}")
    
    # Print test result
    passed = print_test_result(status, inference_time)
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
