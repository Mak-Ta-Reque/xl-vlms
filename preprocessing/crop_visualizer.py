"""
Crop visualization helper for deterministic testing.

This module provides functions to visualize crop locations on images,
supporting both matplotlib-based rendering and rectangle extraction for testing.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


def _ensure_pillow():
    """Ensure Pillow is available."""
    try:
        from PIL import Image, ImageDraw  # noqa: F401
        return True
    except ImportError:
        return False


def load_image(image_path: str, resize_to: Optional[Tuple[int, int]] = None) -> Any:
    """Load an image, optionally resizing it.
    
    Args:
        image_path: Path to the image file.
        resize_to: Optional (width, height) to resize to.
        
    Returns:
        PIL Image object.
    """
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    if resize_to is not None:
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
        img = img.resize(resize_to, resample=resample)
    return img


def draw_boxes_on_image(
    img: Any,
    detections_xyxy: List[Tuple[int, int, int, int]],
    random_crops: List[Tuple[int, int, int, int]],
    detection_color: Tuple[int, int, int] = (255, 0, 0),      # Red for detections
    random_crop_color: Tuple[int, int, int] = (0, 255, 0),    # Green for random crops
    detection_width: int = 3,
    random_width: int = 2,
    draw_labels: bool = True,
    draw_legend: bool = True,
) -> Any:
    """Draw bounding boxes on an image with clear distinction between detection and random crops.
    
    Args:
        img: PIL Image object.
        detections_xyxy: List of detection boxes as (x1, y1, x2, y2).
        random_crops: List of random crop boxes as (x1, y1, x2, y2).
        detection_color: RGB color for detection boxes.
        random_crop_color: RGB color for random crop boxes.
        detection_width: Line width for detection boxes.
        random_width: Line width for random crop boxes.
        draw_labels: Whether to draw labels on boxes.
        draw_legend: Whether to draw a legend in the corner.
        
    Returns:
        PIL Image with boxes drawn.
    """
    from PIL import ImageDraw, ImageFont
    
    # Make a copy to avoid modifying original
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        try:
            font = ImageFont.load_default()
            small_font = font
        except Exception:
            font = None
            small_font = None
    
    # Draw detection boxes (solid lines, thicker)
    for i, (x1, y1, x2, y2) in enumerate(detections_xyxy):
        # Draw solid rectangle
        draw.rectangle([x1, y1, x2, y2], outline=detection_color, width=detection_width)
        # Draw inner line for emphasis
        if detection_width >= 2:
            draw.rectangle([x1+1, y1+1, x2-1, y2-1], outline=detection_color, width=1)
        if draw_labels and font:
            # Draw label with background for visibility
            label = f"D{i}"
            bbox = draw.textbbox((x1 + 4, y1 + 4), label, font=small_font) if hasattr(draw, 'textbbox') else (x1+4, y1+4, x1+30, y1+18)
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=(255, 255, 255, 200))
            draw.text((x1 + 4, y1 + 4), label, fill=detection_color, font=small_font)
    
    # Draw random crop boxes (dashed style using corners)
    for i, (x1, y1, x2, y2) in enumerate(random_crops):
        # Draw main rectangle
        draw.rectangle([x1, y1, x2, y2], outline=random_crop_color, width=random_width)
        # Draw corner markers to distinguish from detections
        corner_len = min(15, (x2-x1)//4, (y2-y1)//4)
        # Top-left corner
        draw.line([(x1, y1), (x1 + corner_len, y1)], fill=random_crop_color, width=random_width + 2)
        draw.line([(x1, y1), (x1, y1 + corner_len)], fill=random_crop_color, width=random_width + 2)
        # Top-right corner
        draw.line([(x2 - corner_len, y1), (x2, y1)], fill=random_crop_color, width=random_width + 2)
        draw.line([(x2, y1), (x2, y1 + corner_len)], fill=random_crop_color, width=random_width + 2)
        # Bottom-left corner
        draw.line([(x1, y2 - corner_len), (x1, y2)], fill=random_crop_color, width=random_width + 2)
        draw.line([(x1, y2), (x1 + corner_len, y2)], fill=random_crop_color, width=random_width + 2)
        # Bottom-right corner
        draw.line([(x2 - corner_len, y2), (x2, y2)], fill=random_crop_color, width=random_width + 2)
        draw.line([(x2, y2 - corner_len), (x2, y2)], fill=random_crop_color, width=random_width + 2)
        
        if draw_labels and font:
            # Draw label with background for visibility
            label = f"R{i}"
            bbox = draw.textbbox((x1 + 4, y1 + 4), label, font=small_font) if hasattr(draw, 'textbbox') else (x1+4, y1+4, x1+30, y1+18)
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=(255, 255, 255, 200))
            draw.text((x1 + 4, y1 + 4), label, fill=random_crop_color, font=small_font)
    
    # Draw legend
    if draw_legend and font:
        img_w, img_h = img_copy.size
        legend_x = 10
        legend_y = img_h - 50
        
        # Legend background
        draw.rectangle([legend_x - 5, legend_y - 5, legend_x + 200, legend_y + 45], 
                      fill=(255, 255, 255, 230), outline=(0, 0, 0))
        
        # Detection legend entry
        draw.rectangle([legend_x, legend_y, legend_x + 20, legend_y + 15], 
                      outline=detection_color, width=2)
        draw.text((legend_x + 25, legend_y), f"Detection ({len(detections_xyxy)})", 
                 fill=detection_color, font=small_font)
        
        # Random crop legend entry  
        draw.rectangle([legend_x, legend_y + 20, legend_x + 20, legend_y + 35], 
                      outline=random_crop_color, width=2)
        draw.text((legend_x + 25, legend_y + 20), f"Random Crop ({len(random_crops)})", 
                 fill=random_crop_color, font=small_font)
    
    return img_copy


def visualize_crops_from_json(
    crops_json: Dict[str, Dict[str, dict]],
    input_root: str,
    output_dir: str,
    tag_filter: Optional[str] = None,
    max_images: int = 0,
    use_resized_size: bool = True,
    detection_color: Tuple[int, int, int] = (255, 0, 0),
    random_crop_color: Tuple[int, int, int] = (0, 255, 0),
    verbose: bool = False,
) -> List[str]:
    """Visualize crops from a crops JSON file.
    
    Args:
        crops_json: Loaded crops JSON data (tag -> {rel_path -> entry}).
        input_root: Root directory containing the original images.
        output_dir: Directory to save visualization images.
        tag_filter: If specified, only process this tag.
        max_images: Maximum images to process (0 = all).
        use_resized_size: If True, resize image to meta.image_size before drawing.
        detection_color: RGB color for detection boxes.
        random_crop_color: RGB color for random crop boxes.
        verbose: If True, print debug information about missing files.
        
    Returns:
        List of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    count = 0
    missing_count = 0
    
    for tag, images_dict in crops_json.items():
        if tag_filter is not None and tag != tag_filter:
            continue
            
        for rel_path, entry in images_dict.items():
            if max_images > 0 and count >= max_images:
                break
                
            img_path = os.path.join(input_root, rel_path)
            if not os.path.isfile(img_path):
                missing_count += 1
                if verbose and missing_count <= 5:
                    print(f"  [MISSING] {img_path}")
                continue
                
            meta = entry.get("meta", {})
            image_size = meta.get("image_size", None)
            detections = entry.get("detections_xyxy", [])
            random_crops = entry.get("random_crops", [])
            
            # Convert to tuples
            detections = [tuple(b) for b in detections]
            random_crops = [tuple(b) for b in random_crops]
            
            # Load and optionally resize
            resize_to = None
            if use_resized_size and image_size:
                resize_to = tuple(image_size)
            
            img = load_image(img_path, resize_to=resize_to)
            
            # Draw boxes
            img_viz = draw_boxes_on_image(
                img, detections, random_crops,
                detection_color=detection_color,
                random_crop_color=random_crop_color,
            )
            
            # Save
            safe_rel = rel_path.replace("/", "_").replace("\\", "_")
            out_name = f"{tag}_{safe_rel}"
            if not out_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                out_name += ".png"
            out_path = os.path.join(output_dir, out_name)
            img_viz.save(out_path)
            output_paths.append(out_path)
            count += 1
            
        if max_images > 0 and count >= max_images:
            break
    
    if missing_count > 0 and verbose:
        print(f"  ... and {missing_count - min(5, missing_count)} more missing files")
            
    return output_paths


def get_box_stats(
    detections_xyxy: List[Tuple[int, int, int, int]],
    random_crops: List[Tuple[int, int, int, int]],
    image_size: Tuple[int, int],
) -> Dict[str, Any]:
    """Get statistics about boxes for testing assertions.
    
    Args:
        detections_xyxy: Detection boxes.
        random_crops: Random crop boxes.
        image_size: (width, height) of image.
        
    Returns:
        Dictionary with box statistics.
    """
    w, h = image_size
    
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    
    def is_within_bounds(box):
        return box[0] >= 0 and box[1] >= 0 and box[2] <= w and box[3] <= h
    
    def boxes_valid(boxes):
        for b in boxes:
            if b[2] <= b[0] or b[3] <= b[1]:
                return False
            if not is_within_bounds(b):
                return False
        return True
    
    return {
        "num_detections": len(detections_xyxy),
        "num_random_crops": len(random_crops),
        "total_boxes": len(detections_xyxy) + len(random_crops),
        "detections_valid": boxes_valid(detections_xyxy),
        "random_crops_valid": boxes_valid(random_crops),
        "detection_areas": [box_area(b) for b in detections_xyxy],
        "random_crop_areas": [box_area(b) for b in random_crops],
        "image_size": image_size,
    }


def visualize_single_image(
    image_path: str,
    detections_xyxy: List[Tuple[int, int, int, int]],
    random_crops: List[Tuple[int, int, int, int]],
    output_path: Optional[str] = None,
    resize_to: Optional[Tuple[int, int]] = None,
    show: bool = False,
) -> Any:
    """Visualize crops on a single image.
    
    Args:
        image_path: Path to image file.
        detections_xyxy: Detection boxes.
        random_crops: Random crop boxes.
        output_path: If provided, save to this path.
        resize_to: If provided, resize image to (width, height).
        show: If True, display using matplotlib (for interactive use).
        
    Returns:
        PIL Image with boxes drawn.
    """
    img = load_image(image_path, resize_to=resize_to)
    img_viz = draw_boxes_on_image(img, detections_xyxy, random_crops)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        img_viz.save(output_path)
    
    if show:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.imshow(img_viz)
            plt.axis('off')
            plt.title(f"Detections: {len(detections_xyxy)}, Random: {len(random_crops)}")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available for display")
    
    return img_viz


if __name__ == "__main__":
    # Example usage / CLI
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Visualize crops from a crops JSON file")
    parser.add_argument("--crops_json", required=True, help="Path to crops JSON file")
    parser.add_argument("--input_root", required=True, help="Root directory of images")
    parser.add_argument("--output_dir", required=True, help="Output directory for visualizations")
    parser.add_argument("--tag", default=None, help="Filter by tag")
    parser.add_argument("--max_images", type=int, default=10, help="Max images to visualize")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print debug info about missing files")
    
    args = parser.parse_args()
    
    with open(args.crops_json, 'r') as f:
        crops_data = json.load(f)
    
    # Print summary of crops JSON
    total_images = sum(len(v) for v in crops_data.values())
    print(f"Crops JSON: {len(crops_data)} tags, {total_images} images total")
    
    # Show first few paths to help debug input_root issues
    if args.verbose:
        print(f"Input root: {args.input_root}")
        print("Sample paths from crops JSON:")
        shown = 0
        for tag, images_dict in crops_data.items():
            for rel_path in images_dict.keys():
                full_path = os.path.join(args.input_root, rel_path)
                exists = "✓" if os.path.isfile(full_path) else "✗"
                print(f"  [{exists}] {full_path}")
                shown += 1
                if shown >= 3:
                    break
            if shown >= 3:
                break
        print()
    
    paths = visualize_crops_from_json(
        crops_data,
        args.input_root,
        args.output_dir,
        tag_filter=args.tag,
        max_images=args.max_images,
        verbose=args.verbose,
    )
    
    print(f"Saved {len(paths)} visualizations to {args.output_dir}")
    
    if len(paths) == 0 and total_images > 0:
        print("\nNo visualizations saved! Possible causes:")
        print("  1. --input_root doesn't match the paths in crops JSON")
        print("  2. Original images have been moved or deleted")
        print("  Hint: Use --verbose to see which paths are being checked")
