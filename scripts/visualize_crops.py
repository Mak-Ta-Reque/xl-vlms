#!/usr/bin/env python3
"""
Visualize crop bounding boxes from crops.json on images.

This script reads a crops JSON file and plots bounding boxes
(both detections and random crops) on sample images.

Usage:
    python scripts/visualize_crops.py --crops_json outputs/inference/crops.json --input_root data --num_tags 5
    python scripts/visualize_crops.py --crops_json /tmp/sam3_test_output.json --input_root data/grids --num_tags 3
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install matplotlib pillow numpy")
    sys.exit(1)


def load_crops_json(json_path: str) -> Dict:
    """Load crops JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def draw_boxes_on_image(
    ax: plt.Axes,
    image: Image.Image,
    detections: List[List[int]],
    random_crops: List[List[int]],
    title: str = "",
):
    """Draw bounding boxes on an image.
    
    Args:
        ax: Matplotlib axes to draw on.
        image: PIL Image.
        detections: List of [x1, y1, x2, y2] detection boxes.
        random_crops: List of [x1, y1, x2, y2] random crop boxes.
        title: Title for the subplot.
    """
    ax.imshow(image)
    ax.set_title(title, fontsize=8, wrap=True)
    ax.axis('off')
    
    # Draw detection boxes (red, thicker)
    for i, box in enumerate(detections):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            linestyle='-',
            label='Detection' if i == 0 else None
        )
        ax.add_patch(rect)
        # Add label
        ax.text(x1, y1 - 2, f'D{i+1}', fontsize=6, color='red', 
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
    
    # Draw random crop boxes (blue, dashed)
    for i, box in enumerate(random_crops):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=1.5,
            edgecolor='blue',
            facecolor='none',
            linestyle='--',
            label='Random Crop' if i == 0 else None
        )
        ax.add_patch(rect)


def visualize_tag(
    tag: str,
    tag_data: Dict[str, Dict],
    input_root: Path,
    num_images: int = 4,
    output_dir: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """Visualize crops for a single tag.
    
    Args:
        tag: Tag name (concept).
        tag_data: Dictionary mapping relative paths to crop data.
        input_root: Root directory for images.
        num_images: Number of images to display per tag.
        output_dir: Optional directory to save the figure.
    
    Returns:
        Matplotlib figure or None if no images found.
    """
    # Get image paths that exist
    valid_images = []
    for rel_path, data in tag_data.items():
        img_path = input_root / rel_path
        if img_path.exists():
            valid_images.append((rel_path, data, img_path))
    
    if not valid_images:
        print(f"  No valid images found for tag '{tag}'")
        return None
    
    # Sample images
    if len(valid_images) > num_images:
        valid_images = random.sample(valid_images, num_images)
    
    # Create figure
    n_cols = min(len(valid_images), 4)
    n_rows = (len(valid_images) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle(f'Tag: "{tag}" ({len(tag_data)} images total)', fontsize=12, fontweight='bold')
    
    # Flatten axes for easy iteration
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for idx, (rel_path, data, img_path) in enumerate(valid_images):
        ax = axes[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            ax.set_title(f"Error loading: {rel_path}")
            ax.axis('off')
            continue
        
        detections = data.get('detections_xyxy', [])
        random_crops = data.get('random_crops', [])
        meta = data.get('meta', {})
        
        # Build title
        img_name = Path(rel_path).name
        if len(img_name) > 30:
            img_name = img_name[:27] + "..."
        title = f"{img_name}\nDet: {len(detections)}, Crops: {len(random_crops)}"
        
        draw_boxes_on_image(ax, img, detections, random_crops, title)
    
    # Hide unused axes
    for idx in range(len(valid_images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = tag.replace('/', '_').replace(' ', '_')[:50]
        save_path = output_dir / f"crops_viz_{safe_tag}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize crop bounding boxes from crops.json"
    )
    parser.add_argument(
        "--crops_json",
        type=str,
        required=True,
        help="Path to crops JSON file"
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Root directory for images"
    )
    parser.add_argument(
        "--num_tags",
        type=int,
        default=5,
        help="Number of random tags to visualize (default: 5)"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images per tag (default: 4)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualization images (optional)"
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated list of specific tags to visualize (optional)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (default: save only if output_dir specified)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    # Load crops JSON
    print(f"Loading crops from: {args.crops_json}")
    crops_data = load_crops_json(args.crops_json)
    
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Get tags to visualize
    all_tags = list(crops_data.keys())
    print(f"Found {len(all_tags)} tags in crops JSON")
    
    if args.tags:
        # Use specified tags
        selected_tags = [t.strip() for t in args.tags.split(',')]
        # Filter to tags that exist
        selected_tags = [t for t in selected_tags if t in crops_data]
    else:
        # Random sample
        num_tags = min(args.num_tags, len(all_tags))
        selected_tags = random.sample(all_tags, num_tags)
    
    print(f"Visualizing {len(selected_tags)} tags: {selected_tags}")
    
    # Summary statistics
    total_images = 0
    total_detections = 0
    total_crops = 0
    
    for tag in all_tags:
        for img_data in crops_data[tag].values():
            total_images += 1
            total_detections += len(img_data.get('detections_xyxy', []))
            total_crops += len(img_data.get('random_crops', []))
    
    print(f"\n=== Crops JSON Summary ===")
    print(f"Total tags: {len(all_tags)}")
    print(f"Total images: {total_images}")
    print(f"Total detections: {total_detections}")
    print(f"Total random crops: {total_crops}")
    print(f"Avg detections/image: {total_detections/max(1,total_images):.2f}")
    print(f"Avg crops/image: {total_crops/max(1,total_images):.2f}")
    print("=" * 30 + "\n")
    
    # Visualize each selected tag
    figures = []
    for tag in selected_tags:
        print(f"Processing tag: '{tag}' ({len(crops_data[tag])} images)")
        tag_data = crops_data[tag]
        
        fig = visualize_tag(
            tag=tag,
            tag_data=tag_data,
            input_root=input_root,
            num_images=args.num_images,
            output_dir=output_dir,
        )
        if fig:
            figures.append(fig)
    
    # Show plots if requested
    if args.show and figures:
        plt.show()
    elif not output_dir:
        print("\nTip: Use --output_dir to save figures or --show to display interactively")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
