import os
import random
import argparse
from PIL import Image
from pathlib import Path


# -------------------- Utility Functions --------------------

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def resize_if_needed(img, min_w, min_h):
    """Resize image if smaller than (min_w, min_h)."""
    w, h = img.size
    if w < min_w or h < min_h:
        return img.resize(
            (max(w, min_w), max(h, min_h)), 
            Image.Resampling.LANCZOS
        )
    return img


def center_crop(img, crop_size):
    """Resize image to crop_size (h, w) maintaining aspect ratio, then center crop."""
    ch, cw = crop_size  # crop_size is (height, width)
    
    # Calculate aspect ratios
    img_ratio = img.width / img.height
    target_ratio = cw / ch
    
    if img_ratio > target_ratio:
        # Image is wider, fit to height
        new_height = ch
        new_width = int(ch * img_ratio)
    else:
        # Image is taller, fit to width
        new_width = cw
        new_height = int(cw / img_ratio)
    
    # Resize maintaining aspect ratio
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Center crop to exact size
    left = (new_width - cw) // 2
    top = (new_height - ch) // 2
    return img.crop((left, top, left + cw, top + ch))



# -------------------- Patch Functions --------------------

def create_grid_patches(image_path, patch_size, output_dir, crop_size=None):
    """Create non-overlapping grid patches."""
    img = Image.open(image_path).convert("RGB")
    if crop_size:
        img = center_crop(img, crop_size)
    img = resize_if_needed(img, patch_size, patch_size)

    os.makedirs(output_dir, exist_ok=True)
    name = Path(image_path).stem
    w, h = img.size
    pid = 0

    for top in range(0, h - patch_size + 1, patch_size):
        for left in range(0, w - patch_size + 1, patch_size):
            patch = img.crop((left, top, left + patch_size, top + patch_size))
            patch.save(os.path.join(output_dir, f"{name}_grid_patch_{pid}.png"))
            pid += 1
    print(f"[GRID] {pid} patches saved for {name}")


def create_random_patches(image_path, patch_size, output_dir, P,
                          max_overlap=0.25, crop_size=None,
                          max_attempts_per_patch=100):
    """Create random patches with limited overlap."""
    img = Image.open(image_path).convert("RGB")
    if crop_size:
        img = center_crop(img, crop_size)
    img = resize_if_needed(img, patch_size, patch_size)
    w, h = img.size

    os.makedirs(output_dir, exist_ok=True)
    name = Path(image_path).stem
    saved = []
    pid, attempts = 0, 0

    while pid < P and attempts < P * max_attempts_per_patch:
        attempts += 1
        left = random.randint(0, w - patch_size)
        top = random.randint(0, h - patch_size)
        box = (left, top, left + patch_size, top + patch_size)

        if any(calculate_iou(box, ex) > max_overlap for ex in saved):
            continue

        patch = img.crop(box)
        patch.save(os.path.join(output_dir, f"{name}_rand_patch_{pid}.png"))
        saved.append(box)
        pid += 1
    print(f"[RANDOM] {pid} patches saved for {name}")


# -------------------- Folder Processor --------------------

def process_folder_structure(root_input, root_output, patch_size=128, P=10, 
                             max_overlap=0.25, crop_size=None, use_grid=False, 
                             use_random=True):
    """Walk through folder, extract patches, save to mirrored structure."""
    for subdir, _, files in os.walk(root_input):
        rel_path = os.path.relpath(subdir, root_input)
        output_subdir = os.path.join(root_output, rel_path)
        os.makedirs(output_subdir, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subdir, file)
                if use_grid:
                    create_grid_patches(image_path, patch_size, output_subdir, crop_size)
                if use_random:
                    create_random_patches(image_path, patch_size, output_subdir, P, max_overlap, crop_size)


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description="Extract image patches (grid and/or random).")
    parser.add_argument("--input_root", type=str, required=True, help="Path to input directory")
    parser.add_argument("--output_root", type=str, required=True, help="Path to output directory")
    parser.add_argument("--patch_size", type=int, default=128, help="Size of each square patch")
    parser.add_argument("--patches_per_image", type=int, default=10, help="Number of random patches per image")
    parser.add_argument("--max_overlap", type=float, default=0.25, help="Max IoU allowed between random patches")
    parser.add_argument("--crop_size", type=int, nargs=2, metavar=("W", "H"), default=None,
                        help="Center crop size (W H). Skip if not needed.")
    parser.add_argument("--grid", action="store_true", help="Enable grid-based patching")
    parser.add_argument("--random", action="store_true", help="Enable random patching")

    args = parser.parse_args()

    # If neither is selected, default to random
    use_grid = args.grid
    use_random = args.random or not args.grid

    process_folder_structure(
        args.input_root,
        args.output_root,
        patch_size=args.patch_size,
        P=args.patches_per_image,
        max_overlap=args.max_overlap,
        crop_size=tuple(args.crop_size) if args.crop_size else None,
        use_grid=use_grid,
        use_random=use_random
    )


if __name__ == "__main__":
    main()




"""
# Example usage
# Random patch extraction
python patch_extractor.py \
  --input_root /mnt/abka03/xlvlm_data/noidle \
  --output_root /mnt/abka03/xlvlm_data/noidle_crops \
  --patch_size 200 \
  --crop_size 500 500 \
  --patches_per_image 30 \
  --max_overlap 0.3
  --random

# Grid patch extraction
python patch_extractor.py \
  --input_root data/images \
  --output_root data/patches \
  --patch_size 180 \
  --grid

  
# Grid and random patch extraction
python patch_extractor.py \
  --input_root /mnt/abka03/xlvlm_data/noidle/train \
  --output_root  /mnt/abka03/xlvlm_data/noidle_crops/train \
  --patch_size 300 \
  --patches_per_image 20 \
  --max_overlap 0.2 \
  --crop_size 1500 1500 \
  --grid \
  --random
"""