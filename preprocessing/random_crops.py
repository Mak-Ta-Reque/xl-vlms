def create_grid_patches(image_path, patch_size, output_dir, center_crop=True):
    img = Image.open(image_path)
    img_name = Path(image_path).stem
    original_width, original_height = img.size

    if center_crop:
        crop_width, crop_height = 500, 500
        if original_width < crop_width or original_height < crop_height:
            new_w = max(original_width, crop_width)
            new_h = max(original_height, crop_height)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            original_width, original_height = img.size
        left = (original_width - crop_width) // 2
        top = (original_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        img = img.crop((left, top, right, bottom))
        width, height = img.size
        if width < patch_size or height < patch_size:
            new_w = max(width, patch_size)
            new_h = max(height, patch_size)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            width, height = img.size
    else:
        width, height = img.size
        if width < patch_size or height < patch_size:
            new_w = max(width, patch_size)
            new_h = max(height, patch_size)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            width, height = img.size

    os.makedirs(output_dir, exist_ok=True)
    patch_id = 0
    for top in range(0, height - patch_size + 1, patch_size):
        for left in range(0, width - patch_size + 1, patch_size):
            right = left + patch_size
            bottom = top + patch_size
            patch_img = img.crop((left, top, right, bottom)).convert("RGB")
            patch_filename = f"{img_name}_patch_{patch_id}.png"
            patch_img.save(os.path.join(output_dir, patch_filename))
            patch_id += 1
import os
import random
import argparse
from PIL import Image
from pathlib import Path

def calculate_iou(boxA, boxB):
    """Calculate the Intersection Over Union (IoU) between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    return inter_area / boxA_area

def ___create_random_patches(image_path, patch_size, output_dir, P, max_overlap_ratio=0.25, max_attempts_per_patch=100):
    img = Image.open(image_path)
    img_name = Path(image_path).stem
    width, height = img.size

    # Skip image if it's smaller than the patch size
    if width < patch_size or height < patch_size:
        print(f"Skipping {image_path}: smaller than patch size.")
        return

    os.makedirs(output_dir, exist_ok=True)

    saved_patches = []
    patches_created = 0
    attempts = 0

    while patches_created < P and attempts < P * max_attempts_per_patch:
        attempts += 1
        left = random.randint(0, width - patch_size)
        top = random.randint(0, height - patch_size)
        right = left + patch_size
        bottom = top + patch_size
        new_patch = (left, top, right, bottom)

        too_much_overlap = any(
            calculate_iou(new_patch, existing_patch) > max_overlap_ratio
            for existing_patch in saved_patches
        )

        if too_much_overlap:
            continue

        patch_img = img.crop(new_patch).convert("RGB")
        patch_filename = f"{img_name}_patch_{patches_created}.png"
        patch_img.save(os.path.join(output_dir, patch_filename))
        saved_patches.append(new_patch)
        patches_created += 1


def calculate_iou(box1, box2):
    # Calculates Intersection over Union for two boxes
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea

    if unionArea == 0:
        return 0
    return interArea / unionArea

from PIL import Image
import os
import random
from pathlib import Path

def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea

    if unionArea == 0:
        return 0
    return interArea / unionArea

def create_random_patches(image_path, patch_size, output_dir, P, max_overlap_ratio=0.25, max_attempts_per_patch=100, center_crop=True):
    img = Image.open(image_path)
    img_name = Path(image_path).stem
    original_width, original_height = img.size

    if center_crop:
        crop_width, crop_height = 500, 500
        if original_width < crop_width or original_height < crop_height:
            new_w = max(original_width, crop_width)
            new_h = max(original_height, crop_height)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            original_width, original_height = img.size
        left = (original_width - crop_width) // 2
        top = (original_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        img = img.crop((left, top, right, bottom))
        width, height = img.size
        if width < patch_size or height < patch_size:
            new_w = max(width, patch_size)
            new_h = max(height, patch_size)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            width, height = img.size
    else:
        # No center crop, just upscale if needed
        width, height = img.size
        if width < patch_size or height < patch_size:
            new_w = max(width, patch_size)
            new_h = max(height, patch_size)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            width, height = img.size

    os.makedirs(output_dir, exist_ok=True)
    saved_patches = []
    patches_created = 0
    attempts = 0
    while patches_created < P and attempts < P * max_attempts_per_patch:
        attempts += 1
        left = random.randint(0, width - patch_size)
        top = random.randint(0, height - patch_size)
        right = left + patch_size
        bottom = top + patch_size
        new_patch = (left, top, right, bottom)
        too_much_overlap = any(
            calculate_iou(new_patch, existing_patch) > max_overlap_ratio
            for existing_patch in saved_patches
        )
        if too_much_overlap:
            continue
        patch_img = img.crop(new_patch).convert("RGB")
        patch_filename = f"{img_name}_patch_{patches_created}.png"
        patch_img.save(os.path.join(output_dir, patch_filename))
        saved_patches.append(new_patch)
        patches_created += 1




def __create_random_patches(image_path, patch_size, output_dir, P, max_overlap_ratio=0.25, max_attempts_per_patch=100):
    img = Image.open(image_path)
    img_name = Path(image_path).stem
    width, height = img.size

    os.makedirs(output_dir, exist_ok=True)

    if width < patch_size or height < patch_size:
        print(f"{image_path} is smaller than patch size. Resizing entire image to patch size.")
        resized_img = img.resize((patch_size, patch_size), Image.LANCZOS).convert("RGB")
        patch_filename = f"{img_name}_patch_0.png"
        resized_img.save(os.path.join(output_dir, patch_filename))
        return

    saved_patches = []
    patches_created = 0
    attempts = 0

    while patches_created < P and attempts < P * max_attempts_per_patch:
        attempts += 1
        left = random.randint(0, width - patch_size)
        top = random.randint(0, height - patch_size)
        right = left + patch_size
        bottom = top + patch_size
        new_patch = (left, top, right, bottom)

        too_much_overlap = any(
            calculate_iou(new_patch, existing_patch) > max_overlap_ratio
            for existing_patch in saved_patches
        )

        if too_much_overlap:
            continue

        patch_img = img.crop(new_patch).convert("RGB")
        patch_filename = f"{img_name}_patch_{patches_created}.png"
        patch_img.save(os.path.join(output_dir, patch_filename))
        saved_patches.append(new_patch)
        patches_created += 1

def _create_random_patches(image_path, patch_size, output_dir, P, max_overlap_ratio=0.25, max_attempts_per_patch=100):
    img = Image.open(image_path)
    img_name = Path(image_path).stem
    width, height = img.size

    if width < patch_size or height < patch_size:
        print(f"Skipping {image_path}: Image smaller than patch size.")
        return

    os.makedirs(output_dir, exist_ok=True)
    saved_patches = []

    patches_created = 0
    attempts = 0

    while patches_created < P and attempts < P * max_attempts_per_patch:
        attempts += 1
        left = random.randint(0, width - patch_size)
        top = random.randint(0, height - patch_size)
        right = left + patch_size
        bottom = top + patch_size
        new_patch = (left, top, right, bottom)

        too_much_overlap = any(
            calculate_iou(new_patch, existing_patch) > max_overlap_ratio
            for existing_patch in saved_patches
        )

        if too_much_overlap:
            continue

        patch_img = img.crop(new_patch)
        patch_filename = f"{img_name}_patch_{patches_created}.png"
        patch_img.save(os.path.join(output_dir, patch_filename))
        saved_patches.append(new_patch)
        patches_created += 1

def process_folder_structure(root_input, root_output, patch_size=128, P=10, max_overlap=0.25, center_crop=True, grid=False):
    for subdir, _, files in os.walk(root_input):
        rel_path = os.path.relpath(subdir, root_input)
        output_subdir = os.path.join(root_output, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subdir, file)
                if grid:
                    create_grid_patches(image_path, patch_size, output_subdir, center_crop)
                else:
                    create_random_patches(image_path, patch_size, output_subdir, P, max_overlap, center_crop)

def main():
    parser = argparse.ArgumentParser(description="Extract random image patches with limited overlap.")
    parser.add_argument("--input_root", type=str, required=True, help="Path to input directory")
    parser.add_argument("--output_root", type=str, required=True, help="Path to output directory")
    parser.add_argument("--patch_size", type=int, default=128, help="Size of each square patch")
    parser.add_argument("--patches_per_image", type=int, default=10, help="Number of patches per image")
    parser.add_argument("--max_overlap", type=float, default=0.25, help="Maximum allowed overlap ratio (0.0 to 1.0)")
    parser.add_argument("--center_crop", dest="center_crop", action="store_true", help="Use center crop before patch extraction")
    parser.add_argument("--no_center_crop", dest="center_crop", action="store_false", help="Do not use center crop before patch extraction")
    parser.set_defaults(center_crop=True)
    parser.add_argument("--grid", action="store_false", help="Use grid-based cropping instead of random cropping")

    args = parser.parse_args()

    process_folder_structure(
        args.input_root,
        args.output_root,
        args.patch_size,
        args.patches_per_image,
        args.max_overlap,
        args.center_crop,
        args.grid
    )

if __name__ == "__main__":
    main()

"""
python   preprocessing/random_crops.py \
  --input_root /mnt/abka03/xlvlm_data/oneclass/oneimage \
  --output_root /mnt/abka03/xlvlm_data/oneclass/oneimage_crops \
  --patch_size 128 \
  --patches_per_image 50 \
  --max_overlap 0.5

"""
