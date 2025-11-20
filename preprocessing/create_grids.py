#!/usr/bin/env python3
"""
Create random image grids by sampling one image from each of N subdirectories.

Example usage:
    python create_grids.py \
        --input_dir "/mnt/abka03/xlvlm_data/imagenet10class/val" \
        --output_dir "/mnt/abka03/xlvlm_data/imagenet10class/val_grids" \
        --n 4 \
        --num_grids 50 \
        --image_size 512
"""

import os
import random
import math
import argparse
from PIL import Image


def create_image_grids(input_dir, output_dir, n=4, num_grids=1, image_size=256):
    """
    Create multiple grid images by randomly sampling 1 image from each subdirectory.
    Output filename encodes the subdirectory class names in grid order.
    """
    grid_size = int(math.sqrt(n))
    if grid_size * grid_size != n:
        raise ValueError("n must be a perfect square (4, 16, 36, ...)")

    subdirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) 
               if os.path.isdir(os.path.join(input_dir, d))]

    if len(subdirs) < n:
        raise ValueError(f"Not enough subdirectories. Found {len(subdirs)}, need {n}.")

    os.makedirs(output_dir, exist_ok=True)

    for grid_idx in range(num_grids):
        chosen_subdirs = random.sample(subdirs, n)

        images, class_names = [], []
        for subdir in chosen_subdirs:
            class_name = os.path.basename(subdir)
            class_names.append(class_name)

            files = [f for f in os.listdir(subdir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            if not files:
                raise ValueError(f"No images found in {subdir}")

            img_path = os.path.join(subdir, random.choice(files))
            img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
            images.append(img)

        grid_img = Image.new("RGB", (grid_size * image_size, grid_size * image_size))
        for idx, img in enumerate(images):
            row, col = divmod(idx, grid_size)
            grid_img.paste(img, (col * image_size, row * image_size))

        class_part = "__".join(class_names)
        if len(class_part) > 150:
            class_part = class_part[:150]

        output_name = f"grid_{n}_{grid_idx+1}__{class_part}.jpg"
        output_path = os.path.join(output_dir, output_name)
        grid_img.save(output_path)

        print(f"Saved grid {grid_idx+1}/{num_grids} at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create random image grids.")
    parser.add_argument("--input_dir", required=True, help="Input directory with subdirectories (classes).")
    parser.add_argument("--output_dir", required=True, help="Directory where output grids will be saved.")
    parser.add_argument("--n", type=int, default=4, help="Number of images per grid (must be a perfect square).")
    parser.add_argument("--num_grids", type=int, default=40, help="How many grids to generate.")
    parser.add_argument("--image_size", type=int, default=256, help="Resize each image to this size (square).")

    args = parser.parse_args()

    create_image_grids(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n=args.n,
        num_grids=args.num_grids,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
