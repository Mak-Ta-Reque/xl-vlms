import os
import argparse
import random
import numpy as np
from PIL import Image

def generate_noise_images(n_samples, classes, output_dir, image_size=(224, 224)):
    for cls in classes:
        cls_key = cls.replace(" ", "_")
        
        for split in ['train', 'val']:
            target_dir = os.path.join(output_dir, split, cls_key)
            os.makedirs(target_dir, exist_ok=True)

            for i in range(n_samples):
                filename = f"{cls_key}_{split}_{i:04d}.jpg"
                dst_path = os.path.join(target_dir, filename)

                # Generate random RGB noise image
                noise = np.random.randint(0, 256, (image_size[1], image_size[0], 3), dtype=np.uint8)
                noise_img = Image.fromarray(noise, 'RGB')
                noise_img.save(dst_path)

            print(f"[{split}] Generated {n_samples} noise images for class '{cls}' in {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random noise images for given classes.')
    parser.add_argument('n_samples', type=int, help='Number of samples per class per split')
    parser.add_argument('classes', type=str, nargs='+', help='Class names (e.g., "dog", "hot dog")')
    parser.add_argument('output_dir', type=str, help='Root directory for output')

    args = parser.parse_args()

    # Optional: For reproducibility
    random.seed(42)
    np.random.seed(42)

    generate_noise_images(
        n_samples=args.n_samples,
        classes=args.classes,
        output_dir=args.output_dir
    )


#python preprocessing/nose_image_sampling.py 50 "dog" "hot dog" "bus" "school bus" "teddy bear" "microwave oven" "fire hydrant" "traffic light" "baseball glove" "train" "cat" "bear" "baby" "car" "stop sign"  /mnt/abka03/xlvlm_data/noise_image
