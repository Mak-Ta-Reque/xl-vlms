import json
import random
import argparse
import os
import shutil

def load_classes_from_file(class_file):
    with open(class_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes

def sample_images_by_class(json_file, n_samples, class_file, output_dir, image_root_dir):
    # Load JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load class names
    classes = load_classes_from_file(class_file)

    for cls in classes:
        cls_key = cls.replace(" ", "_")

        # Check if enough samples exist in both splits
        def count_matches(split):
            return sum(
                1 for img in data["images"]
                if img.get("split") == split and
                any(cls.lower() in s.get("raw", "").lower() for s in img.get("sentences", []))
            )

        train_count = count_matches("train")
        val_count = count_matches("val")

        if train_count < n_samples or val_count < n_samples:
            print(f"[Skipped] Not enough samples for class '{cls}' (train={train_count}, val={val_count})")
            continue

        for split in ['train', 'val']:
            split_folder = f"{split}2014"
            target_dir = os.path.join(output_dir, split, cls_key)
            os.makedirs(target_dir, exist_ok=True)

            matched_images = [
                img for img in data["images"]
                if img.get("split") == split and
                any(cls.lower() in s.get("raw", "").lower() for s in img.get("sentences", []))
            ]
            sampled_images = random.sample(matched_images, n_samples)

            # Save sampled JSON
            sampled_data = {"images": sampled_images}
            json_output_path = os.path.join(target_dir, f"{cls_key}_dataset_coco.json")
            with open(json_output_path, 'w', encoding='utf-8') as f_out:
                json.dump(sampled_data, f_out, indent=4)

            # Copy images
            for img in sampled_images:
                img_filename = img.get("filename") or img.get("filepath")
                if img_filename:
                    src_path = os.path.join(image_root_dir, split_folder, img_filename)
                    dst_path = os.path.join(target_dir, os.path.basename(img_filename))
                    if os.path.exists(src_path):
                        shutil.copy(src_path, dst_path)
                    else:
                        print(f"[Warning] Missing: {src_path}")

            print(f"[{split}] Saved {n_samples} samples for class '{cls}' in {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample COCO JSON images by class names from a file.')
    parser.add_argument('json_file', type=str, help='Input COCO-style JSON file')
    parser.add_argument('n_samples', type=int, help='Samples per class per split')
    parser.add_argument('class_file', type=str, help='Text file with class names (one per line)')
    parser.add_argument('output_dir', type=str, help='Root directory for output')
    parser.add_argument('image_root_dir', type=str, help='Directory with train2014/ and val2014/ subfolders')
    args = parser.parse_args()

    sample_images_by_class(
        json_file=args.json_file,
        n_samples=args.n_samples,
        class_file=args.class_file,
        output_dir=args.output_dir,
        image_root_dir=args.image_root_dir
    )
#python preprocessing/coco_sampling.py /mnt/abka03/mscoco2014/dataset_coco.json 50 classes.txt /mnt/abka03/xlvlm_data/coco_100_samples /mnt/abka03/mscoco2014
