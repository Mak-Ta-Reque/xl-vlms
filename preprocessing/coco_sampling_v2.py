import json
import random
import argparse
import os
import shutil
import re

def load_classes_from_file(class_file):
    with open(class_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def match_class_in_sentences(cls, sentences):
    pattern = re.compile(rf"\b{re.escape(cls.lower())}\b")
    return any(pattern.search(s.get("raw", "").lower()) for s in sentences)

def filter_images_by_class(data, cls, split):
    return [
        img for img in data["images"]
        if img.get("split") == split and match_class_in_sentences(cls, img.get("sentences", []))
    ]

def sample_and_save_images(images, n_samples, target_dir, json_filename, split_folder, image_root_dir):
    sampled_images = random.sample(images, n_samples)
    os.makedirs(target_dir, exist_ok=True)

    # Save JSON
    with open(os.path.join(target_dir, json_filename), 'w', encoding='utf-8') as f_out:
        json.dump({"images": sampled_images}, f_out, indent=4)

    # Copy images
    for img in sampled_images:
        img_filename = img.get("filename") or img.get("filepath")
        if img_filename:
            src = os.path.join(image_root_dir, split_folder, img_filename)
            dst = os.path.join(target_dir, os.path.basename(img_filename))
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"[Warning] Missing file: {src}")

    print(f"[{split_folder}] Saved {n_samples} samples to {target_dir}")

def sample_images_by_class(json_file, train_samples, val_samples, class_file, output_dir, image_root_dir):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    classes = load_classes_from_file(class_file)

    for cls in classes:
        cls_key = cls.replace(" ", "_")

        train_images = filter_images_by_class(data, cls, "train")
        val_images = filter_images_by_class(data, cls, "val")

        if len(train_images) < train_samples or len(val_images) < val_samples:
            print(f"[Skipped] Not enough samples for '{cls}' (train={len(train_images)}, val={len(val_images)})")
            continue

        sample_and_save_images(
            train_images, train_samples,
            os.path.join(output_dir, "train", cls_key),
            f"{cls_key}_train_dataset.json", "train2014", image_root_dir
        )

        sample_and_save_images(
            val_images, val_samples,
            os.path.join(output_dir, "val", cls_key),
            f"{cls_key}_val_dataset.json", "val2014", image_root_dir
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample COCO JSON images by class names with split ratio.')
    parser.add_argument('json_file', type=str, help='Path to input COCO JSON')
    parser.add_argument('class_file', type=str, help='File with class names (one per line)')
    parser.add_argument('output_dir', type=str, help='Directory to save output samples')
    parser.add_argument('image_root_dir', type=str, help='Root with train2014/ and val2014/')
    parser.add_argument('--train_samples', type=int, default=80, help='Number of training samples per class')
    parser.add_argument('--val_samples', type=int, default=20, help='Number of validation samples per class')
    args = parser.parse_args()

    sample_images_by_class(
        json_file=args.json_file,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        class_file=args.class_file,
        output_dir=args.output_dir,
        image_root_dir=args.image_root_dir
    )

#python preprocessing/coco_sampling.py /mnt/abka03/mscoco2014/dataset_coco.json classes.txt /mnt/abka03/xlvlm_data/coco_100_samples /mnt/abka03/mscoco2014 --train_samples 200 --val_samples 50
