import json
import random
import argparse
import os
import shutil

def sample_images_by_class(json_file, n_samples, classes, output_dir, image_root_dir):
    # Load JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for cls in classes:
        cls_key = cls.replace(" ", "_")

        for split in ['train', 'val']:
            # Get subfolder (train2014/val2014)
            split_folder = f"{split}2014"

            # Create output directory: output_dir/train/class_name/ or output_dir/val/class_name/
            target_dir = os.path.join(output_dir, split, cls_key)
            os.makedirs(target_dir, exist_ok=True)

            # Filter images for this class and split
            def matches_class(image):
                return (
                    image.get("split") == split and
                    any(cls.lower() in sentence.get("raw", "").lower() for sentence in image.get("sentences", []))
                )

            matched_images = [img for img in data["images"] if matches_class(img)]
            sampled_images = random.sample(matched_images, min(n_samples, len(matched_images)))

            # Save sampled JSON
            sampled_data = {"images": sampled_images}
            json_output_path = os.path.join(target_dir, f"{cls_key}_dataset_coco.json")
            with open(json_output_path, 'w', encoding='utf-8') as f_out:
                json.dump(sampled_data, f_out, indent=4)

            # Copy images from image_root_dir/split_folder/
            for img in sampled_images:
                img_filename = img.get("filename") or img.get("filepath")
                if img_filename:
                    src_path = os.path.join(image_root_dir, split_folder, img_filename)
                    dst_path = os.path.join(target_dir, os.path.basename(img_filename))
                    if os.path.exists(src_path):
                        shutil.copy(src_path, dst_path)
                    else:
                        print(f"[Warning] Missing: {src_path}")

            print(f"[{split}] Saved {len(sampled_images)} for class '{cls}' in {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample COCO JSON images by class name.')
    parser.add_argument('json_file', type=str, help='Input COCO-style JSON file')
    parser.add_argument('n_samples', type=int, help='Samples per class per split')
    parser.add_argument('classes', type=str, nargs='+', help='Class names (e.g., "hot dog")')
    parser.add_argument('output_dir', type=str, help='Root directory for output')
    parser.add_argument('image_root_dir', type=str, help='Directory with train2014/ and val2014/ subfolders')

    args = parser.parse_args()

    sample_images_by_class(
        json_file=args.json_file,
        n_samples=args.n_samples,
        classes=args.classes,
        output_dir=args.output_dir,
        image_root_dir=args.image_root_dir
    )
#python preprocessing/coco_sampleing.py /mnt/abka03/mscoco2014/dataset_coco.json  50  "dog" "hot dog" "bus" "school bus" "teddy bear" "microwave oven"  "fire hydrant" "traffic light" "baseball glove" "train" "cat" "bear" "baby" "car" "stop sign" /mnt/abka03/xlvlm_data/coco_100_samples /mnt/abka03/mscoco2014