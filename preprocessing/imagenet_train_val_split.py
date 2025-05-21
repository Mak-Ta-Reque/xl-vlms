import os
import tarfile
import shutil

# Configuration
val_tar_path = "/mnt/abka03/raw_data/ILSVRC2012_img_val.tar"  # Single val tar file
label_file = "/mnt/abka03/Projects/xl-vlms/preprocessing/imagenet_names.txt"  # Mapping file
extract_dir = "/mnt/abka03/raw_dataa/imagenet21_val_raw"   # Temp extraction dir
val_output_dir = "/mnt/abka03/processed_data/imagenet21/val"    # Final output directory

# Step 1: Load synset ID → class name mapping
synset_to_name = {}
with open(label_file, "r") as f:
    for line in f:
        parts = line.strip().split(maxsplit=2)
        if len(parts) == 3:
            synset_id = parts[0]
            class_name = parts[2].strip().lower().replace(" ", "_")
            synset_to_name[synset_id] = class_name

# Step 2: Extract val.tar to a temporary directory
os.makedirs(extract_dir, exist_ok=True)
with tarfile.open(val_tar_path) as tar:
    tar.extractall(path=extract_dir)
print(f"✔️ Extracted val.tar to {extract_dir}")

# Step 3: Organize images into class-named folders in val_output_dir
os.makedirs(val_output_dir, exist_ok=True)

for filename in os.listdir(extract_dir):
    if not filename.endswith(".JPEG"):
        continue

    # Extract synset ID (ImageNet val images are typically named like n01440764_12345.JPEG)
    synset_id = filename.split("_")[0]
    class_name = synset_to_name.get(synset_id)

    if not class_name:
        print(f"⚠️ Skipping {filename}: unknown synset ID {synset_id}")
        continue

    class_dir = os.path.join(val_output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    src_path = os.path.join(extract_dir, filename)
    dst_path = os.path.join(class_dir, filename)
    shutil.move(src_path, dst_path)

print("🎉 Done: Validation data organized into class-named folders in 'val/'")
