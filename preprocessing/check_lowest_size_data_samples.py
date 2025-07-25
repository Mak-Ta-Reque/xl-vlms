import os

val_dir = '/mnt/abka03/xlvlm_data/noidle_crops/train'  # change to your validation directory path

class_dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]

min_count = None
min_class = None

for class_name in class_dirs:
    class_path = os.path.join(val_dir, class_name)
    # Count only files (images) inside the class directory
    num_images = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
    
    if min_count is None or num_images < min_count:
        min_count = num_images
        min_class = class_name

print(f"Class with minimum data: '{min_class}' with {min_count} images")
