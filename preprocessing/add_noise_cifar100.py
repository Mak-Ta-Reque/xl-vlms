import os
import random
import shutil

DATASET_DIR = '/mnt/abka03/xlvlm_data/cifar_100_samples/train'  # change this
NOISY_SUFFIX = '_noisy'  # to identify noisy images

# Step 1: Get class directories
class_dirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

# Step 2: Process each class
for class_name in class_dirs:
    class_path = os.path.join(DATASET_DIR, class_name)
    true_images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

    num_to_add = len(true_images)
    print(f'Processing {class_name}: {num_to_add} images')

    # Step 3: Collect candidate noisy images from other classes
    other_classes = [c for c in class_dirs if c != class_name]
    other_images = []

    for other_class in other_classes:
        other_path = os.path.join(DATASET_DIR, other_class)
        for f in os.listdir(other_path):
            other_images.append((os.path.join(other_path, f), f"{other_class}_{f}"))

    # Step 4: Randomly select noisy images
    noisy_samples = random.sample(other_images, num_to_add)

    # Step 5: Copy noisy images to current class
    for src_path, noisy_filename in noisy_samples:
        dest_path = os.path.join(class_path, NOISY_SUFFIX + '_' + noisy_filename)
        shutil.copy(src_path, dest_path)

print("Noisy samples added.")
