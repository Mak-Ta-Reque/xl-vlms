import os
import json
import argparse
import random
import numpy as np
import torch
from PIL import Image
from lang_sam import LangSAM
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForQuestionAnswering

#Vqa patch processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
langsam_model = LangSAM()
def get_all_objects_vqa(image_pil):
    """Asks multiple questions to BLIP VQA to extract all possible objects in the image."""
    questions = [
        "What are the objects in the image?",
        "List all the things present.",
        "What items do you see?",
        "Describe everything in the image.",
        "What is inside this picture?"
    ]

    detected_objects = set()  # Store unique objects

    for question in questions:
        # Batch VQA input
        inputs = processor(image_pil, question, return_tensors="pt").to(device)

        with torch.no_grad():
            out = vqa_model.generate(**inputs, max_length=500)
            answer = processor.decode(out[0], skip_special_tokens=True)
            detected_objects.update(answer.split(","))  # Handle multiple objects

        # Clear GPU memory after each VQA step
        torch.cuda.empty_cache()  # Clear GPU memory after each query

    return list(detected_objects)  # Convert to list

def vqa_and_segment(image_pil, patch_size, num_patches):
    image_np = np.array(image_pil)  # Convert image to numpy for processing

    # Get VQA model's answers
    object_list = get_all_objects_vqa(image_pil)
    print("Detected objects:", object_list)

    # Initialize LangSAM for segmentation
    

    # Segment the image based on the VQA answers
    segmented_images = []
    for answer in object_list[:5]:  # Limit the number of objects (to avoid memory overload)
        results = langsam_model.predict([image_pil], [answer])[0]
        masks = results["masks"]
        labels = results["labels"]
        bboxes = results["boxes"]  # Bounding boxes in (x1, y1, x2, y2) format

        for i, (mask, label, bbox) in enumerate(zip(masks, labels, bboxes)):
            x1, y1, x2, y2 = map(int, bbox)  # Convert bounding box coordinates to integers

            # Convert mask to binary (0 or 255)
            mask_np = (mask * 255).astype(np.uint8)

            # Extract the bounding box region from the image and mask
            cropped_image_np = image_np[y1:y2, x1:x2]
            cropped_mask_np = mask_np[y1:y2, x1:x2]

            # Apply mask to keep only the segmented object
            segmented_np = np.zeros_like(cropped_image_np)
            for c in range(3):  # Apply mask to all RGB channels
                segmented_np[:, :, c] = cropped_image_np[:, :, c] * (cropped_mask_np // 255)

            segmented_pil = Image.fromarray(segmented_np)
            if patch_size != 0:
                segmented_pil = segmented_pil.resize((patch_size, patch_size))

            segmented_images.append((segmented_pil, label))

        torch.cuda.empty_cache()  # Clear GPU memory

    return segmented_images

# Function to create a directory if it doesn't exist
def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Function to read JSON file
def read_json_file(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

# Function to create patches randomly
def create_patche_random(image, patch_size, num_patches):
    patches = []
    img_width, img_height = image.size
    for _ in range(num_patches):
        left = random.randint(0, img_width - patch_size)
        upper = random.randint(0, img_height - patch_size)
        box = (left, upper, left + patch_size, upper + patch_size)
        patch = image.crop(box)
        patches.append((patch, "no label"))
    return patches

def create_patches_grid(image, patch_size, num_patches):
    patches = []
    img_width, img_height = image.size
    rows = int(num_patches ** 0.5)
    cols = int(num_patches / rows)
    
    for i in range(rows):
        for j in range(cols):
            left = int(i * img_width / rows)
            upper = int(j * img_height / cols)
            right = int((i + 1) * img_width / rows)
            lower = int((j + 1) * img_height / cols)
            
            patch = image.crop((left, upper, right, lower))
            patch_resized = patch.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
            patches.append((patch_resized, "no label"))
    
    return patches

# Function to process images and update JSON
def process_and_save_patches(images_data, root_image_folder, json_file_name, json_directory, patch_size, num_patches, technique):
    if technique == "grid":
        patch_processor = create_patches_grid
    elif  technique == "random":
        patch_processor = create_patche_random
    elif technique == "vqa-seg":
        patch_processor = vqa_and_segment
    else:
        raise Exception("method is not implemented")
    
    output_dir = os.path.join(json_directory, json_file_name.split('.')[0])
    create_directory(output_dir)
    train_folder = os.path.join(output_dir, 'train2014')
    val_folder = os.path.join(output_dir, 'val2014')
    
    create_directory(train_folder)
    create_directory(val_folder)

    updated_images_data = []
    for entry in images_data['images']:
        image_file = entry['filename']
        split = entry['split']
        subfolder = train_folder if split == 'train' else val_folder
        image_path = os.path.join(root_image_folder, entry['filepath'], image_file)

        try:
            img = Image.open(image_path).convert("RGB")
            patches = patch_processor(img, patch_size, num_patches)

            for i, (patch, label) in enumerate(patches):
                patch_filename = f"{image_file.split('.')[0]}_patch_{i}.png"
                patch.save(os.path.join(subfolder, patch_filename))

                updated_entry = entry.copy()
                updated_entry['filename'] = patch_filename
                updated_entry['patch'] = label
                updated_images_data.append(updated_entry)

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")

    return updated_images_data

# Function to process all JSON files in the directory
def process_json_files(json_directory, root_image_folder, patch_size, num_patches, technique):
    json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]

    for json_file in json_files:
        json_file_path = os.path.join(json_directory, json_file)
        print(f"Processing {json_file_path}...")

        images_data = read_json_file(json_file_path)
        updated_images_data = process_and_save_patches(
            images_data, root_image_folder, json_file, json_directory, patch_size, num_patches, technique
        )

        updated_json_file_path = os.path.join(json_directory, json_file.split('.')[0] + '_patch.json')
        with open(updated_json_file_path, 'w') as f:
            json.dump({'images': updated_images_data}, f, indent=4)

    print("Processing complete.")

def main():
    parser = argparse.ArgumentParser(description="Generate patches from images and update JSON.")
    parser.add_argument("json_directory", type=str, help="Path to the directory containing JSON files")
    parser.add_argument("root_image_folder", type=str, help="Root directory containing images")
    parser.add_argument("patch_size", type=int, help="Size of each image patch (e.g., 128 for 128x128)")
    parser.add_argument("num_patches", type=int, help="Number of patches per image")
    parser.add_argument("--technique", type=str, help="What patch creation technique should be used: grid, random, segmentation, vqa-seg", default="grid" )

    args = parser.parse_args()
    process_json_files(args.json_directory, args.root_image_folder, args.patch_size, args.num_patches, args.technique)

if __name__ == "__main__":
    main()
