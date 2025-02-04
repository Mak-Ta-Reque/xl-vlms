import json
import random
import argparse
import os

def sample_coco_json(json_file, n_train, n_val, tokens, output_dir):
    # Load JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure tokens is a list
    if isinstance(tokens, str):
        tokens = [tokens]

    # Function to check if any token in the list is in the 'raw' field
    def contains_any_tok_in_raw(image):
        for sentence in image.get("sentences", []):
            if any(tok in sentence.get("raw", "") for tok in tokens):  # Check if any token is in 'raw'
                return True
        return False

    # Extract and filter train and val splits based on raw field containing any token
    train_images = [img for img in data["images"] if img.get("split") == "train" and contains_any_tok_in_raw(img)]
    val_images = [img for img in data["images"] if img.get("split") == "val" and contains_any_tok_in_raw(img)]
    
    # Sample data while ensuring we don't exceed available count
    sampled_train = random.sample(train_images, min(n_train, len(train_images)))
    sampled_val = random.sample(val_images, min(n_val, len(val_images)))
    
    # Combine sampled data
    sampled_images = sampled_train + sampled_val

    # Create new JSON structure
    sampled_data = {"images": sampled_images}

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Output file path - use "combined" if multiple tokens were provided
    output_filename = "combined_dataset_coco.json" if len(tokens) > 1 else f'{tokens[0].replace(" ", "_")}_dataset_coco.json'
    output_file = os.path.join(output_dir, output_filename)

    # Save sampled data to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=4)

    print(f"Sampled dataset saved to {output_file}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Sample a COCO JSON dataset.')
    parser.add_argument('json_file', type=str, help='Input JSON file')
    parser.add_argument('n_train', type=int, help='Number of train samples to select')
    parser.add_argument('n_val', type=int, help='Number of validation samples to select')
    parser.add_argument('tokens', type=str, nargs='+', help='Tokens to search for in the "raw" field (space-separated for multiple)')
    parser.add_argument('output_dir', type=str, help='Directory to save the output JSON file')

    # Parse arguments
    args = parser.parse_args()

    # Call the function with arguments from the command line
    sample_coco_json(args.json_file, args.n_train, args.n_val, args.tokens, args.output_dir)
