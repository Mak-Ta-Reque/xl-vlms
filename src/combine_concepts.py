import os
import torch
import argparse

def main(input_dir, output_path):
    # List all .pth files in the directory
    pth_files = [f for f in os.listdir(input_dir) if f.endswith('.pth')]

    # Initialize the data structure
    new_decomose_data = {
        'concepts': [],
        'activations': [],
        'decomposition_method': [],
        'text_grounding': [],
        'image_grounding_paths': [],
        'analysis_model': [],
    }

    concepts = []
    activations = []
    text_grounding = []
    image_grounding_paths = []
    models = []

    for filename in pth_files:
        filepath = os.path.join(input_dir, filename)
        model_data = torch.load(filepath)

        image_grounding_path = model_data['image_grounding_paths']
        print("image_grounding_path", image_grounding_path)

        # Find index where all items do NOT start with "No "
        index_with_all_no = next(
            i for i, sublist in enumerate(image_grounding_path)
            if all(not item.startswith("No ") for item in sublist)
        )
        print("index_with_all_no", index_with_all_no)

        # Extract and append relevant data
        concept = model_data['concepts'][index_with_all_no]
        concepts.append(concept)

        activation = model_data['activations'][:, index_with_all_no]
        activations.append(activation)

        text_grounding.append(model_data['text_grounding'][index_with_all_no])
        image_grounding_paths.append(image_grounding_path[index_with_all_no])
        models.append(model_data['analysis_model'])

        # Assume decomposition method is the same for all
        new_decomose_data['decomposition_method'] = model_data['decomposition_method']

    concepts = torch.stack(concepts, dim=0)
    print("concepts shape", concepts.shape)

    # Update the combined data dictionary
    new_decomose_data['concepts'] = concepts
    new_decomose_data['activations'] = activations
    new_decomose_data['text_grounding'] = text_grounding
    new_decomose_data['image_grounding_paths'] = image_grounding_paths
    new_decomose_data['analysis_model'] = models

    # Save the combined data
    torch.save(new_decomose_data, output_path)
    print(f"Combined data saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine concept .pth files")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .pth files")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save combined .pth file")

    args = parser.parse_args()
    main(args.input_dir, args.output_path)
