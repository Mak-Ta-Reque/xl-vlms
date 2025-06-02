import os
import torch
import argparse
import numpy as np
from sklearn.preprocessing import normalize
from collections import Counter

def zca_whiten(X):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    U, S, _ = np.linalg.svd(cov_matrix)
    epsilon = 1e-5
    whitening_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    return np.dot(X_centered, whitening_matrix)


def combine_concepts(input_dir):
    pth_files = [f for f in os.listdir(input_dir) if f.endswith('.pth')]

    combined_data = {
        'concepts': [],
        'activations': [],
        'decomposition_method': None,
        'text_grounding': [],
        'image_grounding_paths': [],
        'analysis_model': [],
    }

    concepts = []
    activations = []
    for filename in pth_files:
        filepath = os.path.join(input_dir, filename)
        model_data = torch.load(filepath)

        image_grounding_path = model_data['image_grounding_paths']
        index_with_all_no  = max(
            range(len(image_grounding_path)),
        key=lambda i: sum(1 for item in image_grounding_path[i] if not item.startswith("Not"))
        )

        concepts.append(model_data['concepts'][index_with_all_no])
        activations.append(model_data['activations'][:, index_with_all_no])
        combined_data['text_grounding'].append(model_data['text_grounding'][index_with_all_no])
        combined_data['image_grounding_paths'].append(image_grounding_path[index_with_all_no])
        combined_data['analysis_model'].append(model_data['analysis_model'])

        combined_data['decomposition_method'] = model_data['decomposition_method']

    combined_data['concepts'] = torch.stack(concepts, dim=0)
    combined_data['activations'] = activations
    return combined_data


def combine_concepts_(input_dir):
    pth_files = [f for f in os.listdir(input_dir) if f.endswith('.pth')]

    # First pass: collect all strings from image_grounding_paths to compute global frequency
    global_counter = Counter()
    all_image_groundings = []

    for filename in pth_files:
        filepath = os.path.join(input_dir, filename)
        model_data = torch.load(filepath)
        image_grounding_path = model_data['image_grounding_paths']
        all_image_groundings.append(image_grounding_path)

        for grounding_list in image_grounding_path:
            global_counter.update(grounding_list)

    # Identify the most common string globally
    most_common_string, _ = global_counter.most_common(1)[0]

    # Second pass: extract data based on index with most occurrences of that string
    combined_data = {
        'concepts': [],
        'activations': [],
        'decomposition_method': None,
        'text_grounding': [],
        'image_grounding_paths': [],
        'analysis_model': [],
    }

    concepts = []
    activations = []
    for i, filename in enumerate(pth_files):
        filepath = os.path.join(input_dir, filename)
        model_data = torch.load(filepath)
        image_grounding_path = all_image_groundings[i]

        # Find the index where the most_common_string appears most frequently
        index_with_max_common = max(
            range(len(image_grounding_path)),
            key=lambda idx: image_grounding_path[idx].count(most_common_string)
        )

        concepts.append(model_data['concepts'][index_with_max_common])
        activations.append(model_data['activations'][:, index_with_max_common])
        combined_data['text_grounding'].append(model_data['text_grounding'][index_with_max_common])
        combined_data['image_grounding_paths'].append(image_grounding_path[index_with_max_common])
        combined_data['analysis_model'].append(model_data['analysis_model'])

        combined_data['decomposition_method'] = model_data['decomposition_method']

    combined_data['concepts'] = torch.stack(concepts, dim=0)
    combined_data['activations'] = activations

    return combined_data


def apply_normalization(concepts, method):
    concepts_np = concepts.cpu().numpy() if isinstance(concepts, torch.Tensor) else concepts

    if method == 'l2':
        return normalize(concepts_np, norm='l2')
    elif method == 'l1':
        return normalize(concepts_np, norm='l1')
    elif method == 'zca':
        return zca_whiten(concepts_np)
    
    elif method == 'l2zca':
        l2_normalized = normalize(concepts_np, norm='l2')
        return zca_whiten(l2_normalized)
    elif method == 'l1zca':
        l1_normalized = normalize(concepts_np, norm='l1')
        return zca_whiten(l1_normalized)
    else:
        raise ValueError("Unsupported normalization method: choose from 'l2', 'zca', or 'l2zca'")


def save_combined_data(data, output_path):
    torch.save(data, output_path)
    print(f"Saved combined data to: {output_path}")


def delete_original_files(paths):
    for f in paths:
        if f.endswith('.pth'):
            os.remove(f)
    print("Original .pth files deleted.")


def main(args):
    pth_files = [ os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.pth')] #original path files
    combined_data = combine_concepts(args.input_dir)

    base_output = args.output_path.rsplit('.', 1)[0]

    # Save raw combined concepts
    save_combined_data(combined_data, f"{base_output}_raw.pth")

    # Apply normalizations
    for method in ['l2', 'zca', 'l2zca', 'l1', 'l1zca']:
        if method in args.normalization:
            normalized_concepts = apply_normalization(combined_data['concepts'], method)
            data_copy = combined_data.copy()
            data_copy['concepts'] = torch.tensor(normalized_concepts, dtype=torch.float32)
            save_combined_data(data_copy, f"{base_output}_{method}.pth")
    
    if args.delete:
        delete_original_files(pth_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine and normalize concept .pth files")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with .pth files")
    parser.add_argument("--output_path", type=str, required=True, help="Base path to save output files")
    parser.add_argument("--normalization", nargs="+", choices=['l2', 'zca', 'l2zca', 'l1', 'l1zca'], required=True,
                        help="Normalization methods to apply")
    parser.add_argument("--delete", default=False, action="store_true", help="Delete input .pth files after processing")
    args = parser.parse_args()
    main(args)
