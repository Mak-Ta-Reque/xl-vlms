import os
import torch
import argparse
import numpy as np
from sklearn.preprocessing import normalize


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
        key=lambda i: sum(1 for item in image_grounding_path[i] if not item.startswith("No"))
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


def apply_normalization(concepts, method):
    concepts_np = concepts.cpu().numpy() if isinstance(concepts, torch.Tensor) else concepts

    if method == 'l2':
        return normalize(concepts_np, norm='l2')
    elif method == 'zca':
        return zca_whiten(concepts_np)
    elif method == 'l2zca':
        l2_normalized = normalize(concepts_np, norm='l2')
        return zca_whiten(l2_normalized)
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
    for method in ['l2', 'zca', 'l2zca']:
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
    parser.add_argument("--normalization", nargs="+", choices=['l2', 'zca', 'l2zca'], required=True,
                        help="Normalization methods to apply")
    parser.add_argument("--delete", default=True, action="store_true", help="Delete input .pth files after processing")
    args = parser.parse_args()
    main(args)
