import os
import torch
import argparse
import numpy as np
from sklearn.preprocessing import normalize
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import inv
import random
def zca_whiten(X):

    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    U, S, _ = np.linalg.svd(cov_matrix)
    epsilon = 1e-5
    whitening_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    return np.dot(X_centered, whitening_matrix)



def dominant_positive_index(lists):
    def is_positive_majority(sublist):
        no_not = sum(
            item.lower().startswith('no_') or item.lower().startswith('not_') or item.lower().startswith('unk') or item.lower().startswith('thing') or item.lower().startswith('nc') 
            for item in sublist
        )
        positive = len(sublist) - no_not
        return positive > no_not

    indices = [idx for idx, sublist in enumerate(lists) if is_positive_majority(sublist)]
    return indices[0] if len(indices) == 1 else None


def count_conditioned_items(sublist):
    """
    Count items that start with the conditioned prefixes: no_, not_, unk, thing, nc (case-insensitive).
    """
    return sum(
        (s.lower().startswith('no_')
         or s.lower().startswith('no ')
         or s.lower().startswith('not_')
         or s.lower().startswith('unk')
         or s.lower().startswith('thing')
         or s.lower().startswith('nc'))
        for s in sublist
    )


def eligible_indices_by_threshold(lists):
    """
    Given a list of lists, return indices i where the number of conditioned items in lists[i]
    is strictly less than len(lists[0]) / 2, per user specification.

    Notes/assumptions:
    - Uses length of lists[0] as denominator as requested.
    - If lists is empty or lists[0] empty, returns [].
    """
    if not lists or not lists[0]:
        return []
    threshold = len(lists[0]) / 2.0
    return [i for i, sub in enumerate(lists) if count_conditioned_items(sub) < threshold]




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
    print(f"Loaded {len(pth_files)} .pth files from {input_dir}")
    for filename in pth_files:
        filepath = os.path.join(input_dir, filename)
        model_data = torch.load(filepath)
        image_grounding_path = model_data['image_grounding_paths']
        image_grounding_predictions = model_data.get('image_grounding_predictions', None)
        # Eligible indices: conditioned count < len(list[0]) / 2
        eligible = eligible_indices_by_threshold(image_grounding_predictions)

        # If none eligible, fallback to index with minimal conditioned count
        if not eligible:
            counts = [count_conditioned_items(sub) for sub in image_grounding_path]
            min_idx = int(np.argmin(counts)) if counts else 0
            eligible = [min_idx]

        # Append each eligible index
        for idx in eligible:
            concepts.append(model_data['concepts'][idx])
            activations.append(model_data['activations'][:, idx])
            combined_data['text_grounding'].append(model_data['text_grounding'][idx])
            combined_data['image_grounding_paths'].append(image_grounding_path[idx])
            combined_data['analysis_model'].append(model_data['analysis_model'])
            combined_data['image_grounding_predictions'] = model_data.get('image_grounding_predictions', [])[idx] if image_grounding_predictions else None
        combined_data['decomposition_method'] = model_data['decomposition_method']

    combined_data['concepts'] = torch.stack(concepts, dim=0)
    combined_data['activations'] = activations
    return combined_data



def laplacian_smoothing(X, alpha=0.5):
    """
    Perform unsupervised Laplacian smoothing on a matrix.
    
    Args:
        X (np.ndarray): Input matrix of shape (n_samples, n_features)
        alpha (float): Smoothing strength (0 < alpha <= 1)

    Returns:
        np.ndarray: Smoothed matrix of the same shape
    """

    #positive_threshold = 0.01
    #negative_threshold = -0.01

    # Set values to 0 if they are:
    # - less than the positive threshold but positive
    # - greater than the negative threshold but negative
    #X = np.where(((X > 0) & (X < positive_threshold)) | ((X < 0) & (X > negative_threshold)),0,  X   )
    # Step 1: Compute cosine similarity matrix
    W = cosine_similarity(X)
    np.fill_diagonal(W, 0)  # Remove self-similarity

    # Step 2: Compute Laplacian matrix L = D - W
    D = np.diag(W.sum(axis=1))
    L = D - W

    # Step 3: Solve smoothed matrix: X_new = (I + alpha * L)^(-1) @ X
    I = np.eye(W.shape[0])
    X_smoothed = inv(I + alpha * L) @ X

    return X_smoothed

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

        # Determine eligible indices per threshold rule
        eligible = eligible_indices_by_threshold(image_grounding_path)

        # Fallback: original selection based on most_common_string
        if not eligible:
            index_with_max_common = max(
                range(len(image_grounding_path)),
                key=lambda idx: image_grounding_path[idx].count(most_common_string)
            )
            eligible = [index_with_max_common]

        for idx in eligible:
            concepts.append(model_data['concepts'][idx])
            activations.append(model_data['activations'][:, idx])
            combined_data['text_grounding'].append(model_data['text_grounding'][idx])
            combined_data['image_grounding_paths'].append(image_grounding_path[idx])
            combined_data['analysis_model'].append(model_data['analysis_model'])

        combined_data['decomposition_method'] = model_data['decomposition_method']

    combined_data['concepts'] = torch.stack(concepts, dim=0)
    combined_data['activations'] = activations

    return combined_data


def apply_normalization(concepts, method):
    if concepts.shape[1] < 2:
        raise ValueError("Concepts must have at least 2 features for normalization.")
    concepts_np = concepts.cpu().numpy() if isinstance(concepts, torch.Tensor) else concepts

    positive_threshold = 0.01
    negative_threshold = -0.01

    # Set values to 0 if they are:
    # - less than the positive threshold but positive
    # - greater than the negative threshold but negative
    concepts_np = np.where((( concepts_np > 0) & ( concepts_np < positive_threshold)) | (( concepts_np < 0) & ( concepts_np > negative_threshold)),0,   concepts_np   )

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
    elif method == 'gl':
        concepts_np = zca_whiten(concepts_np)
        features = laplacian_smoothing(concepts_np)
        return normalize(features, norm='l2')
        
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
    for method in ['l2', 'zca', 'l2zca', 'l1', 'l1zca' , 'gl']:
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
    parser.add_argument("--normalization", nargs="+", choices=['l2', 'zca', 'l2zca', 'l1', 'l1zca', 'gl'], required=True,
                        help="Normalization methods to apply")
    parser.add_argument("--delete", default=False, action="store_true", help="Delete input .pth files after processing")
    args = parser.parse_args()
    main(args)
