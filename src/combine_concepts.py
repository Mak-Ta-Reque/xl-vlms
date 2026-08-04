import os
import torch
import argparse
import numpy as np
from sklearn.preprocessing import normalize
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import inv
import random
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=project_root / '.env', override=False)
def zca_whiten(X):

    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    U, S, _ = np.linalg.svd(cov_matrix)
    epsilon = 1e-5
    whitening_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    return np.dot(X_centered, whitening_matrix)



def dominant_positive_index(lists):
    def is_positive_majority(sublist):
        no_not = sum(is_negative_prediction(item) for item in sublist)
        positive = len(sublist) - no_not
        return positive > no_not

    indices = [idx for idx, sublist in enumerate(lists) if is_positive_majority(sublist)]
    return indices[0] if len(indices) == 1 else None


# Sentence-ending punctuation/quotes VLMs commonly wrap short yes/no answers
# in (e.g. "No.", "no!", '"no"') — stripped before matching so yn-mode
# predictions aren't silently miscounted as positive.
_NEGATIVE_STRIP_CHARS = ".,!?;:'\")( \t\n\r"


def is_negative_prediction(s):
    """
    Return True if a prediction string is a negative/unknown outcome.
    Matches: exact 'no', or prefixes 'no_', 'not_', 'unk', 'thing', 'nc' (case-insensitive).
    """
    low = str(s).lower().strip()
    low = low.strip(_NEGATIVE_STRIP_CHARS)
    return (
        low == 'no'
        or low.startswith('no_')
        or low.startswith('not_')
        or low.startswith('unk')
        or low.startswith('thing')
        or low.startswith('nc')
        or low.startswith('no ')
    )


def has_all_positive_predictions(predictions):
    """
    Return True only if ALL predictions in the list are positive
    (none match the negative conditions).
    """
    if not predictions:
        return False
    return not any(is_negative_prediction(s) for s in predictions)


def count_conditioned_items(sublist):
    """
    Count items that match the negative/unknown conditions (case-insensitive).
    """
    return sum(is_negative_prediction(s) for s in sublist)


def parse_clean_example_ratio(default=0.8):
    """
    Read CLEAN_EXAMPLE_RATIO from environment and validate range [0, 1].
    """
    raw = os.getenv('CLEAN_EXAMPLE_RATIO', str(default))
    try:
        ratio = float(raw)
    except (TypeError, ValueError):
        print(f"WARNING: Invalid CLEAN_EXAMPLE_RATIO='{raw}'. Using default {default}.")
        return default

    if ratio < 0.0 or ratio > 1.0:
        print(f"WARNING: CLEAN_EXAMPLE_RATIO={ratio} out of range [0, 1]. Using default {default}.")
        return default

    return ratio


def eligible_indices_by_threshold(lists, clean_example_ratio=0.8):
    """
    Given a list of lists of predictions, return indices where the proportion
    of positive predictions is >= clean_example_ratio.

    A prediction is positive when it does not match negative conditions.
    Returns list of eligible indices (can be empty if no concept qualifies).
    """
    if not lists:
        return []

    eligible = []
    for i, preds in enumerate(lists):
        if not preds:
            continue
        positive_count = sum(not is_negative_prediction(p) for p in preds)
        positive_ratio = positive_count / len(preds)
        if positive_ratio >= clean_example_ratio:
            eligible.append(i)

    return eligible




def _empty_combined_dict():
    return {
        'concepts': [],
        'concept_names': [],
        'activations': [],
        'decomposition_method': None,
        'text_grounding': [],
        'image_grounding_paths': [],
        'image_grounding_bboxes': [],
        'image_grounding_masks': [],
        'analysis_model': [],
        'image_grounding_predictions': [],
        # Per-concept source layer (differs per tag with logit-lens selection)
        'selected_layer': [],
        # Per-concept positive ratio (assigned-samples based), the size of the
        # assigned set it was computed over, and the threshold used for the
        # split — kept for later evaluation.
        'direction_positive_ratio': [],
        'direction_num_assigned': [],
        'direction_weighted_ratio': [],
        'clean_example_ratio': None,
    }


def _append_concept(store, concepts_list, activations_list, model_data, idx,
                    image_grounding_path, image_grounding_predictions):
    concepts_list.append(model_data['concepts'][idx])
    # image_grounding_activations[idx] (added in multimodal_grounding.py) is
    # index-aligned with image_grounding_path[idx]: concept_image_grounding()
    # now hard-assigns each sample to exactly one concept (argmax direction),
    # so it's no longer a plain prefix of the full sorted column — re-deriving
    # it from model_data['activations'][:, idx] would silently pick the wrong
    # (unassigned) samples. Fall back to the old re-derivation only for
    # legacy files that predate this field.
    aligned_activations = model_data.get('image_grounding_activations', None)
    if isinstance(aligned_activations, list) and idx < len(aligned_activations):
        concept_activations = aligned_activations[idx]
    else:
        concept_activations = torch.sort(model_data['activations'][:, idx], descending=True).values
        concept_activations = concept_activations[: len(image_grounding_path[idx])]
    activations_list.append(concept_activations)
    store['text_grounding'].append(model_data['text_grounding'][idx])
    store['image_grounding_paths'].append(image_grounding_path[idx])
    store['concept_names'].append(model_data['image_concept_names'][idx])
    store['analysis_model'].append(model_data['analysis_model'])
    store['image_grounding_predictions'].append(
        model_data.get('image_grounding_predictions', [])[idx] if image_grounding_predictions else None
    )
    store['image_grounding_bboxes'].append(
        model_data.get('image_grounding_bboxes', [])[idx]
        if idx < len(model_data.get('image_grounding_bboxes', [])) else None
    )
    ig_masks = model_data.get('image_grounding_masks', [])
    store['image_grounding_masks'].append(ig_masks[idx] if idx < len(ig_masks) else None)
    store['selected_layer'].append(model_data.get('selected_layer', None))
    ratios = model_data.get('direction_positive_ratio', None)
    store['direction_positive_ratio'].append(
        ratios[idx] if isinstance(ratios, list) and idx < len(ratios) else None
    )
    counts = model_data.get('direction_num_assigned', None)
    store['direction_num_assigned'].append(
        counts[idx] if isinstance(counts, list) and idx < len(counts) else None
    )
    weighted = model_data.get('direction_weighted_ratio', None)
    store['direction_weighted_ratio'].append(
        weighted[idx] if isinstance(weighted, list) and idx < len(weighted) else None
    )
    store['decomposition_method'] = model_data['decomposition_method']


def combine_concepts(input_dir, clean_example_ratio=0.8):
    """Split concepts into a positive bank (passed the purity filter) and a
    negative bank (failed it). Returns (combined_data, negative_data)."""
    pth_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.pth')])

    combined_data = _empty_combined_dict()
    negative_data = _empty_combined_dict()

    concepts = []
    activations = []
    negative_concepts = []
    negative_activations = []

    # ── Tracking stats ──
    total_files = len(pth_files)
    total_concepts_seen = 0
    total_concepts_included = 0
    total_concepts_skipped = 0
    files_fully_skipped = 0

    print(f"\n{'='*60}")
    print(f"Combine Concepts: loaded {total_files} .pth files from {input_dir}")
    print(f"{'='*60}")

    for filename in pth_files:
        filepath = os.path.join(input_dir, filename)
        model_data = torch.load(filepath)
        image_grounding_path = model_data['image_grounding_paths']
        image_grounding_predictions = model_data.get('image_grounding_predictions', None)
        direction_ratios = model_data.get('direction_positive_ratio', None)
        direction_weighted = model_data.get('direction_weighted_ratio', None)
        direction_counts = model_data.get('direction_num_assigned', None)

        n_concepts_in_file = len(image_grounding_predictions) if image_grounding_predictions else 0
        total_concepts_seen += n_concepts_in_file

        # Selection: the per-direction positive ratio must be
        # >= CLEAN_EXAMPLE_RATIO. DIRECTION_RATIO_MODE picks the metric:
        #   assigned (default) — positives among argmax-assigned / #assigned
        #   weighted           — positive activation mass / total mass
        #   both               — must pass the threshold on BOTH metrics
        # MIN_DIRECTION_ASSIGNED additionally requires a minimum number of
        # assigned samples (guards against tiny high-variance sets).
        # Falls back to the grounding-prediction ratio for older files
        # without direction_positive_ratio.
        ratio_mode = os.getenv('DIRECTION_RATIO_MODE', 'assigned').strip().lower()
        min_assigned = int(os.getenv('MIN_DIRECTION_ASSIGNED', '1'))
        if isinstance(direction_ratios, list) and len(direction_ratios) == n_concepts_in_file:
            def _passes(idx):
                r_assigned = direction_ratios[idx]
                r_weighted = (
                    direction_weighted[idx]
                    if isinstance(direction_weighted, list) and idx < len(direction_weighted)
                    else None
                )
                n_owned = (
                    direction_counts[idx]
                    if isinstance(direction_counts, list) and idx < len(direction_counts)
                    else None
                )
                if isinstance(n_owned, int) and n_owned < min_assigned:
                    return False
                ok_assigned = r_assigned is not None and r_assigned >= clean_example_ratio
                ok_weighted = r_weighted is not None and r_weighted >= clean_example_ratio
                if ratio_mode == 'weighted':
                    return ok_weighted
                if ratio_mode == 'both':
                    return ok_assigned and ok_weighted
                return ok_assigned
            eligible = [idx for idx in range(n_concepts_in_file) if _passes(idx)]
            ratio_source = f'direction-{ratio_mode}' + (f', min_assigned={min_assigned}' if min_assigned > 1 else '')
        else:
            eligible = eligible_indices_by_threshold(image_grounding_predictions, clean_example_ratio)
            ratio_source = 'grounding-predictions (legacy fallback)'
        n_skipped = n_concepts_in_file - len(eligible)

        # Log per-file details
        print(f"\n  File: {filename}  ({n_concepts_in_file} concepts, ratio source: {ratio_source})")
        for c_idx in range(n_concepts_in_file):
            preds = image_grounding_predictions[c_idx] if image_grounding_predictions else []
            neg_items = [p for p in preds if is_negative_prediction(p)]
            if isinstance(direction_ratios, list) and c_idx < len(direction_ratios) and direction_ratios[c_idx] is not None:
                pos_ratio = direction_ratios[c_idx]
            elif preds:
                pos_ratio = sum(not is_negative_prediction(p) for p in preds) / len(preds)
            else:
                pos_ratio = 0.0
            extra = ""
            if isinstance(direction_counts, list) and c_idx < len(direction_counts) and direction_counts[c_idx] is not None:
                extra += f", assigned={direction_counts[c_idx]}"
            if isinstance(direction_weighted, list) and c_idx < len(direction_weighted) and direction_weighted[c_idx] is not None:
                extra += f", weighted_ratio={direction_weighted[c_idx]:.2f}"
            status = "POSITIVE" if c_idx in eligible else "NEGATIVE"
            if neg_items:
                print(f"    concept {c_idx}: {status} -- positive_ratio={pos_ratio:.2f}{extra}, negative predictions: {neg_items[:10]}")
            else:
                print(f"    concept {c_idx}: {status} -- positive_ratio={pos_ratio:.2f}{extra}")

        total_concepts_skipped += n_skipped

        # Every concept goes to one of the two banks: eligible -> positive,
        # below-threshold -> negative (kept, not discarded).
        for idx in range(n_concepts_in_file):
            if idx in eligible:
                _append_concept(combined_data, concepts, activations, model_data,
                                idx, image_grounding_path, image_grounding_predictions)
            else:
                _append_concept(negative_data, negative_concepts, negative_activations,
                                model_data, idx, image_grounding_path, image_grounding_predictions)

        if len(eligible) < 1:
            files_fully_skipped += 1
            print(f"    >> FILE HAS NO ELIGIBLE CONCEPT (0/{n_concepts_in_file}) — all kept in negative bank")
            continue

        total_concepts_included += len(eligible)
        print(f"    >> {len(eligible)}/{n_concepts_in_file} concepts eligible")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"COMBINE CONCEPTS SUMMARY")
    print(f"  Files processed:        {total_files}")
    print(f"  Files w/o eligible:     {files_fully_skipped}")
    print(f"  Total concepts seen:    {total_concepts_seen}")
    print(f"  Positive bank:          {total_concepts_included}")
    print(f"  Negative bank:          {total_concepts_skipped}  (below positive-ratio threshold {clean_example_ratio:.2f})")
    if total_concepts_seen > 0:
        pct = 100 * total_concepts_skipped / total_concepts_seen
        print(f"  Negative rate:          {pct:.1f}%")
    print(f"{'='*60}\n")

    if not concepts:
        print(f"WARNING: No concepts met CLEAN_EXAMPLE_RATIO threshold ({clean_example_ratio:.2f}) in any file.")
        combined_data['concepts'] = torch.empty(0)
        combined_data['activations'] = []
    else:
        combined_data['concepts'] = torch.stack(concepts, dim=0)
        combined_data['activations'] = activations

    if not negative_concepts:
        negative_data['concepts'] = torch.empty(0)
        negative_data['activations'] = []
    else:
        negative_data['concepts'] = torch.stack(negative_concepts, dim=0)
        negative_data['activations'] = negative_activations

    combined_data['clean_example_ratio'] = clean_example_ratio
    negative_data['clean_example_ratio'] = clean_example_ratio

    return combined_data, negative_data



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

def combine_concepts_(input_dir, clean_example_ratio=0.8):
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
        eligible = eligible_indices_by_threshold(image_grounding_path, clean_example_ratio)

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
    if isinstance(concepts, torch.Tensor):
        if concepts.numel() == 0:
            return concepts
        if concepts.ndim < 2:
            return concepts
        feature_dim = concepts.shape[1]
    else:
        concepts = np.asarray(concepts)
        if concepts.size == 0:
            return concepts
        if concepts.ndim < 2:
            return concepts
        feature_dim = concepts.shape[1]

    if feature_dim < 2:
        return concepts
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
        #concepts_np = zca_whiten(concepts_np)
        features = laplacian_smoothing(concepts_np)
        return normalize(features, norm='l2')
        
    else:
        raise ValueError("Unsupported normalization method: choose from 'l2', 'zca', or 'l2zca'")


def save_combined_data(data, output_path):
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    torch.save(data, output_path)
    print(f"Saved combined data to: {output_path}")


def delete_original_files(paths):
    for f in paths:
        if f.endswith('.pth'):
            os.remove(f)
    print("Original .pth files deleted.")


def main(args):
    """
    Main entry point for combining and normalizing concept .pth files.
    
    Args:
        args: Namespace object with input_dir, output_path, normalization, and delete attributes
    """
    pth_files = [
        os.path.join(args.input_dir, f) 
        for f in os.listdir(args.input_dir) 
        if f.endswith('.pth')
    ]
    
    clean_example_ratio = parse_clean_example_ratio(default=0.8)
    print(f"Using CLEAN_EXAMPLE_RATIO={clean_example_ratio:.2f}")

    combined_data, negative_data = combine_concepts(args.input_dir, clean_example_ratio=clean_example_ratio)
    # Embed the selection threshold in the file names so different ratios can
    # be produced and evaluated side by side, e.g.
    # combined_concept_snmf_cr0.5_raw.pth
    ratio_tag = f"cr{clean_example_ratio:g}"
    base_output = args.output_path.rsplit('.', 1)[0]
    if not base_output.endswith(f"_{ratio_tag}"):
        base_output = f"{base_output}_{ratio_tag}"
    # Negative bank mirrors the positive naming:
    # combined_concept_snmf* -> combined_negative_concept_snmf*
    if 'combined_concept' in os.path.basename(base_output):
        neg_base = os.path.join(
            os.path.dirname(base_output),
            os.path.basename(base_output).replace('combined_concept', 'combined_negative_concept', 1),
        )
    else:
        neg_base = f"{base_output}_negative"

    # Save raw combined concepts (positive + negative banks)
    save_combined_data(combined_data, f"{base_output}_raw.pth")
    save_combined_data(negative_data, f"{neg_base}_raw.pth")

    # Apply normalizations to both banks
    normalization_methods = ['l2', 'zca', 'l2zca', 'l1', 'l1zca', 'gl']
    for method in normalization_methods:
        if method in args.normalization:
            for data, base in ((combined_data, base_output), (negative_data, neg_base)):
                if not torch.is_tensor(data['concepts']) or data['concepts'].numel() == 0:
                    continue
                normalized_concepts = apply_normalization(data['concepts'], method)
                data_copy = data.copy()
                data_copy['concepts'] = torch.tensor(normalized_concepts, dtype=torch.float32)
                save_combined_data(data_copy, f"{base}_{method}.pth")
    
    # Delete original files if requested (default: False)
    if getattr(args, 'delete', False):
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
