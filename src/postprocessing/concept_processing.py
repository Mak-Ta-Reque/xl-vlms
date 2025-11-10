# root output dir

import os

import shutil

import torch

import numpy as np

import json
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Union

# Ensure both repo root and src directory are on sys.path when running directly
_ROOT = Path(__file__).resolve().parents[2]
_SRC = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.models import get_model_class


xai_output = "/mnt/sdz/abka03_data/xl-vlms/outputs"
front_end_output = f"{xai_output}/frontend"
prototypes_path = f"{front_end_output}/prototypes"
image_crop_path = f"{front_end_output}/crops"
concept_pth = f"{xai_output}/run_20251016_140704/concept/snmf/combined_concept_snmf_raw.pth"

def tensor_to_list(obj):
    """Recursively convert torch.Tensor → list (via numpy), and handle nested structures."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [tensor_to_list(v) for v in obj]
    else:
        return obj

def get_relative_crop_path(img_path, crop_root):
    """Truncate the path at 'crops', so the returned path is always after crops/."""
    # Find 'crops' in the path and return everything after it
    parts = os.path.normpath(img_path).split(os.sep)
    if 'crops' in parts:
        crops_idx = parts.index('crops')
        return os.path.join(*parts[crops_idx+1:])
    else:
        return os.path.basename(img_path)

def process_image_grounding_paths(data, crop_root, proto_root):
    """Copy images to prototypes with crop subdir pattern and update paths. Handles list of lists. Returns relative path from prototypes dir."""
    def process(paths):
        # If paths is a list of lists, process each sublist
        if isinstance(paths, list) and paths and isinstance(paths[0], list):
            return [process(sublist) for sublist in paths]
        new_paths = []
        for img_path in paths:
            rel_path = get_relative_crop_path(img_path, crop_root)
            target_path = os.path.join(proto_root, rel_path)
            target_dir = os.path.dirname(target_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            try:
                shutil.copy(img_path, target_path)
            except Exception as e:
                print(f"Warning: Could not copy {img_path} to {target_path}: {e}")
            # Always return path as 'prototypes/...' (relative to workspace)
            new_paths.append(os.path.join("prototypes", rel_path))
        return new_paths

    if isinstance(data, dict):
        for k, v in data.items():
            if k == "image_grounding_paths":
                data[k] = process(v)
            else:
                process_image_grounding_paths(v, crop_root, proto_root)
    elif isinstance(data, list):
        for item in data:
            process_image_grounding_paths(item, crop_root, proto_root)
    return data

def get_pca_data(concepts):
    """Compute 2D and 3D PCA embeddings for concept vectors.

    Args:
        concepts: Array-like of shape (n_samples, n_features). Can be a
                  torch.Tensor or numpy.ndarray. Example: (40, 2048).

    Returns:
        pca_2d: numpy.ndarray of shape (n_samples, min(2, n_components_used))
        pca_3d: numpy.ndarray of shape (n_samples, n_components_used) where
                n_components_used = min(3, n_samples, n_features)
    """
    # Convert to numpy array
    if isinstance(concepts, torch.Tensor):
        X = concepts.detach().cpu().numpy()
    else:
        X = np.asarray(concepts)

    # Ensure 2D shape
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array for concepts, got shape {X.shape}")

    n_samples, n_features = X.shape
    n_comp = int(min(3, n_samples, n_features))

    if n_comp == 0:
        # No data to compute PCA; return empty arrays
        return np.empty((0, 0)), np.empty((0, 0))

    # Center the data
    X_centered = X - X.mean(axis=0, keepdims=True)
    # SVD-based PCA (no sklearn dependency)
    # X_centered = U S Vt, principal axes are rows of Vt
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_comp, :]  # shape: (n_comp, n_features)
    # Project data onto principal components
    X_pca = X_centered @ components.T  # shape: (n_samples, n_comp)

    # 3D representation (or less if limited by data)
    pca_3d = X_pca  # shape: (n_samples, n_comp)
    # 2D is first two principal components (or fewer if not available)
    pca_2d = X_pca[:, : min(2, n_comp)]

    return pca_2d, pca_3d


def get_intra_cluster_similarity(concepts, decimals: int = 4):
    """Compute full cosine similarity matrix (n x n) with 1s on diagonal.

    Given an (n, d) matrix, returns an (n, n) cosine similarity matrix where
    sims[i, j] = cosine(x_i, x_j). Values are in [-1, 1], diagonal entries are
    set to 1, and results are rounded to `decimals`.

    Args:
        concepts: torch.Tensor or np.ndarray with shape (n, d)
        decimals: number of decimal places to round in the result

    Returns:
        np.ndarray of shape (n, n) with float similarities.
    """
    # Convert to numpy float64
    if isinstance(concepts, torch.Tensor):
        X = concepts.detach().cpu().numpy()
    else:
        X = np.asarray(concepts)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array for concepts, got shape {X.shape}")

    X = X.astype(np.float64, copy=False)
    n, d = X.shape
    if n <= 1:
        return np.empty((n, 0), dtype=np.float64)

    # Normalize rows to unit length; avoid division by zero
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Replace zeros with one to avoid division by zero; rows with zero norm become zeros after normalization
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    Xn = X / safe_norms

    # Cosine similarity matrix
    sims = Xn @ Xn.T  # (n, n)
    # Clip for numerical safety
    np.clip(sims, -1.0, 1.0, out=sims)

    # Ensure perfect 1.0 on the diagonal (numerical safety)
    np.fill_diagonal(sims, 1.0)

    if decimals is not None and decimals >= 0:
        sims = np.round(sims, decimals=decimals)
    return sims

def process_text_grounding(
    text_grounding,
    model_name_or_path: Optional[str] = None,
    processor_name: Optional[str] = None,
    device: Optional[Union[torch.device, str]] = None,
    max_new_tokens: int = 48,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
):
    """Combine lists of short phrases into single strings.

    If `model_name_or_path` is provided, load the specified VLM/LLM via
    `models.get_model_class` and use it to produce a concise combined phrase
    for each top-level list. If not provided, fall back to deterministic joining.

    Args:
        text_grounding: List[List[str]], e.g., length 40, each sublist has 2-3 strings.
        model_name_or_path: Optional HF model id or local path for the model.
        processor_name: Optional processor name; defaults to `model_name_or_path` if None.
        device: torch.device or string (e.g., 'cuda' or 'cpu'). Defaults to CUDA if available.
        max_new_tokens: Generation cap for LLM output.
        local_files_only: Pass-through for model loading.
        cache_dir: Optional cache directory for model weights.

    Returns:
        List[str]: One combined string per top-level list.
    """
    # Basic validation
    if not isinstance(text_grounding, (list, tuple)):
        raise ValueError("text_grounding must be a list[list[str]]")

    # Fallback: simple deterministic join if no model was provided
    if model_name_or_path is None:
        combined = []
        for phrases in text_grounding:
            if isinstance(phrases, (list, tuple)):
                # Remove Nones and strip whitespace
                cleaned = [str(s).strip() for s in phrases if s is not None and str(s).strip()]
                combined.append("; ".join(cleaned))
            else:
                combined.append(str(phrases))
        return combined

    # Prepare device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Minimal args namespace expected by get_model_class
    args = argparse.Namespace(
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        generation_mode=True,
        max_new_tokens=max_new_tokens,
    )

    # Load model class
    model_cls = get_model_class(
        model_name_or_path=model_name_or_path,
        processor_name=processor_name if processor_name is not None else model_name_or_path,
        device=device,
        logger=None,
        args=args,
    )
    model = model_cls.get_model()
    tokenizer = model_cls.get_tokenizer()

    results: List[str] = []
    for phrases in text_grounding:
        if len(phrases) < 2:
            results.append(phrases[0] if phrases else "")
        else:
            if not isinstance(phrases, (list, tuple)):
                phrases = [str(phrases)]
            cleaned = [str(s).strip() for s in phrases if s is not None and str(s).strip()]
            # Build a concise prompt
            instruction = (f"Combine the following short phrases/ suffix prefix into a single concise phrase {cleaned}"
            )

            inputs = model_cls.preprocessor(
                instruction=instruction,
                image_file="",  # force text-only path
                response="",
                generation_mode=True,
            )

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

            # Slice off the prompt tokens to keep only generated continuation
            input_ids = inputs.get("input_ids")
            if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
                L = input_ids.shape[1]
                gen_ids = out[0, L:]
            else:
                # Fallback: best-effort decode whole output
                gen_ids = out[0]

            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            # Final cleanup
            text = " ".join(text.split())
            results.append(text)

    return results

def convert_pth_to_json(pth_path, json_path, crop_root, proto_root):
    print(f"Loading {pth_path} ...")
    data = torch.load(pth_path, map_location="cpu")

    print("Converting tensors to JSON-serializable format ...")

    pca_data_2d,  pca_data_3d =  get_pca_data(data["concepts"])
    intra_cluster_similarity = get_intra_cluster_similarity(data["concepts"])  # cosine similarity, (40, 40)
    processed_text_grounding = process_text_grounding(data["text_grounding"], model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct")  # Fallback join if no model provided
    clean_data = tensor_to_list(data)
    print("Processing image_grounding_paths ...")
    clean_data = process_image_grounding_paths(clean_data, crop_root, proto_root)

    print(f"Saving to {json_path} ...")
    with open(json_path, "w") as f:
        json.dump(clean_data, f, indent=2)

    print("✅ Conversion complete!")

# Example usage
file_path = concept_pth
# Save JSON in prototypes_path directory
json_filename = os.path.splitext(os.path.basename(file_path))[0] + ".json"
output_path = os.path.join(front_end_output, json_filename)
convert_pth_to_json(file_path, output_path, image_crop_path, prototypes_path)



