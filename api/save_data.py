"""
Utilities for converting concept PTH files to graph-ready JSON data.

This module is a Python adaptation of the TypeScript/JavaScript logic:

- processRawData
- cosineSimilarity / cosineSimilarityMatrix
- getColouredLinks

It is intended to be used by the /concept-projection/run API endpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA

from pth_to_json_converter import tensor_to_list


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity between two 1D vectors.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length for cosine similarity.")

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(y * y for y in b) ** 0.5
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def cosine_similarity_matrix(vectors: List[List[float]]) -> List[List[float]]:
    """
    Compute an upper-triangular cosine similarity matrix for a list of vectors.
    """
    n = len(vectors)
    matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(vectors[i], vectors[j])
            matrix[i][j] = sim
    return matrix


def get_coloured_links(links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add a green color with opacity to links based on their similarity value.
    This mirrors the getColouredLinks implementation from math.js, but uses RGBA
    so the frontend can directly control link opacity.
    """
    if not links:
        return []

    values = [float(l["value"]) for l in links if "value" in l]
    if not values:
        return links

    v_min = min(values)
    v_max = max(values)
    denom = (v_max - v_min) or 1.0

    coloured: List[Dict[str, Any]] = []
    for l in links:
        v = float(l.get("value", 0.0))
        t = (v - v_min) / denom
        # Logistic squashing around 0.5 to emphasise mid–high similarities
        t = 1.0 / (1.0 + np.exp(-14.0 * (t - 0.5)))

        # Map t ∈ [0,1] to opacity. Keep a small minimum opacity so very low
        # similarities are still faintly visible.
        alpha_min, alpha_max = 0.1, 1.0
        alpha = alpha_min + t * (alpha_max - alpha_min)

        coloured.append(
            {
                **l,
                # Fixed green channel, opacity encodes similarity
                "color": f"rgba(0, 254, 0, {alpha:.3f})",
            }
        )
    return coloured


def process_raw_data(
    raw: Dict[str, Any],
    n_components_2d: int = 2,
    n_components_3d: int = 3,
    similarity_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Python port of processRawData(raw) from TypeScript.

    Args:
        raw: JSON-like dict with keys 'concepts', 'text_grounding', 'image_grounding_paths', 'image_grounding_bboxes'.
        n_components_2d: Number of PCA components for 2D embedding.
        n_components_3d: Number of PCA components for 3D embedding.
        similarity_threshold: Threshold for creating links.

    Returns:
        dict with 'nodes' and 'links'.
    """
    vectors = raw.get("concepts") or []
    text_grounding = raw.get("text_grounding") or []
    images = raw.get("image_grounding_paths") or []
    bboxes = raw.get("image_grounding_bboxes") or []

    # Ensure lists of lists of floats
    vectors_arr = np.array(vectors, dtype=float)
    if vectors_arr.ndim != 2:
        raise ValueError(f"Expected 'concepts' to be 2D, got shape {vectors_arr.shape}")

    n_samples, n_features = vectors_arr.shape

    # Texts: first element of each text_grounding entry or 'unknown'
    texts: List[str] = []
    for entry in text_grounding:
        if isinstance(entry, (list, tuple)) and entry:
            texts.append(str(entry[0]))
        else:
            texts.append("unknown")

    # Ensure lengths match or pad/truncate
    if len(texts) < n_samples:
        texts.extend(["unknown"] * (n_samples - len(texts)))
    elif len(texts) > n_samples:
        texts = texts[:n_samples]

    if len(images) < n_samples:
        images = list(images) + [None] * (n_samples - len(images))
    elif len(images) > n_samples:
        images = list(images)[:n_samples]

    if len(bboxes) < n_samples:
        bboxes = list(bboxes) + [[]] * (n_samples - len(bboxes))
    elif len(bboxes) > n_samples:
        bboxes = list(bboxes)[:n_samples]

    # PCA embeddings
    n_components_2d = max(1, min(int(n_components_2d), n_features, n_samples))
    n_components_3d = max(1, min(int(n_components_3d), n_features, n_samples))

    pca2 = PCA(n_components=n_components_2d).fit_transform(vectors_arr)
    pca3 = PCA(n_components=n_components_3d).fit_transform(vectors_arr)

    # Convert to Python lists for JSON-serialisable output
    pca2_list: List[List[float]] = pca2.tolist()
    pca3_list: List[List[float]] = pca3.tolist()

    nodes: List[Dict[str, Any]] = []
    for i in range(n_samples):
        # Convert bboxes to list of lists if needed (handle tensor conversion)
        node_bboxes = bboxes[i] if i < len(bboxes) else []
        if node_bboxes is None:
            node_bboxes = []
        # Ensure bboxes are plain Python lists (not tensors)
        if isinstance(node_bboxes, (list, tuple)):
            node_bboxes = [
                [float(x) for x in bbox] if isinstance(bbox, (list, tuple)) else []
                for bbox in node_bboxes
            ]
        else:
            node_bboxes = []
        
        node = {
            "id": i,
            "name": texts[i],
            "images": images[i],
            "bboxes": node_bboxes,
            "x2d": pca2_list[i][0] if n_components_2d >= 1 else 0.0,
            "y2d": pca2_list[i][1] if n_components_2d >= 2 else 0.0,
            "x3d": pca3_list[i][0] if n_components_3d >= 1 else 0.0,
            "y3d": pca3_list[i][1] if n_components_3d >= 2 else 0.0,
            "z3d": pca3_list[i][2] if n_components_3d >= 3 else 0.0,
        }
        nodes.append(node)

    # Cosine similarity matrix operates on plain Python lists
    vectors_list: List[List[float]] = vectors_arr.tolist()
    cosine_matrix = cosine_similarity_matrix(vectors_list)

    links: List[Dict[str, Any]] = []
    for i, row in enumerate(cosine_matrix):
        for j, sim in enumerate(row):
            if i < j and sim > similarity_threshold:
                # Use cosine distance as strength for links (1 - similarity)
                dist = 1.0 - float(sim)
                links.append({"source": i, "target": j, "value": dist})

    coloured_links = get_coloured_links(links)
    return {"nodes": nodes, "links": coloured_links}


def pth_to_processed_graph(
    pth_path: str,
    similarity_threshold: float = 0.5,
    n_components_2d: int = 2,
    n_components_3d: int = 3,
) -> Dict[str, Any]:
    """
    Convenience helper:
    1) load a .pth file
    2) convert tensors to JSON-like Python structures using tensor_to_list
    3) run process_raw_data on the resulting dict
    """
    import torch

    data = torch.load(pth_path, map_location="cpu", weights_only=False)
    clean = tensor_to_list(data)
    return process_raw_data(
        clean,
        n_components_2d=n_components_2d,
        n_components_3d=n_components_3d,
        similarity_threshold=similarity_threshold,
    )


def process_interactive_graph_data(raw_results: Dict[str, Any], graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Python port of processInteractiveGraphData from TypeScript.
    
    Processes the raw API response and graph data to create an interactive graph structure.
    
    Args:
        raw_results: The raw response from explainer.explain_with_concept() 
                     (should have results[0].per_token_concepts and results[0].model_output)
        graph_data: The graph data from /concept-projection/run 
                    (should have nodes array with concept embeddings)
    
    Returns:
        dict with keys: nodes, links, textualOutput, colorMap
    """
    COLORS = [
        '#E57373', '#F06292', '#BA68C8', '#64B5F6', '#4DB6AC',
        '#81C784', '#DCE775', '#FFD54F', '#FFB74D', '#A1887F'
    ]
    
    # Extract per_token_concepts from raw results
    raw_data = raw_results.get("results", [])
    if not raw_data:
        return {"nodes": [], "links": [], "textualOutput": "", "colorMap": {}}
    
    first_result = raw_data[0]
    concepts = first_result.get("per_token_concepts", [])
    if not isinstance(concepts, list):
        concepts = [concepts] if concepts else []
    
    textual_output = first_result.get("model_output", "")
    
    node_map: Dict[int, Dict[str, Any]] = {}
    links: List[Dict[str, Any]] = []
    color_map: Dict[str, str] = {}
    color_index = 0
    
    def get_color_for_name(name: str) -> str:
        nonlocal color_index
        if name not in color_map:
            color_map[name] = COLORS[color_index % len(COLORS)]
            color_index += 1
        return color_map[name]
    
    def get_node_from_data(obj: Dict[str, Any], name: str, colored: bool = False) -> Dict[str, Any]:
        concept_idx = obj.get("concept_index")
        if concept_idx is None:
            return None
        
        # Get base node data from graph_data
        base = {}
        if graph_data and "nodes" in graph_data:
            nodes_list = graph_data["nodes"]
            if isinstance(nodes_list, list) and concept_idx < len(nodes_list):
                base = nodes_list[concept_idx] or {}
        
        color = get_color_for_name(name) if colored else None
        
        # Extract image_grounding_path - prefer from concept object, fallback to base
        img_paths = obj.get("image_grounding_path")
        if not img_paths:
            img_paths = base.get("images", [])
        
        # Normalize img_paths to list
        if not isinstance(img_paths, list):
            img_paths = [img_paths] if img_paths else []
        
        # Extract bboxes - prefer from base (graph_data), fallback to concept object
        bboxes = base.get("bboxes", [])
        if not bboxes:
            # Try to get from concept's image_grounding_bboxes
            bboxes = obj.get("image_grounding_bboxes", [])
        
        # Normalize bboxes to list
        if not isinstance(bboxes, list):
            bboxes = [bboxes] if bboxes else []
        
        return {
            "id": concept_idx,
            "name": name,
            "color": color,
            "images": img_paths,
            "bboxes": bboxes,
            "x2d": base.get("x2d"),
            "y2d": base.get("y2d"),
            "x3d": base.get("x3d"),
            "y3d": base.get("y3d"),
            "z3d": base.get("z3d"),
        }
    
    # Add base token nodes
    for obj in concepts:
        # For tokens, use the concept_index from the token itself (set during post-processing)
        concept_idx = obj.get("concept_index")
        if concept_idx is None:
            # Fallback: try to get from top_concepts[0] if available
            top_concepts = obj.get("top_concepts", [])
            if top_concepts and len(top_concepts) > 0:
                concept_idx = top_concepts[0].get("concept_index")
        
        # Only process if we have a valid concept_index
        if concept_idx is not None:
            if concept_idx not in node_map:
                token_text = obj.get("token_text", "unknown")
                # Create a node data object with concept_index for get_node_from_data
                token_obj = {**obj, "concept_index": concept_idx}
                node = get_node_from_data(token_obj, token_text, colored=True)
                if node:
                    node_map[concept_idx] = node
            
            # Add linked concept nodes + links
            top_concepts = obj.get("top_concepts", [])
            for sim in top_concepts:
                sim_concept_idx = sim.get("concept_index")
                if sim_concept_idx is not None:
                    # Use cosine distance as strength (1 - similarity)
                    sim_val = float(sim.get("similarity", 0.0))
                    dist = 1.0 - sim_val
                    links.append({
                        "source": concept_idx,
                        "target": sim_concept_idx,
                        "value": dist,
                    })
                    
                    if sim_concept_idx not in node_map:
                        text_grounding = sim.get("text_grounding", [])
                        name = text_grounding[0] if text_grounding and len(text_grounding) > 0 else "unknown"
                        node = get_node_from_data(sim, name, colored=False)
                        if node:
                            node_map[sim_concept_idx] = node
    
    # Apply colored links
    colored_links = get_coloured_links(links)
    
    return {
        "nodes": list(node_map.values()),
        "links": colored_links,
        "textualOutput": textual_output,
        "colorMap": color_map,
    }


if __name__ == "__main__":
    # Simple manual test hook; adjust paths as needed.
    from pathlib import Path

    ROOT = Path(__file__).parent.parent
    sample_pth = ROOT / "outputs" / "screen_run" / "concept" / "snmf" / "combined_concept_snmf_raw.pth"
    if sample_pth.exists():
        result = pth_to_processed_graph(str(sample_pth))
        print(
            f"Processed graph: {len(result['nodes'])} nodes, {len(result['links'])} links"
        )
    else:
        print(f"Sample PTH not found at {sample_pth}")



