"""
PTH to JSON converter for frontend data.
Converts PyTorch .pth files to JSON format suitable for frontend consumption.
Based on notebooks/front_end_data.ipynb
"""
import ast
import os
import shutil
import json
from pathlib import Path

# Import torch lazily to avoid issues if not installed
try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch/numpy not available. PTH conversion will be skipped.")


def tensor_to_list(obj):
    """Recursively convert torch.Tensor → list (via numpy), and handle nested structures."""
    if not HAS_TORCH:
        return obj
    
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
    parts = os.path.normpath(img_path).split(os.sep)
    if 'crops' in parts:
        crops_idx = parts.index('crops')
        return os.path.join(*parts[crops_idx+1:])
    else:
        return os.path.basename(img_path)


def process_image_grounding_paths(data, crop_root, proto_root):
    """Copy images to prototypes with crop subdir pattern and update paths. 
    Handles list of lists. Returns relative path from prototypes dir."""
    
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
                os.makedirs(target_dir, exist_ok=True)
            try:
                if os.path.exists(img_path):
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


def convert_pth_to_json(pth_path, json_path, crop_root, proto_root):
    """Convert a PTH file to JSON format."""
    if not HAS_TORCH:
        raise RuntimeError("torch is required for PTH conversion")
    
    print(f"Loading {pth_path} ...")
    data = torch.load(pth_path, map_location="cpu", weights_only=False)

    print("Converting tensors to JSON-serializable format ...")
    clean_data = tensor_to_list(data)

    print("Processing image_grounding_paths ...")
    clean_data = process_image_grounding_paths(clean_data, crop_root, proto_root)

    print(f"Saving to {json_path} ...")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(clean_data, f, indent=2)

    print("✅ PTH to JSON conversion complete!")
    return clean_data


def process_vlm_explanations(xai_output, frontend_dir, snmf_dir=None):
    """Process VLM explanations JSON and update paths for frontend.
    Saves the updated JSON in both frontend_dir and snmf_dir (if provided)."""
    prediction_path = os.path.join(xai_output, "explanations", "snmf", "vlm_explanations.json")
    
    if not os.path.exists(prediction_path):
        print(f"Warning: VLM explanations not found at {prediction_path}")
        return None
    
    frontend_input_dir = os.path.join(frontend_dir, "input")
    prototypes_dir = os.path.join(frontend_dir, "prototypes")
    
    os.makedirs(frontend_input_dir, exist_ok=True)
    os.makedirs(prototypes_dir, exist_ok=True)
    
    with open(prediction_path, "r") as f:
        pred_data = json.load(f)
    
    for result in pred_data.get("results", []):
        # Update image_path: copy to frontend/input and set relative path
        orig_img_path = result.get("image_path")
        if orig_img_path and os.path.exists(orig_img_path):
            img_filename = os.path.basename(orig_img_path)
            input_img_path = os.path.join(frontend_input_dir, img_filename)
            try:
                shutil.copy(orig_img_path, input_img_path)
            except Exception as e:
                print(f"Warning: Could not copy {orig_img_path} to {input_img_path}: {e}")
            result["image_path"] = os.path.join("input", img_filename)
        
        # Update image_grounding_path entries in per_token_concepts
        for token in result.get("per_token_concepts", []):
            # Add concept_index from the top-ranked concept (rank 1)
            top_concepts = token.get("top_concepts", [])
            if top_concepts and len(top_concepts) > 0:
                token["concept_index"] = top_concepts[0].get("concept_index")
            
            for concept in top_concepts:
                img_grounding_str = concept.get("image_grounding_path")
                if img_grounding_str:
                    try:
                        img_list = ast.literal_eval(img_grounding_str) if isinstance(img_grounding_str, str) else img_grounding_str
                    except Exception:
                        img_list = []
                    new_img_list = []
                    for abs_path in img_list:
                        parts = os.path.normpath(abs_path).split(os.sep)
                        if 'crops' in parts:
                            crops_idx = parts.index('crops')
                            rel_path = os.path.join(*parts[crops_idx+1:])
                            proto_path = os.path.join("prototypes", rel_path)
                            new_img_list.append(proto_path)
                        else:
                            new_img_list.append(os.path.basename(abs_path))
                    concept["image_grounding_path"] = new_img_list
        
        # Update image_grounding_path entries in top_concepts_over_sequence
        for concept in result.get("top_concepts_over_sequence", []):
            img_grounding_str = concept.get("image_grounding_path")
            if img_grounding_str:
                try:
                    img_list = ast.literal_eval(img_grounding_str) if isinstance(img_grounding_str, str) else img_grounding_str
                except Exception:
                    img_list = []
                new_img_list = []
                for abs_path in img_list:
                    parts = os.path.normpath(abs_path).split(os.sep)
                    if 'crops' in parts:
                        crops_idx = parts.index('crops')
                        rel_path = os.path.join(*parts[crops_idx+1:])
                        proto_path = os.path.join("prototypes", rel_path)
                        new_img_list.append(proto_path)
                    else:
                        new_img_list.append(os.path.basename(abs_path))
                concept["image_grounding_path"] = new_img_list
    
    # Save updated prediction JSON in frontend directory
    updated_pred_path = os.path.join(frontend_dir, "vlm_explanations_frontend.json")
    with open(updated_pred_path, "w") as f:
        json.dump(pred_data, f, indent=2)
    
    print(f"✅ Updated VLM explanations saved to {updated_pred_path}")
    
    # Also save in snmf directory if provided
    if snmf_dir:
        snmf_pred_path = os.path.join(snmf_dir, "vlm_explanations_frontend.json")
        os.makedirs(snmf_dir, exist_ok=True)
        with open(snmf_pred_path, "w") as f:
            json.dump(pred_data, f, indent=2)
        print(f"✅ Updated VLM explanations also saved to {snmf_pred_path}")
    
    return pred_data


def process_pipeline_output(xai_output_dir):
    """
    Process the entire pipeline output directory and generate frontend-ready data.
    
    Args:
        xai_output_dir: Path to the pipeline output directory (e.g., outputs/api_runs)
    
    Returns:
        dict with paths to generated files and loaded data
    """
    xai_output = Path(xai_output_dir)
    
    # Frontend directories for prototypes and input images
    frontend_dir = xai_output / "frontend"
    prototypes_path = frontend_dir / "prototypes"
    image_crop_path = frontend_dir / "crops"
    
    # The snmf directory where PTH files are and where JSON will be saved
    snmf_dir = xai_output / "concept" / "snmf"
    
    # Create directories
    frontend_dir.mkdir(parents=True, exist_ok=True)
    prototypes_path.mkdir(parents=True, exist_ok=True)
    image_crop_path.mkdir(parents=True, exist_ok=True)
    
    result = {
        "success": False,
        "frontend_dir": str(frontend_dir),
        "snmf_dir": str(snmf_dir),
        "files": {},
        "concept_data": None,
        "vlm_explanations_data": None
    }
    
    # Find and convert concept PTH file
    concept_pth = snmf_dir / "combined_concept_snmf_raw.pth"
    
    if not concept_pth.exists():
        # Try alternative path
        concept_pth = snmf_dir / "combined_concept_snmf_gl.pth"
    
    if concept_pth.exists():
        try:
            # Save JSON in the SAME directory as PTH (concept/snmf/)
            json_filename = concept_pth.stem + ".json"
            output_json_path = snmf_dir / json_filename
            
            concept_data = convert_pth_to_json(
                str(concept_pth),
                str(output_json_path),
                str(image_crop_path),
                str(prototypes_path)
            )
            
            result["files"]["concept_json"] = str(output_json_path)
            result["concept_data"] = concept_data
            print(f"✅ Concept data converted: {output_json_path}")
        except Exception as e:
            print(f"Error converting concept PTH: {e}")
            result["error"] = str(e)
    else:
        print(f"Warning: Concept PTH file not found at {concept_pth}")
    
    # Process VLM explanations - save in both frontend and snmf directories
    try:
        vlm_data = process_vlm_explanations(str(xai_output), str(frontend_dir), str(snmf_dir))
        if vlm_data:
            result["files"]["vlm_explanations"] = str(snmf_dir / "vlm_explanations_frontend.json")
            result["files"]["vlm_explanations_frontend"] = str(frontend_dir / "vlm_explanations_frontend.json")
            result["vlm_explanations_data"] = vlm_data
    except Exception as e:
        print(f"Error processing VLM explanations: {e}")
    
    result["success"] = bool(result.get("concept_data") or result.get("vlm_explanations_data"))
    return result


def check_image_grounding_paths_exist(json_path, frontend_root):
    """Verify all image grounding paths exist."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    missing_files = []
    
    def check_paths(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'image_grounding_paths':
                    paths_to_check = v if isinstance(v, list) else [v]
                    for sublist in paths_to_check:
                        if isinstance(sublist, list):
                            for rel_path in sublist:
                                abs_path = os.path.join(frontend_root, rel_path)
                                if not os.path.exists(abs_path):
                                    missing_files.append(rel_path)
                        else:
                            abs_path = os.path.join(frontend_root, sublist)
                            if not os.path.exists(abs_path):
                                missing_files.append(sublist)
                else:
                    check_paths(v)
        elif isinstance(obj, list):
            for item in obj:
                check_paths(item)
    
    check_paths(data)
    
    if missing_files:
        print(f"Missing files ({len(missing_files)}):")
        for path in missing_files[:10]:  # Show first 10
            print(f"  - {path}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        return False
    else:
        print("✅ All files listed under 'image_grounding_paths' exist.")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PTH files to JSON for frontend")
    parser.add_argument("--output_dir", required=True, help="Pipeline output directory")
    parser.add_argument("--verify", action="store_true", help="Verify image paths exist")
    
    args = parser.parse_args()
    
    result = process_pipeline_output(args.output_dir)
    
    if result["success"]:
        print("\n" + "="*50)
        print("Frontend data generation complete!")
        print(f"Frontend directory: {result['frontend_dir']}")
        for name, path in result["files"].items():
            print(f"  - {name}: {path}")
        
        if args.verify and result["files"].get("concept_json"):
            print("\nVerifying image paths...")
            check_image_grounding_paths_exist(
                result["files"]["concept_json"],
                result["frontend_dir"]
            )
    else:
        print("Frontend data generation failed!")
        if result.get("error"):
            print(f"Error: {result['error']}")

