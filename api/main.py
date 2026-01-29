"""
FastAPI application for XL-VLMS pipeline.
Two endpoints:
1. /run - Fast VLM explainer with pre-built concepts
2. /run-full - Full pipeline (slower, generates concepts from scratch)
"""
import os
import shutil
import subprocess
import json
import sys
import threading
import ast
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from io import BytesIO
from enum import Enum

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image

# Import PTH to JSON converter
from api.pth_to_json_converter import process_pipeline_output, process_vlm_explanations, tensor_to_list
from api.save_data import process_raw_data, process_interactive_graph_data

# Optional imports for dimensionality reduction
try:
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
except ImportError:  # pragma: no cover - handled gracefully at runtime
    PCA = None
    LinearDiscriminantAnalysis = None

try:
    import umap
except ImportError:  # pragma: no cover - handled gracefully at runtime
    umap = None

# Add inference directory to path for vlm_explainer_multibatch
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR / "inference"))

# Initialize FastAPI app
app = FastAPI(
    title="XL-VLMS Pipeline API",
    description="Upload an image and prompt, run the full pipeline, get JSON results",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration (note: ROOT_DIR already defined above)
TRAIN_DIR = ROOT_DIR / "data" / "train"
TRAIN1_DIR = ROOT_DIR / "data" / "train-og"
PIPELINE_SCRIPT = ROOT_DIR / "scripts" / "run_full_pipeline_without_coroping.sh"
OUTPUTS_BASE = ROOT_DIR / "outputs"
TEMP_IMAGES_DIR = ROOT_DIR / "api" / "temp_images"

# Pre-built concept file (use the larger one with more concepts)
CONCEPT_PTH = ROOT_DIR / "outputs" / "screen_run" / "concept" / "snmf" / "combined_concept_snmf_raw.pth"
CONCEPT_PROJECTION_DIR = CONCEPT_PTH.parent / "projections"

# Screen run concept files for dimensionality reduction config endpoint
SCREEN_RUN_CONCEPT_DIR = ROOT_DIR / "outputs" / "screen_run" / "concept" / "snmf"
SCREEN_RUN_GL_PTH = SCREEN_RUN_CONCEPT_DIR / "combined_concept_snmf_gl.pth"

VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LAYER_PATH = "model.language_model.norm"

# Global state to track pipeline status
pipeline_status = {
    "running": False,
    "last_error": None
}

# Global explainer instance (loaded once at startup, reused across requests)
global_explainer = None
explainer_lock = threading.Lock()  # Thread lock for thread-safe access


class ProjectionMethod(str, Enum):
    PCA = "pca"
    LDA = "lda"


class ProjectionRequest(BaseModel):
    """
    Configuration for dimensionality reduction on concept embeddings.
    Only two methods are supported: PCA and LDA.
    Only one parameter is configurable: n_components (1–3).
    """

    method: ProjectionMethod = Field(
        ...,
        description="Dimensionality reduction method to apply: 'pca' or 'lda'.",
    )
    n_components: Optional[int] = Field(
        None,
        ge=1,
        le=3,
        description="Number of components to compute (1–3). If omitted, a sensible default is used per method.",
    )


class CropRequest(BaseModel):
    """
    Request body for cropping an image from data/train1.
    """

    image_path: str = Field(
        ...,
        description='Relative path under data/train1, e.g. "flecked/flecked_0052.jpg".',
    )
    bboxes: Optional[List[List[int]]] = Field(
        None,
        description=(
            "Optional list of bounding boxes as [x1, y1, x2, y2]. "
            "If omitted or empty, the full image is returned."
        ),
    )


def resize_image_to_512x512(image_data: bytes) -> bytes:
    """
    Resize/crop image to 512x512 pixels.
    Maintains aspect ratio by cropping from center if needed.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Resized image as bytes
    """
    # Open image from bytes
    img = Image.open(BytesIO(image_data))
    
    # Convert to RGB if necessary (handles RGBA, P, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Get original dimensions
    original_width, original_height = img.size
    
    # Calculate target size (512x512)
    target_size = (512, 512)
    
    # Calculate scaling to maintain aspect ratio
    # We'll crop to center to get exactly 512x512
    if original_width > original_height:
        # Landscape: scale height to 512, crop width
        scale = 512 / original_height
        new_width = int(original_width * scale)
        new_height = 512
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Crop from center
        left = (new_width - 512) // 2
        img = img.crop((left, 0, left + 512, 512))
    elif original_height > original_width:
        # Portrait: scale width to 512, crop height
        scale = 512 / original_width
        new_width = 512
        new_height = int(original_height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Crop from center
        top = (new_height - 512) // 2
        img = img.crop((0, top, 512, top + 512))
    else:
        # Square: just resize
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert back to bytes
    output = BytesIO()
    img.save(output, format='JPEG', quality=95)
    return output.getvalue()


def get_or_create_explainer():
    """Get the global explainer instance, creating it if it doesn't exist (thread-safe)."""
    global global_explainer
    if global_explainer is None:
        with explainer_lock:
            # Double-check pattern: another thread might have created it while we waited
            if global_explainer is None:
                print("Initializing global VLM explainer (loading model and concepts)...")
                try:
                    from vlm_explainer_multibatch import VLMConceptExplainer
                except ImportError as e:
                    raise RuntimeError(f"Failed to import VLM explainer: {e}")
                
                # Check if concept file exists
                if not CONCEPT_PTH.exists():
                    raise FileNotFoundError(
                        f"Concept file not found: {CONCEPT_PTH}. Please run /run-full first to generate concepts."
                    )
                
                # Create explainer with default settings (prompt settings will be updated per request)
                global_explainer = VLMConceptExplainer(
                    model_name=VLM_MODEL,
                    concept_path=str(CONCEPT_PTH),
                    layer_path=LAYER_PATH,
                    prompt_mode="unsupervised",  # Default, will be updated per request
                    verbose=False,
                    save_only_generated_tokens=False
                )
                print("✅ Global VLM explainer initialized successfully")
    return global_explainer


def _load_screen_run_concepts(pth_path: Path):
    """
    Load concept embeddings from a screen_run SNMF .pth file.
    Returns (data_dict, concepts_np) and validates expected structure.
    """
    if not pth_path.exists():
        raise FileNotFoundError(f"Concept file not found: {pth_path}")

    try:
        import torch
        import numpy as np
    except ImportError as e:
        raise RuntimeError(f"torch and numpy are required to load concepts: {e}")

    data = torch.load(pth_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "concepts" not in data:
        raise ValueError(f"Unexpected PTH structure in {pth_path}: expected dict with 'concepts' key.")

    concepts = data["concepts"]
    if not hasattr(concepts, "shape"):
        raise ValueError(f"'concepts' in {pth_path} does not have a shape attribute.")

    concepts_np = concepts.detach().cpu().numpy().astype("float32")
    return data, concepts_np


def _ensure_projection_dependencies(method: ProjectionMethod):
    """
    Ensure that the required libraries for the given method are available.
    """
    if method in (ProjectionMethod.PCA, ProjectionMethod.LDA):
        if PCA is None or LinearDiscriminantAnalysis is None:
            raise HTTPException(
                status_code=500,
                detail="scikit-learn is required for PCA/LDA. Please install 'scikit-learn' in the API environment.",
            )


def _run_projection(config: ProjectionRequest, concepts_np):
    """
    Apply the requested dimensionality reduction method to the given concept matrix.

    Args:
        config: ProjectionRequest with configuration parameters.
        concepts_np: numpy array of shape (n_concepts, n_features).

    Returns:
        (embedding_np, used_params_dict)
    """
    import numpy as np

    n_samples, n_features = concepts_np.shape
    method = config.method

    _ensure_projection_dependencies(method)

    if method == ProjectionMethod.PCA:
        n_components = config.n_components or min(2, n_samples, n_features)
        n_components = int(n_components)
        if n_components > min(n_samples, n_features):
            raise HTTPException(
                status_code=400,
                detail=f"PCA n_components must be <= min(n_samples, n_features) = {min(n_samples, n_features)}.",
            )
        model = PCA(
            n_components=n_components,
            whiten=bool(config.whiten),
            random_state=config.random_state,
        )
        embedding = model.fit_transform(concepts_np)
        used_params = {
            "n_components": n_components,
            "whiten": bool(config.whiten),
            "random_state": config.random_state,
        }
    elif method == ProjectionMethod.LDA:
        if not config.labels:
            raise HTTPException(
                status_code=400,
                detail="LDA requires 'labels' in the request body (one integer label per concept).",
            )
        if len(config.labels) != n_samples:
            raise HTTPException(
                status_code=400,
                detail=f"LDA labels length ({len(config.labels)}) must equal number of concepts ({n_samples}).",
            )
        y = np.array(config.labels, dtype=int)
        n_classes = int(len(np.unique(y)))
        max_components = max(1, n_classes - 1)
        n_components = config.n_components or max_components
        n_components = int(n_components)
        if n_components > max_components:
            raise HTTPException(
                status_code=400,
                detail=f"LDA n_components must be <= n_classes - 1 = {max_components}.",
            )
        model = LinearDiscriminantAnalysis(n_components=n_components)
        embedding = model.fit_transform(concepts_np, y)
        used_params = {
            "n_components": n_components,
            "n_classes": n_classes,
        }
    else:  # UMAP
        n_components = config.n_components or 2
        n_components = int(n_components)
        if n_components < 1:
            raise HTTPException(
                status_code=400,
                detail="UMAP n_components must be >= 1.",
            )
        n_neighbors = config.n_neighbors or 15
        reducer = umap.UMAP(
            n_neighbors=int(n_neighbors),
            n_components=n_components,
            min_dist=float(config.min_dist if config.min_dist is not None else 0.1),
            metric=config.metric or "euclidean",
            random_state=config.random_state,
        )
        embedding = reducer.fit_transform(concepts_np)
        used_params = {
            "n_components": n_components,
            "n_neighbors": int(n_neighbors),
            "min_dist": float(config.min_dist if config.min_dist is not None else 0.1),
            "metric": config.metric or "euclidean",
            "random_state": config.random_state,
        }

    return embedding, used_params


@app.post("/crop")
async def crop_train1_image(req: CropRequest):
    """
    Crop an image from data/train1 using provided bounding boxes.
    
    - image_path must be of the form "concept/image", e.g. "flecked/flecked_0052.jpg"
      (no leading slashes or additional directories).
    - If bboxes is omitted or empty, the full image is returned.
    - If multiple bboxes are provided, their union is used for cropping.
    """
    # Basic validation on the relative path format
    rel_path = req.image_path.strip()
    if not rel_path or rel_path.startswith("/") or ".." in rel_path:
        raise HTTPException(
            status_code=400,
            detail=(
                'Invalid image_path. Expected a path like "concept/image", '
                'e.g. "flecked/flecked_0052.jpg".'
            ),
        )

    # Ensure it contains exactly one directory separator (concept/image)
    if "/" not in rel_path:
        raise HTTPException(
            status_code=400,
            detail=(
                'image_path must be of the form "concept/image", '
                'e.g. "flecked/flecked_0052.jpg".'
            ),
        )

    img_path = (TRAIN1_DIR / rel_path).resolve()
    # Prevent path traversal outside TRAIN1_DIR
    try:
        img_path.relative_to(TRAIN1_DIR.resolve())
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                'Invalid image_path. We only accept paths like "concept/image", '
                'e.g. "flecked/flecked_0052.jpg".'
            ),
        )

    if not img_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f'Image not found for path "{req.image_path}". '
                'We only accept paths like "concept/image", '
                'e.g. "flecked/flecked_0052.jpg".'
            ),
        )

    # Open image
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to open image: {e}",
        )

    width, height = img.size

    # If no bboxes provided, return full image
    bboxes = req.bboxes or []
    if not bboxes:
        crop_box = (0, 0, width, height)
    else:
        # Compute union of all bounding boxes
        xs1, ys1, xs2, ys2 = [], [], [], []
        for box in bboxes:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise HTTPException(
                    status_code=400,
                    detail="Each bounding box must be a list of 4 integers: [x1, y1, x2, y2].",
                )
            x1, y1, x2, y2 = map(int, box)
            xs1.append(max(0, x1))
            ys1.append(max(0, y1))
            xs2.append(min(width, x2))
            ys2.append(min(height, y2))

        crop_box = (
            max(0, min(xs1)),
            max(0, min(ys1)),
            min(width, max(xs2)),
            min(height, max(ys2)),
        )

        # Ensure valid crop box
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            raise HTTPException(
                status_code=400,
                detail="Computed crop area is empty. Please check bounding boxes.",
            )

    cropped = img.crop(crop_box)

    # Encode to JPEG in-memory
    buf = BytesIO()
    cropped.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")


@app.get("/concept-projection/config")
async def get_concept_projection_config():
    """
    Return available configuration options for PCA, LDA, and UMAP
    based on the screen_run SNMF concepts in combined_concept_snmf_gl.pth.
    """
    try:
        _, concepts_np = _load_screen_run_concepts(SCREEN_RUN_GL_PTH)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load concepts from {SCREEN_RUN_GL_PTH}: {e}",
        )

    n_concepts, n_features = concepts_np.shape
    max_pca_components = min(n_concepts, n_features, 3)

    config = {
        "source_file": str(SCREEN_RUN_GL_PTH),
        "data_shape": {
            "n_concepts": n_concepts,
            "n_features": n_features,
        },
        "methods": {
            "pca": {
                "description": "Principal Component Analysis on the concept embedding matrix.",
                "params": {
                    "n_components": {
                        "type": "int",
                        "min": 1,
                        "max": max_pca_components,
                        "default": min(2, max_pca_components),
                    }
                },
            },
            "lda": {
                "description": "Linear Discriminant Analysis on concept embeddings.",
                "params": {
                    "n_components": {
                        "type": "int",
                        "min": 1,
                        "max": min(3, n_concepts - 1 if n_concepts > 1 else 1),
                        "default": 1,
                    }
                },
            },
        },
    }

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Concept projection configuration.",
            "config": config,
        },
    )


@app.post("/concept-projection/run")
async def run_concept_projection(config: ProjectionRequest):
    """
    Apply PCA-based graph processing to the concepts in combined_concept_snmf_raw.pth
    using the provided configuration and return {nodes, links}.

    Note: Currently, the underlying processing is PCA-based. When method='lda',
    the same graph is computed but cached under a different key so clients can
    distinguish configurations.
    """

    try:
        import torch
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"torch is required to load concept file: {e}",
        )

    if not CONCEPT_PTH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Concept file not found: {CONCEPT_PTH}",
        )

    # Map API config to PCA settings (PCA-like graph processing)
    n_components_2d = 2  # always compute 2D embedding
    n_components_3d = config.n_components if config.n_components is not None else 3
    if n_components_3d > 3:
        n_components_3d = 3

    # If we ever add similarity_threshold to config, we can wire it here
    similarity_threshold = 0.95

    # Build a cache key based on config + threshold
    CONCEPT_PROJECTION_DIR.mkdir(parents=True, exist_ok=True)
    cache_key_parts = [
        f"method-{config.method.value}",
        f"nc3-{n_components_3d}",
        f"thr-{int(similarity_threshold * 100)}",
    ]
    cache_key = "_".join(cache_key_parts)
    cache_filename = f"concept_graph_{cache_key}.json"
    cache_path = CONCEPT_PROJECTION_DIR / cache_filename

    # Fast path: if cached file exists, load it and ensure links are colored
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            
            # Ensure links have colored rgba format (in case cache was created before colored links)
            if "links" in cached and cached["links"]:
                from api.save_data import get_coloured_links
                # Check if links already have color field with rgba format
                needs_coloring = False
                for link in cached["links"]:
                    if "color" not in link:
                        needs_coloring = True
                        break
                    elif "rgba" not in str(link.get("color", "")):
                        # Has color but not rgba format, needs update
                        needs_coloring = True
                        break
                
                if needs_coloring:
                    cached["links"] = get_coloured_links(cached["links"])
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load cached concept graph: {e}",
            )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Concept graph loaded from cache.",
                "method": config.method.value,
                "params": {
                    "n_components_2d": n_components_2d,
                    "n_components_3d": n_components_3d,
                    "similarity_threshold": similarity_threshold,
                },
                "cached": True,
                "cache_file": str(cache_path),
                "result": cached,
            },
        )

    # Slow path: compute from scratch and cache
    try:
        raw_pth = torch.load(CONCEPT_PTH, map_location="cpu", weights_only=False)
        raw_json_like = tensor_to_list(raw_pth)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load/convert concept file: {e}",
        )

    try:
        processed = process_raw_data(
            raw_json_like,
            n_components_2d=n_components_2d,
            n_components_3d=n_components_3d,
            similarity_threshold=similarity_threshold,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during PCA graph processing: {e}",
        )

    # Save to cache for future identical requests
    try:
        with open(cache_path, "w") as f:
            json.dump(processed, f, indent=2)
    except Exception as e:
        # Log but don't fail the request just because caching failed
        print(f"Warning: failed to write cache file {cache_path}: {e}")

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Concept graph processed successfully.",
            "method": config.method.value,
            "params": {
                "n_components_2d": n_components_2d,
                "n_components_3d": n_components_3d,
                "similarity_threshold": similarity_threshold,
            },
            "cached": False,
            "cache_file": str(cache_path),
            "result": processed,
        },
    )


@app.post("/run")
async def run_fast_explainer(
    file: UploadFile = File(...),
    prompt_mode: Optional[str] = Form("unsupervised"),
    label: Optional[str] = Form(None),
    choices: Optional[str] = Form(None),
    top_n: Optional[int] = Form(5),
    prompt: Optional[str] = Form(None),
    projection_method: Optional[str] = Form(None),
    projection_dim: Optional[int] = Form(None)
):
    """
    Fast endpoint: Explain image using pre-built concepts.
    Much faster than /run-full (no concept generation needed).
    Uses a pre-loaded model that stays in memory for better performance.
    
    Args:
        file: Image file to explain
        prompt_mode: VLM explainer mode - 'unsupervised', 'binary', or 'mcq' (default: unsupervised)
        label: Label for binary mode (e.g., "cat")
        choices: Comma-separated choices for MCQ mode (e.g., "cat,dog,bird")
        top_n: Number of top concepts to return (default: 5)
        prompt: Custom prompt to identify objects in image (optional, uses default if not provided)
        
    Returns:
        JSON with vlm_explanations_data
    """
    global pipeline_status
    
    # Check if pipeline is already running
    if pipeline_status["running"]:
        raise HTTPException(
            status_code=409,
            detail="Another request is already running. Please wait for completion."
        )
    
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    image_path = None
    try:
        pipeline_status["running"] = True
        pipeline_status["last_error"] = None
        
        # Step 1: Read, resize to 512x512, and save image to temp directory
        print("Step 1: Processing and saving image...")
        TEMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        image_filename = f"temp_{file.filename}"
        image_path = TEMP_IMAGES_DIR / image_filename
        
        # Read original image
        image_data = await file.read()
        
        # Resize/crop to 512x512
        print("   Resizing image to 512x512...")
        resized_image_data = resize_image_to_512x512(image_data)
        
        # Save resized image
        with open(image_path, "wb") as f:
            f.write(resized_image_data)
        print(f"   Image resized and saved to: {image_path}")
        
        # Step 2: Get or create global explainer (model stays loaded in memory)
        print("Step 2: Using pre-loaded VLM explainer...")
        
        try:
            explainer = get_or_create_explainer()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize explainer: {e}"
            )
        
        # Parse choices
        prompt_choices = None
        if choices:
            prompt_choices = [c.strip() for c in choices.split(",") if c.strip()]
        
        # Update prompt settings for this request (model stays loaded)
        # Use lock to ensure thread-safe updates (though current design prevents concurrent requests)
        with explainer_lock:
            explainer.prompt_mode = (prompt_mode or "unsupervised").lower()
            explainer.prompt_label = label
            explainer.prompt = prompt
            if isinstance(prompt_choices, str):
                explainer.prompt_choices = [c.strip() for c in prompt_choices.split(',') if c.strip()]
            else:
                explainer.prompt_choices = prompt_choices
        
        # Run explanation
        results = explainer.explain_with_concept(
            images=[str(image_path)],
            ground_truth_labels=None,
            top_n=top_n,
            batch_size=1
        )
        
        # Note: We do NOT close the explainer here - it stays loaded for next request
        # explainer.close()  # REMOVED - keep model in memory
        
        print("   VLM explainer completed successfully")
        
        # Step 3: Compute PCA/UMAP projections for tokens if requested
        print(f"DEBUG: projection_method={projection_method}, projection_dim={projection_dim}, type(projection_dim)={type(projection_dim)}")
        if projection_method and projection_dim:
            try:
                import numpy as np
                from sklearn.decomposition import PCA
                
                # Convert projection_dim to int (form data comes as string)
                try:
                    projection_dim_int = int(projection_dim)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid projection_dim '{projection_dim}', skipping projection")
                    projection_dim_int = None
                
                if projection_dim_int and projection_dim_int in [2, 3]:
                    # Get concept vectors from explainer to fit the projection
                    concept_vectors = explainer.concept_vectors.cpu().numpy()  # (K, D)
                    print(f"Computing {projection_method} projection with {projection_dim_int} dimensions...")
                    print(f"Concept vectors shape: {concept_vectors.shape}")
                    
                    # Fit projection on concept space
                    if projection_method.lower() == "pca":
                        projector = PCA(n_components=projection_dim_int)
                        projector.fit(concept_vectors)
                    elif projection_method.lower() == "umap":
                        try:
                            import umap
                            projector = umap.UMAP(n_components=projection_dim_int, random_state=42)
                            projector.fit(concept_vectors)
                        except ImportError:
                            print("Warning: umap-learn not available, skipping UMAP projection")
                            projector = None
                    else:
                        print(f"Warning: Unknown projection method '{projection_method}', skipping")
                        projector = None
                    
                    # Project each token's embedding
                    if projector is not None:
                        projection_count = 0
                        for result in results:
                            for token in result.get("per_token_concepts", []):
                                if "_token_embedding_np" in token:
                                    try:
                                        token_emb = token["_token_embedding_np"].reshape(1, -1)
                                        projection = projector.transform(token_emb)[0].tolist()
                                        field_name = f"projection_{projection_method.lower()}_{projection_dim_int}d"
                                        token[field_name] = projection
                                        projection_count += 1
                                    except Exception as e:
                                        print(f"ERROR projecting token {token.get('token_text', 'unknown')}: {e}")
                                        import traceback
                                        traceback.print_exc()
                        print(f"✅ Added {projection_count} token projections")
                else:
                    print(f"Warning: projection_dim must be 2 or 3, got '{projection_dim}'")
            except Exception as e:
                import traceback
                print(f"Error: Failed to compute token projections: {e}")
                traceback.print_exc()
        
        # Step 4: Format response
        pipeline_status["running"] = False
        
        # Debug: Check if projections were added (after computation, before response)
        if projection_method and projection_dim:
            try:
                proj_dim_int = int(projection_dim) if projection_dim else None
                if proj_dim_int:
                    proj_key = f"projection_{projection_method.lower()}_{proj_dim_int}d"
                    found_count = 0
                    missing_count = 0
                    for result in results:
                        for token in result.get("per_token_concepts", []):
                            if proj_key in token:
                                found_count += 1
                                if found_count <= 3:  # Only print first 3
                                    print(f"DEBUG: ✅ Found {proj_key} in token '{token.get('token_text')}': {token[proj_key][:2] if len(token[proj_key]) >= 2 else token[proj_key]}")
                            else:
                                missing_count += 1
                                if missing_count <= 3:  # Only print first 3
                                    print(f"DEBUG: ❌ Missing {proj_key} in token '{token.get('token_text')}'. Token keys: {list(token.keys())}")
                    print(f"DEBUG: Summary - Found projections: {found_count}, Missing: {missing_count}")
            except Exception as e:
                print(f"DEBUG: Error checking projections: {e}")
        
        # Post-process results to ensure arrays are properly formatted
        for result in results:
            # Process per_token_concepts
            # Fix: Assign unique concept_index to each token to avoid duplicates
            # Use token_index as the unique ID (each token already has a unique token_index)
            # This ensures the parser can create colored nodes for all tokens
            token_counter = 0
            for token in result.get("per_token_concepts", []):
                # Remove internal token embedding field (not returned to user)
                if "_token_embedding_np" in token:
                    del token["_token_embedding_np"]
                
                # CRITICAL FIX: Use token_index as unique concept_index
                # Previously, multiple tokens could have the same concept_index if they matched
                # the same concept, causing the parser to skip duplicates and show fewer colored nodes
                token_index = token.get("token_index")
                if token_index is not None:
                    token["concept_index"] = token_index
                else:
                    # Fallback: use sequential counter if token_index is missing
                    token["concept_index"] = token_counter
                    token_counter += 1
                
                top_concepts = token.get("top_concepts", [])
                # Ensure image_grounding_path and image_grounding_bboxes are arrays, not strings
                for concept in top_concepts:
                        if "image_grounding_path" in concept:
                            val = concept["image_grounding_path"]
                            if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                                # Convert string representation of list back to list
                                try:
                                    concept["image_grounding_path"] = ast.literal_eval(val)
                                except:
                                    pass
                        if "image_grounding_bboxes" in concept:
                            val = concept["image_grounding_bboxes"]
                            if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                                # Convert string representation of list back to list
                                try:
                                    concept["image_grounding_bboxes"] = ast.literal_eval(val)
                                except:
                                    pass
            
            # Process top_concepts_over_sequence
            for concept in result.get("top_concepts_over_sequence", []):
                if "image_grounding_path" in concept:
                    val = concept["image_grounding_path"]
                    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                        try:
                            concept["image_grounding_path"] = ast.literal_eval(val)
                        except:
                            pass
                if "image_grounding_bboxes" in concept:
                    val = concept["image_grounding_bboxes"]
                    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                        try:
                            concept["image_grounding_bboxes"] = ast.literal_eval(val)
                        except:
                            pass
        
        # Step 5: Load graph data and process into interactive graph format
        try:
            # Load or compute graph data (use default PCA config for graph structure)
            import torch
            raw_pth = torch.load(CONCEPT_PTH, map_location="cpu", weights_only=False)
            raw_json_like = tensor_to_list(raw_pth)
            graph_data = process_raw_data(
                raw_json_like,
                n_components_2d=2,
                n_components_3d=3,
                similarity_threshold=0.95,
            )
            
            # Process into interactive graph format (from explainer results)
            interactive_graph = process_interactive_graph_data(
                raw_results={"results": results},
                graph_data=graph_data
            )
            
            # Merge nodes and links from graph_data (full concept graph) with interactive_graph
            # Create a map of node IDs to avoid duplicates
            node_id_map = {node["id"]: node for node in interactive_graph.get("nodes", [])}
            
            # Add nodes from graph_data that aren't already in interactive_graph
            graph_nodes = graph_data.get("nodes", [])
            for node in graph_nodes:
                node_id = node.get("id")
                if node_id is not None and node_id not in node_id_map:
                    # Only add if not already present (union)
                    node_id_map[node_id] = node
            
            # Merge links - use a set of (source, target) tuples to avoid duplicates
            link_set = set()
            merged_links = []
            
            # Add links from interactive_graph
            for link in interactive_graph.get("links", []):
                source = link.get("source")
                target = link.get("target")
                if source is not None and target is not None:
                    link_key = (source, target)
                    if link_key not in link_set:
                        link_set.add(link_key)
                        merged_links.append(link)
            
            # Add links from graph_data
            for link in graph_data.get("links", []):
                source = link.get("source")
                target = link.get("target")
                if source is not None and target is not None:
                    link_key = (source, target)
                    if link_key not in link_set:
                        link_set.add(link_key)
                        merged_links.append(link)
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "VLM explainer completed successfully",
                    "nodes": list(node_id_map.values()),  # Union of nodes from both sources
                    "links": merged_links,  # Union of links from both sources
                    "textualOutput": interactive_graph.get("textualOutput", ""),
                    "colorMap": interactive_graph.get("colorMap", {})
                }
            )
        except Exception as e:
            import traceback
            print(f"Error processing interactive graph: {e}")
            traceback.print_exc()
            # Fallback to original format if processing fails
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "VLM explainer completed successfully",
                    "prompt_mode": prompt_mode,
                    "using_prebuilt_concepts": True,
                    "concept_file": str(CONCEPT_PTH),
                    "vlm_explanations_data": {
                        "model_card": VLM_MODEL,
                        "layer_path": LAYER_PATH,
                        "results": results
                    }
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        pipeline_status["running"] = False
        pipeline_status["last_error"] = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )
    finally:
        # Cleanup temp image
        try:
            if image_path and image_path.exists():
                image_path.unlink()
        except:
            pass


@app.post("/run-full")
async def run_full_pipeline_endpoint(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    prompt_mode: Optional[str] = Form("unsupervised"),
    label: Optional[str] = Form(None),
    choices: Optional[str] = Form(None)
):
    """
    Full pipeline endpoint (SLOW - generates concepts from scratch):
    1. Receives an image and optional prompt
    2. Clears and saves to data/train/
    3. Runs the full pipeline (concept discovery + feature extraction + decomposition + explanation)
    4. Converts PTH to JSON
    5. Returns the JSON data
    
    Note: Use /run for faster inference with pre-built concepts.
    
    Args:
        file: Image file to process
        prompt: Custom prompt for dataset inference (optional)
        prompt_mode: VLM explainer mode - 'unsupervised', 'binary', or 'mcq' (default: unsupervised)
        label: Label for binary mode (e.g., "cat")
        choices: Comma-separated choices for MCQ mode (e.g., "cat,dog,bird")
        
    Returns:
        JSON with concept_data and vlm_explanations_data
    """
    global pipeline_status
    
    # Check if pipeline is already running
    if pipeline_status["running"]:
        raise HTTPException(
            status_code=409,
            detail="Pipeline is already running. Please wait for completion."
        )
    
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        pipeline_status["running"] = True
        pipeline_status["last_error"] = None
        
        # Step 1: Clear train folder
        print("Step 1: Clearing train folder...")
        if TRAIN_DIR.exists():
            for item in TRAIN_DIR.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        else:
            TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        
        # Step 2: Read, resize to 512x512, and save image to train folder
        print("Step 2: Processing and saving image to train folder...")
        image_data = await file.read()
        
        # Resize/crop to 512x512
        print("   Resizing image to 512x512...")
        resized_image_data = resize_image_to_512x512(image_data)
        
        image_stem = Path(file.filename).stem
        subfolder = TRAIN_DIR / image_stem
        subfolder.mkdir(parents=True, exist_ok=True)
        image_path = subfolder / file.filename
        with open(image_path, "wb") as f:
            f.write(resized_image_data)
        print(f"   Image resized and saved to: {image_path}")
        
        # Step 3: Setup output directory
        print("Step 3: Setting up output directory...")
        output_dir = OUTPUTS_BASE / "api_runs"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 4: Run pipeline with environment variables for prompts
        print("Step 4: Running pipeline...")
        os.chmod(PIPELINE_SCRIPT, 0o755)
        
        # Set up environment with prompt parameters
        env = os.environ.copy()
        if prompt:
            env["PROMPT"] = prompt
            print(f"   Using custom prompt: {prompt[:50]}...")
        if prompt_mode:
            env["EXPL_PROMPT_MODE"] = prompt_mode
            print(f"   Prompt mode: {prompt_mode}")
        if label:
            env["EXPL_LABEL"] = label
            print(f"   Label: {label}")
        if choices:
            env["EXPL_CHOICES"] = choices
            print(f"   Choices: {choices}")
        
        cmd = [
            "bash",
            str(PIPELINE_SCRIPT),
            "--input-dir", str(ROOT_DIR / "data"),
            "--output-dir", str(output_dir)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT_DIR),
            env=env,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            pipeline_status["running"] = False
            pipeline_status["last_error"] = result.stderr
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline failed: {result.stderr}"
            )
        
        print("   Pipeline completed successfully")
        
        # Step 5: Convert PTH to JSON
        print("Step 5: Converting PTH to JSON...")
        frontend_results = process_pipeline_output(str(output_dir))
        
        if not frontend_results.get("success"):
            pipeline_status["running"] = False
            pipeline_status["last_error"] = frontend_results.get("error")
            raise HTTPException(
                status_code=500,
                detail=f"PTH to JSON conversion failed: {frontend_results.get('error')}"
            )
        
        print("   Conversion completed successfully")
        
        # Step 6: Return JSON data
        pipeline_status["running"] = False
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Pipeline completed successfully",
                "prompt_used": prompt,
                "prompt_mode": prompt_mode,
                "concept_data": frontend_results.get("concept_data"),
                "vlm_explanations_data": frontend_results.get("vlm_explanations_data")
            }
        )
        
    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        pipeline_status["running"] = False
        pipeline_status["last_error"] = "Pipeline timed out after 1 hour"
        raise HTTPException(
            status_code=500,
            detail="Pipeline execution timed out after 1 hour"
        )
    except Exception as e:
        pipeline_status["running"] = False
        pipeline_status["last_error"] = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )


@app.get("/")
async def root():
    """API info."""
    model_loaded = global_explainer is not None
    return {
        "message": "XL-VLMS Pipeline API",
        "usage": "POST /run with an image file to run the pipeline and get JSON results",
        "status": "running" if pipeline_status["running"] else "idle",
        "model_loaded": model_loaded
    }


@app.on_event("startup")
async def startup_event():
    """Load the model at startup (optional - can also be lazy loaded on first request)."""
    print("Starting up API...")
    # Optionally pre-load the model here, or let it load lazily on first request
    # Uncomment the next line to pre-load the model at startup:
    # try:
    #     get_or_create_explainer()
    # except Exception as e:
    #     print(f"Warning: Could not pre-load model at startup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up the global explainer on shutdown."""
    global global_explainer
    if global_explainer is not None:
        print("Shutting down: Closing global explainer...")
        try:
            global_explainer.close()
        except Exception as e:
            print(f"Error closing explainer: {e}")
        global_explainer = None
        print("✅ Global explainer closed")


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
