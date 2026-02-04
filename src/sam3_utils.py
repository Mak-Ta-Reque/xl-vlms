"""
SAM3 utilities for extracting bounding boxes by text tag.

Public API:
- load_sam3(device: Optional[str] = None, confidence_threshold: float = 0.5) -> Tuple[model, ...]
- predict_bboxes_for_tag_sam3(model_tuple, images, tag, topn=10) -> List[List[List[int]]]
- predict_bboxes_for_tag_sam3_batched(model_tuple, images, tag, batch_size=8, topn=10) -> List[List[List[int]]]

Each item of the returned outer list corresponds to an input image.
Each inner list contains zero or more [x, y, w, h] integer boxes for that image.

Install SAM3:
    pip install 'git+https://github.com/facebookresearch/sam3.git'

Batch Processing:
    SAM3 supports true batch inference using collate_fn_api. This module now
    implements proper batching for significantly faster throughput.
"""

from __future__ import annotations

from typing import List, Union, Optional, Sequence, Tuple, Any, Dict

import numpy as np
from PIL import Image

try:  # Optional import; module works without torch
    import torch  # type: ignore
except Exception:  # pragma: no cover - runtime convenience
    torch = None  # type: ignore

# Lazy imports for SAM3 to avoid hard dependency at module load time
_SAM3_AVAILABLE: Optional[bool] = None
_SAM3_BATCH_AVAILABLE: Optional[bool] = None


def _check_sam3_available() -> bool:
    """Check if sam3 package is available."""
    global _SAM3_AVAILABLE
    if _SAM3_AVAILABLE is not None:
        return _SAM3_AVAILABLE
    try:
        import sam3  # noqa: F401
        from sam3 import build_sam3_image_model  # noqa: F401
        from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
        _SAM3_AVAILABLE = True
    except ImportError:
        _SAM3_AVAILABLE = False
    return _SAM3_AVAILABLE


def _check_sam3_batch_available() -> bool:
    """Check if sam3 batch processing APIs are available."""
    global _SAM3_BATCH_AVAILABLE
    if _SAM3_BATCH_AVAILABLE is not None:
        return _SAM3_BATCH_AVAILABLE
    try:
        from sam3.train.data.collator import collate_fn_api  # noqa: F401
        from sam3.model.utils.misc import copy_data_to_device  # noqa: F401
        from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, InferenceMetadata  # noqa: F401
        from sam3.train.data.sam3_image_dataset import Image as SAMImage  # noqa: F401
        from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI  # noqa: F401
        from sam3.eval.postprocessors import PostProcessImage  # noqa: F401
        _SAM3_BATCH_AVAILABLE = True
    except ImportError:
        _SAM3_BATCH_AVAILABLE = False
    return _SAM3_BATCH_AVAILABLE


def load_sam3(
    device: Optional[str] = None,
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Create and return a SAM3 model instance with batch processing support.

    Args:
        device: Optional device spec (e.g., "cuda", "cpu"). If None, auto-selects.
        confidence_threshold: Confidence threshold for detections (default 0.5).

    Returns:
        Dict containing:
            - model: SAM3 model instance
            - transform: Preprocessing transform pipeline
            - postprocessor: PostProcessImage instance
            - collate_fn: Collation function for batching
            - copy_to_device_fn: Function to move data to device
            - confidence_threshold: Detection threshold
            - device: torch.device instance
            - batch_supported: Whether true batch processing is available

    Raises:
        ImportError: If sam3 package is not installed.
    """
    if not _check_sam3_available():
        raise ImportError(
            "sam3 package is required. Install via: pip install 'git+https://github.com/facebookresearch/sam3.git'"
        )

    import sam3
    from sam3 import build_sam3_image_model

    # Determine sam3 root for BPE path
    import os
    sam3_root = os.path.dirname(sam3.__file__)
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

    # Handle device selection
    use_cuda = False
    target_device = torch.device("cpu") if torch is not None else None
    if torch is not None and torch.cuda.is_available():
        use_cuda = True
        target_device = torch.device("cuda")
        if device:
            dev = str(device).lower()
            if dev == "cpu":
                use_cuda = False
                target_device = torch.device("cpu")
            elif dev.startswith("cuda"):
                target_device = torch.device(dev)
                if ":" in dev:
                    try:
                        idx = int(dev.split(":", 1)[1])
                        torch.cuda.set_device(idx)
                    except Exception:
                        pass

    # Enable TF32 and autocast for better performance on Ampere GPUs
    if use_cuda and torch is not None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build model
    model = build_sam3_image_model(bpe_path=bpe_path)

    # Check if batch processing APIs are available
    batch_supported = _check_sam3_batch_available()
    
    result = {
        "model": model,
        "confidence_threshold": confidence_threshold,
        "device": target_device,
        "batch_supported": batch_supported,
        "transform": None,
        "postprocessor": None,
        "collate_fn": None,
        "copy_to_device_fn": None,
    }

    if batch_supported:
        from sam3.train.data.collator import collate_fn_api
        from sam3.model.utils.misc import copy_data_to_device
        from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
        from sam3.eval.postprocessors import PostProcessImage

        # Create transform pipeline (same as notebook)
        transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Create postprocessor
        postprocessor = PostProcessImage(
            max_dets_per_img=-1,  # We limit by confidence threshold instead
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=confidence_threshold,
            to_cpu=False,
        )

        result["transform"] = transform
        result["postprocessor"] = postprocessor
        result["collate_fn"] = collate_fn_api
        result["copy_to_device_fn"] = copy_data_to_device

    return result


def _to_pil(img: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """Convert various image formats to PIL Image."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def _to_numpy(a) -> np.ndarray:
    """Safely convert tensor/array/list to numpy array."""
    if torch is not None and hasattr(a, "detach") and hasattr(a, "cpu"):
        # Convert bfloat16 to float32 before numpy conversion
        t = a.detach()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.cpu().numpy()
    return np.array(a)


def _binarize_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binarize a mask array."""
    if mask.dtype == bool:
        return mask
    if np.issubdtype(mask.dtype, np.floating):
        return mask > threshold
    return mask > 0


def _clamp_bbox_xywh(x: float, y: float, w: float, h: float, W: int, H: int) -> List[int]:
    """Clamp bbox to image bounds and return [x, y, w, h] as integers."""
    x1 = x + w
    y1 = y + h
    x0 = float(np.clip(x, 0, max(W - 1, 0)))
    y0 = float(np.clip(y, 0, max(H - 1, 0)))
    xr = float(np.clip(x1, 0, W))
    yb = float(np.clip(y1, 0, H))

    left = int(np.floor(x0))
    top = int(np.floor(y0))
    right = int(np.ceil(xr))
    bottom = int(np.ceil(yb))

    # Ensure minimum size of 1x1
    if right <= left:
        right = min(W, left + 1)
    if bottom <= top:
        bottom = min(H, top + 1)

    return [left, top, right - left, bottom - top]


def _bboxes_from_masks(
    masks: Union[np.ndarray, Sequence], image_size: Tuple[int, int], threshold: float = 0.5
) -> List[List[int]]:
    """
    Compute tight XYWH bboxes for each mask.

    Args:
        masks: 2D mask, stack of masks (N,H,W), list/tuple of masks, or tensor equivalents.
        image_size: (W, H) of the source image.
        threshold: Threshold for binarizing float masks.

    Returns:
        List of [x, y, w, h] ints, one per non-empty mask.
    """
    W, H = image_size

    # Normalize to a list of 2D numpy arrays
    masks_list: List[np.ndarray] = []
    if isinstance(masks, (list, tuple)):
        for m in masks:
            m_np = _to_numpy(m)
            if m_np.ndim > 2:
                m_np = np.squeeze(m_np)
            masks_list.append(m_np)
    else:
        m_np = _to_numpy(masks)
        if m_np.ndim == 3:  # (N,H,W)
            for k in range(m_np.shape[0]):
                masks_list.append(m_np[k])
        else:  # (H,W)
            masks_list.append(np.squeeze(m_np))

    bboxes: List[List[int]] = []
    for m in masks_list:
        m_bin = _binarize_mask(m, threshold)
        if not m_bin.any():
            continue
        ys, xs = np.where(m_bin)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x = float(x_min)
        y = float(y_min)
        w = float(x_max - x_min + 1)
        h = float(y_max - y_min + 1)
        bboxes.append(_clamp_bbox_xywh(x, y, w, h, W, H))
    return bboxes


def _extract_bboxes_from_inference_state(
    inference_state: dict,
    image_size: Tuple[int, int],
) -> List[List[int]]:
    """
    Extract bounding boxes from SAM3 inference_state.

    Args:
        inference_state: The state dict returned by Sam3Processor after text prompt.
        image_size: (width, height) of the image.

    Returns:
        List of [x, y, w, h] integer boxes sorted by score (highest first).
    """
    W, H = image_size
    
    # SAM3 provides boxes directly in xyxy format
    boxes = inference_state.get("boxes")
    scores = inference_state.get("scores")
    
    if boxes is None:
        return []
    
    # Convert to numpy
    boxes_np = _to_numpy(boxes)
    
    # Sort by scores (descending) if available
    if scores is not None:
        scores_np = _to_numpy(scores).flatten()
        if len(scores_np) == len(boxes_np):
            idx = np.argsort(-scores_np, kind="stable")
            boxes_np = boxes_np[idx]
    
    # Convert xyxy to xywh and clamp to image bounds
    bboxes: List[List[int]] = []
    for box in boxes_np:
        x1, y1, x2, y2 = box
        x = max(0, min(int(round(x1)), W - 1))
        y = max(0, min(int(round(y1)), H - 1))
        x2_clamped = max(0, min(int(round(x2)), W))
        y2_clamped = max(0, min(int(round(y2)), H))
        w = x2_clamped - x
        h = y2_clamped - y
        if w > 0 and h > 0:
            bboxes.append([x, y, w, h])
    
    return bboxes


def _compute_mask_areas(masks: np.ndarray) -> np.ndarray:
    """Compute area of each mask for sorting."""
    if masks.ndim == 2:
        return np.array([masks.sum()])
    return np.array([m.sum() for m in masks])


def _create_datapoint_with_text_prompt(pil_image: Image.Image, text_query: str, query_id: int) -> Any:
    """
    Create a SAM3 datapoint with a text prompt for batched inference.
    
    Args:
        pil_image: PIL Image to process.
        text_query: Text prompt for the object to detect.
        query_id: Unique ID for tracking this query in results.
    
    Returns:
        Datapoint object ready for transformation.
    """
    from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, InferenceMetadata
    from sam3.train.data.sam3_image_dataset import Image as SAMImage
    
    w, h = pil_image.size  # PIL returns (width, height)
    
    datapoint = Datapoint(find_queries=[], images=[])
    # SAMImage.size expects (height, width)
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]
    
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],  # unused for inference
            is_exhaustive=True,  # unused for inference
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                # original_size expects (height, width)
                original_size=[h, w],
                object_id=0,
                frame_index=0,
            )
        )
    )
    
    return datapoint


def _extract_bboxes_from_postprocessed_result(
    result: Dict[str, Any],
    image_size: Tuple[int, int],
    topn: int = 10,
) -> List[List[int]]:
    """
    Extract bounding boxes from postprocessed SAM3 result.
    
    Args:
        result: Dict with 'boxes', 'scores', optionally 'masks'.
        image_size: (width, height) of the image.
        topn: Maximum number of boxes to return.
    
    Returns:
        List of [x, y, w, h] integer boxes.
    """
    W, H = image_size
    
    boxes = result.get("boxes")
    scores = result.get("scores")
    
    if boxes is None or len(boxes) == 0:
        return []
    
    # Convert to numpy
    boxes_np = _to_numpy(boxes)
    if boxes_np.ndim == 1:
        boxes_np = boxes_np.reshape(-1, 4)
    
    # Sort by scores (descending) if available
    if scores is not None:
        scores_np = _to_numpy(scores).flatten()
        if len(scores_np) == len(boxes_np):
            idx = np.argsort(-scores_np, kind="stable")
            boxes_np = boxes_np[idx]
    
    # Convert xyxy to xywh and clamp to image bounds
    bboxes: List[List[int]] = []
    for box in boxes_np[:topn]:
        x1, y1, x2, y2 = box
        x = max(0, min(int(round(x1)), W - 1))
        y = max(0, min(int(round(y1)), H - 1))
        x2_clamped = max(0, min(int(round(x2)), W))
        y2_clamped = max(0, min(int(round(y2)), H))
        w = x2_clamped - x
        h = y2_clamped - y
        if w > 0 and h > 0:
            bboxes.append([x, y, w, h])
    
    return bboxes


def predict_bboxes_for_tag_sam3(
    model_dict: Dict[str, Any],
    images: Union[str, Image.Image, np.ndarray, Sequence[Union[str, Image.Image, np.ndarray]]],
    tag: str,
    topn: int = 10,
) -> List[List[List[int]]]:
    """
    Predict bounding boxes for a tag over a list of images using SAM3.
    Falls back to per-image processing if batch APIs unavailable.

    Args:
        model_dict: Dict from load_sam3() containing model and processing utilities.
        images: Single image or sequence of image paths, PIL Images, or numpy arrays.
        tag: Text prompt for the object name (e.g., "apple").
        topn: Maximum number of boxes to return per image.

    Returns:
        A list of length len(images). Each element is a list of [x, y, w, h] integer bboxes
        for the corresponding image; can be empty if nothing is detected.
    """
    # Handle legacy tuple format for backward compatibility
    if isinstance(model_dict, tuple):
        model, Sam3Processor, confidence_threshold = model_dict
        model_dict = {
            "model": model,
            "confidence_threshold": confidence_threshold,
            "batch_supported": False,
        }
    
    # Handle single image input
    if isinstance(images, (str, Image.Image, np.ndarray)):
        images = [images]
    
    images_pil = [_to_pil(im) for im in images]
    
    # If batch processing not available, fall back to per-image processing
    if not model_dict.get("batch_supported", False):
        return _predict_bboxes_per_image(model_dict, images_pil, tag, topn)
    
    # Use true batch processing
    return _predict_bboxes_batched_internal(model_dict, images_pil, tag, topn, batch_size=len(images_pil))


def _predict_bboxes_per_image(
    model_dict: Dict[str, Any],
    images_pil: List[Image.Image],
    tag: str,
    topn: int,
) -> List[List[List[int]]]:
    """Fallback per-image processing using Sam3Processor."""
    from sam3.model.sam3_image_processor import Sam3Processor
    
    model = model_dict["model"]
    confidence_threshold = model_dict.get("confidence_threshold", 0.5)
    
    all_bboxes: List[List[List[int]]] = []
    
    for img in images_pil:
        try:
            processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
            inference_state = processor.set_image(img)
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(state=inference_state, prompt=tag)
            
            width, height = img.size
            img_bbs = _extract_bboxes_from_inference_state(inference_state, (width, height))
            
            if topn > 0:
                img_bbs = img_bbs[:topn]
            
            all_bboxes.append(img_bbs)
            
        except Exception as e:
            print(f"SAM3 prediction failed for image: {e}")
            all_bboxes.append([])
    
    return all_bboxes


def _predict_bboxes_batched_internal(
    model_dict: Dict[str, Any],
    images_pil: List[Image.Image],
    tag: str,
    topn: int,
    batch_size: int,
    debug: bool = False,
) -> List[List[List[int]]]:
    """
    True batched inference using SAM3's collate and forward APIs.
    
    Args:
        debug: If True, print detailed debugging information.
    """
    import gc
    
    model = model_dict["model"]
    transform = model_dict["transform"]
    postprocessor = model_dict["postprocessor"]
    collate_fn = model_dict["collate_fn"]
    copy_to_device_fn = model_dict["copy_to_device_fn"]
    device = model_dict.get("device", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    all_bboxes: List[List[List[int]]] = [[] for _ in range(len(images_pil))]
    image_sizes = [img.size for img in images_pil]  # (width, height) per image
    
    if debug:
        print(f"[DEBUG] _predict_bboxes_batched_internal called with {len(images_pil)} images, batch_size={batch_size}")
    
    # Process in batches
    for batch_start in range(0, len(images_pil), batch_size):
        batch_end = min(batch_start + batch_size, len(images_pil))
        batch_images = images_pil[batch_start:batch_end]
        batch_sizes = image_sizes[batch_start:batch_end]
        
        if debug:
            print(f"[DEBUG] Processing batch {batch_start}-{batch_end}")
        
        try:
            # Create datapoints for this batch
            datapoints = []
            query_id_to_batch_idx = {}
            
            for i, img in enumerate(batch_images):
                query_id = batch_start + i + 1  # Unique ID per image
                datapoint = _create_datapoint_with_text_prompt(img, tag, query_id)
                datapoint = transform(datapoint)
                datapoints.append(datapoint)
                query_id_to_batch_idx[query_id] = batch_start + i
            
            if debug:
                print(f"[DEBUG] Created {len(datapoints)} datapoints, query_ids: {list(query_id_to_batch_idx.keys())}")
            
            # Collate batch
            batch = collate_fn(datapoints, dict_key="dummy")["dummy"]
            batch = copy_to_device_fn(batch, device, non_blocking=True)
            
            if debug:
                print(f"[DEBUG] Batch type: {type(batch)}")
                if hasattr(batch, 'find_metadatas'):
                    print(f"[DEBUG] find_metadatas: {batch.find_metadatas}")
            
            # Forward pass (with autocast for bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)
            
            if debug:
                print(f"[DEBUG] Model output keys: {output.keys() if hasattr(output, 'keys') else type(output)}")
                # Check output contents
                if hasattr(output, 'keys'):
                    for k in list(output.keys())[:5]:
                        v = output[k]
                        if hasattr(v, 'shape'):
                            print(f"[DEBUG]   {k}: shape={v.shape}")
                        else:
                            print(f"[DEBUG]   {k}: type={type(v)}")
            
            # Post-process results
            processed_results = postprocessor.process_results(output, batch.find_metadatas)
            
            if debug:
                print(f"[DEBUG] Processed results keys: {list(processed_results.keys())}")
                for qid, res in processed_results.items():
                    boxes = res.get("boxes")
                    scores = res.get("scores")
                    print(f"[DEBUG]   Query {qid}: boxes={boxes.shape if hasattr(boxes, 'shape') else 'None'}, scores={scores.shape if hasattr(scores, 'shape') else 'None'}")
            
            # Extract bboxes for each image
            for query_id, result in processed_results.items():
                if query_id in query_id_to_batch_idx:
                    global_idx = query_id_to_batch_idx[query_id]
                    local_idx = global_idx - batch_start
                    img_size = batch_sizes[local_idx]
                    bboxes = _extract_bboxes_from_postprocessed_result(result, img_size, topn)
                    all_bboxes[global_idx] = bboxes
                    if debug:
                        print(f"[DEBUG]   Query {query_id} -> image {global_idx}: {len(bboxes)} boxes")
            
            # Clean up GPU memory after each batch
            del batch, output, processed_results, datapoints
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                    
        except Exception as e:
            error_msg = str(e)
            is_oom = "out of memory" in error_msg.lower() or "CUDA" in error_msg
            
            if is_oom and batch_size > 1:
                print(f"SAM3 batch OOM with batch_size={len(batch_images)}. Retrying with batch_size=1...")
            else:
                print(f"SAM3 batch prediction failed: {e}. Falling back to per-image processing for this batch.")
            
            if debug:
                import traceback
                traceback.print_exc()
            
            # Clean up before fallback
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Fallback for this batch - process one at a time
            for i, img in enumerate(batch_images):
                try:
                    from sam3.model.sam3_image_processor import Sam3Processor
                    
                    # Clear memory before each image
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    processor = Sam3Processor(model, confidence_threshold=model_dict.get("confidence_threshold", 0.5))
                    inference_state = processor.set_image(img)
                    processor.reset_all_prompts(inference_state)
                    inference_state = processor.set_text_prompt(state=inference_state, prompt=tag)
                    
                    width, height = img.size
                    img_bbs = _extract_bboxes_from_inference_state(inference_state, (width, height))
                    if topn > 0:
                        img_bbs = img_bbs[:topn]
                    all_bboxes[batch_start + i] = img_bbs
                    
                    # Clean up after each image
                    del processor, inference_state
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e2:
                    print(f"SAM3 fallback prediction also failed: {e2}")
                    all_bboxes[batch_start + i] = []
    
    return all_bboxes


def predict_bboxes_for_tag_sam3_batched(
    model_dict: Dict[str, Any],
    images: Sequence[Union[str, Image.Image, np.ndarray]],
    tag: str,
    batch_size: int = 8,
    topn: int = 10,
    debug: bool = False,
) -> List[List[List[int]]]:
    """
    Batched prediction using SAM3's native batch processing APIs.
    
    This function now performs TRUE batch inference using SAM3's collate_fn_api,
    significantly improving throughput compared to per-image processing.

    Args:
        model_dict: Dict from load_sam3() containing model and batch processing utilities.
        images: Sequence of image paths, PIL Images, or numpy arrays.
        tag: Text prompt for the object name.
        batch_size: Number of images to process in each batch.
        topn: Maximum boxes per image.
        debug: If True, print detailed debugging information.

    Returns:
        List parallel to images, each element is a list of [x, y, w, h] boxes.
    """
    # Handle legacy tuple format for backward compatibility
    if isinstance(model_dict, tuple):
        model, Sam3Processor, confidence_threshold = model_dict
        model_dict = {
            "model": model,
            "confidence_threshold": confidence_threshold,
            "batch_supported": False,
        }
    
    images_pil = [_to_pil(im) for im in images]
    
    # If batch processing not available, fall back to per-image
    if not model_dict.get("batch_supported", False):
        print(f"SAM3 batch APIs not available. Processing {len(images)} images one at a time.")
        return _predict_bboxes_per_image(model_dict, images_pil, tag, topn)
    
    # True batch processing
    return _predict_bboxes_batched_internal(model_dict, images_pil, tag, topn, batch_size, debug=debug)


__all__ = [
    "load_sam3",
    "predict_bboxes_for_tag_sam3",
    "predict_bboxes_for_tag_sam3_batched",
]
