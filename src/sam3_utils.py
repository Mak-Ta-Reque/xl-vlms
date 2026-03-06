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
            try:
                from device_utils import get_device_config
                _dc = get_device_config(str(device))
                target_device = _dc.primary_device
                use_cuda = target_device.type == "cuda"
            except Exception:
                dev = str(device).lower()
                if dev == "cpu":
                    use_cuda = False
                    target_device = torch.device("cpu")
                elif dev.startswith("cuda"):
                    target_device = torch.device(dev)

    # Enable TF32 and autocast for better performance on Ampere GPUs
    if use_cuda and torch is not None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build model (enable SAM1-style point prompt for auto mask generation)
    model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)

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


def _extract_bboxes_and_masks_from_inference_state(
    inference_state: dict,
    image_size: Tuple[int, int],
) -> List[tuple]:
    """
    Extract (bbox_xywh, binary_mask) tuples from SAM3 inference_state.

    Returns:
        List of (bbox_xywh: List[int], mask: np.ndarray|None) tuples,
        sorted by score (highest first).
    """
    W, H = image_size

    boxes = inference_state.get("boxes")
    masks = inference_state.get("masks")
    scores = inference_state.get("scores")

    if boxes is None:
        return []

    boxes_np = _to_numpy(boxes)

    masks_np: Optional[np.ndarray] = None
    if masks is not None:
        masks_np = _to_numpy(masks)
        if masks_np.ndim == 4:  # (N, 1, H, W)
            masks_np = masks_np.squeeze(1)

    if scores is not None:
        scores_np = _to_numpy(scores).flatten()
        if len(scores_np) == len(boxes_np):
            idx = np.argsort(-scores_np, kind="stable")
            boxes_np = boxes_np[idx]
            if masks_np is not None and len(masks_np) == len(scores_np):
                masks_np = masks_np[idx]

    pairs: List[tuple] = []
    for i, box in enumerate(boxes_np):
        x1, y1, x2, y2 = box
        x = max(0, min(int(round(x1)), W - 1))
        y = max(0, min(int(round(y1)), H - 1))
        x2c = max(0, min(int(round(x2)), W))
        y2c = max(0, min(int(round(y2)), H))
        w = x2c - x
        h = y2c - y
        if w > 0 and h > 0:
            m = None
            if masks_np is not None and i < len(masks_np):
                m = _binarize_mask(masks_np[i])
            pairs.append(([x, y, w, h], m))

    return pairs


def _extract_bboxes_from_inference_state(
    inference_state: dict,
    image_size: Tuple[int, int],
) -> List[List[int]]:
    """Legacy wrapper: returns only bboxes."""
    pairs = _extract_bboxes_and_masks_from_inference_state(inference_state, image_size)
    return [bb for bb, _m in pairs]


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


def _extract_bboxes_and_masks_from_postprocessed_result(
    result: Dict[str, Any],
    image_size: Tuple[int, int],
    topn: int = 10,
) -> List[tuple]:
    """
    Extract (bbox_xywh, binary_mask) tuples from postprocessed SAM3 result.

    Returns:
        List of (bbox_xywh: List[int], mask: np.ndarray|None) tuples.
    """
    W, H = image_size

    boxes = result.get("boxes")
    masks = result.get("masks")
    scores = result.get("scores")

    if boxes is None or len(boxes) == 0:
        return []

    boxes_np = _to_numpy(boxes)
    if boxes_np.ndim == 1:
        boxes_np = boxes_np.reshape(-1, 4)

    masks_np: Optional[np.ndarray] = None
    if masks is not None:
        masks_np = _to_numpy(masks)
        if masks_np.ndim == 4:  # (N, 1, H, W)
            masks_np = masks_np.squeeze(1)

    if scores is not None:
        scores_np = _to_numpy(scores).flatten()
        if len(scores_np) == len(boxes_np):
            idx = np.argsort(-scores_np, kind="stable")
            boxes_np = boxes_np[idx]
            if masks_np is not None and len(masks_np) == len(scores_np):
                masks_np = masks_np[idx]

    pairs: List[tuple] = []
    for i, box in enumerate(boxes_np[:topn]):
        x1, y1, x2, y2 = box
        x = max(0, min(int(round(x1)), W - 1))
        y = max(0, min(int(round(y1)), H - 1))
        x2c = max(0, min(int(round(x2)), W))
        y2c = max(0, min(int(round(y2)), H))
        w = x2c - x
        h = y2c - y
        if w > 0 and h > 0:
            m = None
            if masks_np is not None and i < len(masks_np):
                m = _binarize_mask(masks_np[i])
            pairs.append(([x, y, w, h], m))

    return pairs


def _extract_bboxes_from_postprocessed_result(
    result: Dict[str, Any],
    image_size: Tuple[int, int],
    topn: int = 10,
) -> List[List[int]]:
    """Legacy wrapper: returns only bboxes."""
    pairs = _extract_bboxes_and_masks_from_postprocessed_result(result, image_size, topn)
    return [bb for bb, _m in pairs]


def predict_bboxes_and_masks_for_tag_sam3(
    model_dict: Dict[str, Any],
    images: Union[str, Image.Image, np.ndarray, Sequence[Union[str, Image.Image, np.ndarray]]],
    tag: str,
    topn: int = 10,
) -> List[List[tuple]]:
    """
    Predict (bbox, mask) pairs for a tag over a list of images using SAM3.

    Returns:
        List of length len(images). Each element is a list of
        (bbox_xywh: List[int], mask: np.ndarray|None) tuples.
    """
    if isinstance(model_dict, tuple):
        model, Sam3Processor, confidence_threshold = model_dict
        model_dict = {
            "model": model,
            "confidence_threshold": confidence_threshold,
            "batch_supported": False,
        }

    if isinstance(images, (str, Image.Image, np.ndarray)):
        images = [images]

    images_pil = [_to_pil(im) for im in images]

    if not model_dict.get("batch_supported", False):
        return _predict_bboxes_and_masks_per_image(model_dict, images_pil, tag, topn)

    return _predict_bboxes_and_masks_batched_internal(
        model_dict, images_pil, tag, topn, batch_size=len(images_pil)
    )


def predict_bboxes_for_tag_sam3(
    model_dict: Dict[str, Any],
    images: Union[str, Image.Image, np.ndarray, Sequence[Union[str, Image.Image, np.ndarray]]],
    tag: str,
    topn: int = 10,
) -> List[List[List[int]]]:
    """Legacy wrapper: returns only bboxes."""
    all_pairs = predict_bboxes_and_masks_for_tag_sam3(model_dict, images, tag, topn)
    return [[bb for bb, _m in pairs] for pairs in all_pairs]


def _predict_bboxes_and_masks_per_image(
    model_dict: Dict[str, Any],
    images_pil: List[Image.Image],
    tag: str,
    topn: int,
) -> List[List[tuple]]:
    """Fallback per-image processing returning (bbox, mask) pairs."""
    from sam3.model.sam3_image_processor import Sam3Processor

    model = model_dict["model"]
    confidence_threshold = model_dict.get("confidence_threshold", 0.5)

    all_pairs: List[List[tuple]] = []

    import gc

    for img in images_pil:
        try:
            with torch.no_grad():
                processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
                inference_state = processor.set_image(img)
                processor.reset_all_prompts(inference_state)
                inference_state = processor.set_text_prompt(state=inference_state, prompt=tag)

                width, height = img.size
                pairs = _extract_bboxes_and_masks_from_inference_state(
                    inference_state, (width, height)
                )

                if topn > 0:
                    pairs = pairs[:topn]

                all_pairs.append(pairs)

                del processor, inference_state

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"SAM3 prediction failed for image: {e}")
            all_pairs.append([])
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_pairs


def _predict_bboxes_per_image(
    model_dict: Dict[str, Any],
    images_pil: List[Image.Image],
    tag: str,
    topn: int,
) -> List[List[List[int]]]:
    """Legacy per-image wrapper: returns only bboxes."""
    all_pairs = _predict_bboxes_and_masks_per_image(model_dict, images_pil, tag, topn)
    return [[bb for bb, _m in pairs] for pairs in all_pairs]


def _predict_bboxes_and_masks_batched_internal(
    model_dict: Dict[str, Any],
    images_pil: List[Image.Image],
    tag: str,
    topn: int,
    batch_size: int,
    debug: bool = False,
) -> List[List[tuple]]:
    """
    True batched inference returning (bbox, mask) pairs.
    """
    import gc

    model = model_dict["model"]
    transform = model_dict["transform"]
    postprocessor = model_dict["postprocessor"]
    collate_fn = model_dict["collate_fn"]
    copy_to_device_fn = model_dict["copy_to_device_fn"]
    device = model_dict.get("device", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    all_pairs: List[List[tuple]] = [[] for _ in range(len(images_pil))]
    image_sizes = [img.size for img in images_pil]

    if debug:
        print(f"[DEBUG] _predict_bboxes_and_masks_batched_internal: {len(images_pil)} images, batch_size={batch_size}")

    with torch.no_grad():
     for batch_start in range(0, len(images_pil), batch_size):
      batch_end = min(batch_start + batch_size, len(images_pil))
      batch_images = images_pil[batch_start:batch_end]
      batch_sizes = image_sizes[batch_start:batch_end]

      if debug:
          print(f"[DEBUG] Processing batch {batch_start}-{batch_end}")

      try:
          datapoints = []
          query_id_to_batch_idx = {}

          for i, img in enumerate(batch_images):
              query_id = batch_start + i + 1
              datapoint = _create_datapoint_with_text_prompt(img, tag, query_id)
              datapoint = transform(datapoint)
              datapoints.append(datapoint)
              query_id_to_batch_idx[query_id] = batch_start + i

          if debug:
              print(f"[DEBUG] Created {len(datapoints)} datapoints, query_ids: {list(query_id_to_batch_idx.keys())}")

          batch = collate_fn(datapoints, dict_key="dummy")["dummy"]
          batch = copy_to_device_fn(batch, device, non_blocking=True)

          with torch.autocast("cuda", dtype=torch.bfloat16):
              output = model(batch)

          if debug:
              print(f"[DEBUG] Model output keys: {output.keys() if hasattr(output, 'keys') else type(output)}")

          processed_results = postprocessor.process_results(output, batch.find_metadatas)

          if debug:
              print(f"[DEBUG] Processed results keys: {list(processed_results.keys())}")

          for query_id, result in processed_results.items():
              if query_id in query_id_to_batch_idx:
                  global_idx = query_id_to_batch_idx[query_id]
                  local_idx = global_idx - batch_start
                  img_size = batch_sizes[local_idx]
                  pairs = _extract_bboxes_and_masks_from_postprocessed_result(
                      result, img_size, topn
                  )
                  all_pairs[global_idx] = pairs
                  if debug:
                      print(f"[DEBUG]   Query {query_id} -> image {global_idx}: {len(pairs)} detections")

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
              print(f"SAM3 batch prediction failed: {e}. Falling back to per-image.")

          if debug:
              import traceback
              traceback.print_exc()

          gc.collect()
          if torch.cuda.is_available():
              torch.cuda.empty_cache()

          for i, img in enumerate(batch_images):
              try:
                  from sam3.model.sam3_image_processor import Sam3Processor

                  gc.collect()
                  if torch.cuda.is_available():
                      torch.cuda.empty_cache()

                  processor = Sam3Processor(
                      model,
                      confidence_threshold=model_dict.get("confidence_threshold", 0.5),
                  )
                  inference_state = processor.set_image(img)
                  processor.reset_all_prompts(inference_state)
                  inference_state = processor.set_text_prompt(
                      state=inference_state, prompt=tag
                  )

                  width, height = img.size
                  pairs = _extract_bboxes_and_masks_from_inference_state(
                      inference_state, (width, height)
                  )
                  if topn > 0:
                      pairs = pairs[:topn]
                  all_pairs[batch_start + i] = pairs

                  del processor, inference_state
                  gc.collect()
                  if torch.cuda.is_available():
                      torch.cuda.empty_cache()

              except Exception as e2:
                  print(f"SAM3 fallback prediction also failed: {e2}")
                  all_pairs[batch_start + i] = []

    return all_pairs


def _predict_bboxes_batched_internal(
    model_dict: Dict[str, Any],
    images_pil: List[Image.Image],
    tag: str,
    topn: int,
    batch_size: int,
    debug: bool = False,
) -> List[List[List[int]]]:
    """Legacy batched wrapper: returns only bboxes."""
    all_pairs = _predict_bboxes_and_masks_batched_internal(
        model_dict, images_pil, tag, topn, batch_size, debug
    )
    return [[bb for bb, _m in pairs] for pairs in all_pairs]


def predict_bboxes_and_masks_for_tag_sam3_batched(
    model_dict: Dict[str, Any],
    images: Sequence[Union[str, Image.Image, np.ndarray]],
    tag: str,
    batch_size: int = 8,
    topn: int = 10,
    debug: bool = False,
) -> List[List[tuple]]:
    """
    Batched prediction returning (bbox, mask) pairs using SAM3.

    Returns:
        List parallel to images, each element is a list of
        (bbox_xywh: List[int], mask: np.ndarray|None) tuples.
    """
    if isinstance(model_dict, tuple):
        model, Sam3Processor, confidence_threshold = model_dict
        model_dict = {
            "model": model,
            "confidence_threshold": confidence_threshold,
            "batch_supported": False,
        }

    images_pil = [_to_pil(im) for im in images]

    if not model_dict.get("batch_supported", False):
        print(f"SAM3 batch APIs not available. Processing {len(images)} images one at a time.")
        return _predict_bboxes_and_masks_per_image(model_dict, images_pil, tag, topn)

    return _predict_bboxes_and_masks_batched_internal(
        model_dict, images_pil, tag, topn, batch_size, debug=debug
    )


def predict_bboxes_for_tag_sam3_batched(
    model_dict: Dict[str, Any],
    images: Sequence[Union[str, Image.Image, np.ndarray]],
    tag: str,
    batch_size: int = 8,
    topn: int = 10,
    debug: bool = False,
) -> List[List[List[int]]]:
    """Legacy batched API: returns only bboxes."""
    all_pairs = predict_bboxes_and_masks_for_tag_sam3_batched(
        model_dict, images, tag, batch_size, topn, debug
    )
    return [[bb for bb, _m in pairs] for pairs in all_pairs]


# ---------------------------------------------------------------------------
# Automatic (non-tag) segmentation for SAM3
# ---------------------------------------------------------------------------

_SAM3_GENERIC_PROMPTS = ["object", "thing", "item", "part"]


def predict_all_masks_sam3(
    model_dict: Dict[str, Any],
    images: Sequence[Union[str, Image.Image, np.ndarray]],
    topn: int = 10,
    min_mask_area: int = 100,
    batch_size: int = 8,
    debug: bool = False,
    other_tags: Optional[List[str]] = None,
    exclude_tag: Optional[str] = None,
) -> List[List[tuple]]:
    """Run non-tag segmentation using SAM3 with other concept tags.

    SAM3 is text-prompted and cannot do auto-segmentation with generic
    prompts like "object".  Instead, we iterate through all *other*
    concept tags (from the mapping) and collect their detections.  The
    image backbone is computed once per image and reused across prompts.

    When ``other_tags`` is not supplied the function falls back to the
    generic-prompt approach (kept for backward compat, but produces few
    results).

    Args:
        model_dict: Dict returned by load_sam3().
        images: List of image paths, PIL Images, or numpy arrays.
        topn: Maximum number of masks per image.
        min_mask_area: Discard masks with fewer pixels than this.
        batch_size: Batch size for inference (unused in multi-tag path).
        debug: Enable debug output.
        other_tags: All concept tags from the mapping.  The function
            will iterate through every tag *except* ``exclude_tag``.
        exclude_tag: The current concept tag to skip (already handled
            as the concept mask by the caller).

    Returns:
        List of length len(images). Each element is a list of
        (bbox_xywh: List[int], mask: np.ndarray bool H×W) tuples.
    """
    if model_dict is None:
        model_dict = load_sam3()

    # --- Multi-tag path (preferred): iterate other concept tags ---
    if other_tags:
        tags_to_query = [t for t in other_tags if t != exclude_tag]
        if debug:
            print(f"[predict_all_masks_sam3] multi-tag mode: {len(tags_to_query)} tags (excluding '{exclude_tag}')")
        return _predict_nontag_masks_multitag(
            model_dict, images, tags_to_query,
            topn=topn, min_mask_area=min_mask_area, debug=debug,
        )

    # --- Legacy fallback: generic prompt (rarely produces results) ---
    generic_prompt = _SAM3_GENERIC_PROMPTS[0]
    all_pairs = predict_bboxes_and_masks_for_tag_sam3_batched(
        model_dict, images, tag=generic_prompt,
        batch_size=batch_size, topn=topn, debug=debug,
    )

    # Filter by min_mask_area
    filtered: List[List[tuple]] = []
    for img_pairs in all_pairs:
        kept = []
        for bbox, mask in img_pairs:
            if mask is not None:
                area = int(mask.sum()) if hasattr(mask, 'sum') else 0
                if area < min_mask_area:
                    continue
            kept.append((bbox, mask))
        filtered.append(kept[:topn])

    return filtered


def _predict_nontag_masks_multitag(
    model_dict: Dict[str, Any],
    images: Sequence[Union[str, Image.Image, np.ndarray]],
    tags: List[str],
    topn: int = 10,
    min_mask_area: int = 100,
    debug: bool = False,
) -> List[List[tuple]]:
    """For each image run SAM3 with every tag in *tags*, reusing the
    image backbone.  Collect all resulting (bbox, mask) pairs per image.

    Uses Sam3Processor per-image so ``set_image`` is called once and
    ``set_text_prompt`` is called once per tag (cheap).
    """
    from sam3.model.sam3_image_processor import Sam3Processor

    model = model_dict["model"]
    confidence_threshold = model_dict.get("confidence_threshold", 0.5)

    import gc

    images_pil = [_to_pil(im) for im in images]
    all_results: List[List[tuple]] = [[] for _ in range(len(images_pil))]

    for idx, img in enumerate(images_pil):
        try:
            with torch.no_grad():
                processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
                state = processor.set_image(img)
                width, height = img.size

                for tag in tags:
                    try:
                        processor.reset_all_prompts(state)
                        state = processor.set_text_prompt(state=state, prompt=tag)
                        pairs = _extract_bboxes_and_masks_from_inference_state(
                            state, (width, height),
                        )
                        for bbox, mask in pairs:
                            if mask is not None:
                                area = int(mask.sum()) if hasattr(mask, 'sum') else 0
                                if area < min_mask_area:
                                    continue
                            all_results[idx].append((bbox, mask))
                    except Exception as e:
                        if debug:
                            print(f"[predict_nontag] tag='{tag}' failed for image {idx}: {e}")

                del processor, state

            # Keep only top-N by mask area (largest first)
            all_results[idx].sort(
                key=lambda p: int(p[1].sum()) if p[1] is not None else 0,
                reverse=True,
            )
            all_results[idx] = all_results[idx][:topn]

            if debug:
                print(f"[predict_nontag] image {idx}: {len(all_results[idx])} non-tag masks from {len(tags)} tags")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"SAM3 non-tag prediction failed for image {idx}: {e}")
            all_results[idx] = []
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_results


# ---------------------------------------------------------------------------
# True automatic (prompt-free) mask generation via point-grid
# ---------------------------------------------------------------------------

def predict_auto_masks_sam3(
    model_dict: Dict[str, Any],
    images: Sequence[Union[str, Image.Image, np.ndarray]],
    topn: int = 10,
    min_mask_area: int = 100,
    points_per_side: int = 16,
) -> List[List[tuple]]:
    """Generate masks for all objects without text prompts using a point grid.

    Uses SAM3's interactive predictor (SAM1-style) with a dense grid of
    foreground points.  For each point the model returns up to 3 candidate
    masks; the best (highest IoU score) is kept.  After collecting all
    point masks, NMS-style deduplication removes near-duplicates (IoU > 0.8),
    and the remaining masks are sorted by area (largest first) then
    truncated to *topn*.

    Args:
        model_dict: Dict returned by :func:`load_sam3`.
        images: List of image paths / PIL Images / numpy arrays.
        topn: Maximum number of masks per image to return.
        min_mask_area: Discard masks with fewer pixels than this.
        points_per_side: Grid density (total points = points_per_side²).
            Default 16 → 256 points (was 32 → 1024, too memory-heavy).

    Returns:
        List parallel to *images*.  Each element is a list of
        ``(bbox_xywh, mask_bool_HxW)`` tuples.
    """
    import gc

    model = model_dict["model"]
    confidence_threshold = model_dict.get("confidence_threshold", 0.5)
    images_pil = [_to_pil(im) for im in images]

    all_results: List[List[tuple]] = []

    for img_idx, img in enumerate(images_pil):
        W, H = img.size

        # ------ build uniform point grid ------
        xs = np.linspace(0, W - 1, points_per_side)
        ys = np.linspace(0, H - 1, points_per_side)
        xv, yv = np.meshgrid(xs, ys)
        grid_points = np.stack([xv.ravel(), yv.ravel()], axis=-1)  # (N, 2)

        try:
            from sam3.model.sam3_image_processor import Sam3Processor

            with torch.no_grad():
                processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
                state = processor.set_image(img)

                candidate_masks: List[np.ndarray] = []
                candidate_scores: List[float] = []

                for pt in grid_points:
                    try:
                        point_coords = pt.reshape(1, 2)
                        point_labels = np.array([1])  # foreground

                        masks_out, scores_out, _ = model.predict_inst(
                            state,
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=True,
                        )

                        # masks_out: (C, H, W) tensor/array, scores_out: (C,)
                        masks_np = _to_numpy(masks_out)
                        scores_np = _to_numpy(scores_out).flatten()

                        # Free GPU tensors immediately
                        del masks_out, scores_out

                        if len(scores_np) == 0:
                            continue

                        best_idx = int(np.argmax(scores_np))
                        mask = _binarize_mask(masks_np[best_idx])
                        area = int(mask.sum())
                        del masks_np
                        if area < min_mask_area:
                            continue

                        candidate_masks.append(mask)
                        candidate_scores.append(float(scores_np[best_idx]))
                    except Exception:
                        continue

                # Free the processor and image state before NMS
                del processor, state

            # ------ NMS-style deduplication (IoU > 0.8 → discard lower-score) ------
            if candidate_masks:
                order = np.argsort([-s for s in candidate_scores])
                keep: List[int] = []
                for idx in order:
                    m = candidate_masks[idx]
                    duplicate = False
                    for kept_idx in keep:
                        km = candidate_masks[kept_idx]
                        inter = int((m & km).sum())
                        union = int((m | km).sum())
                        if union > 0 and inter / union > 0.8:
                            duplicate = True
                            break
                    if not duplicate:
                        keep.append(idx)
                        if len(keep) >= topn:
                            break

                # Sort kept masks by area descending
                keep.sort(key=lambda i: int(candidate_masks[i].sum()), reverse=True)

                pairs: List[tuple] = []
                for i in keep:
                    m = candidate_masks[i]
                    ys_m, xs_m = np.where(m)
                    bbox = _clamp_bbox_xywh(
                        float(xs_m.min()), float(ys_m.min()),
                        float(xs_m.max() - xs_m.min() + 1),
                        float(ys_m.max() - ys_m.min() + 1),
                        W, H,
                    )
                    pairs.append((bbox, m))
                all_results.append(pairs)
            else:
                all_results.append([])

            del candidate_masks, candidate_scores
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"SAM3 auto-mask generation failed for image {img_idx}: {e}")
            all_results.append([])
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_results


__all__ = [
    "load_sam3",
    "predict_bboxes_for_tag_sam3",
    "predict_bboxes_for_tag_sam3_batched",
    "predict_bboxes_and_masks_for_tag_sam3",
    "predict_bboxes_and_masks_for_tag_sam3_batched",
    "predict_all_masks_sam3",
    "predict_auto_masks_sam3",
]
