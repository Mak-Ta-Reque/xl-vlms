
Pythonconda  env: xlvlm-v1               /mnt/abka03/.conda/envs/xlvlm-v1 for testing 
# Plan: Add Segmentation Masks + Tag-Free YOLO to Pipeline

**TL;DR:** Add YOLOv11-seg as a tag-free detector that runs once per image and returns all bboxes + masks (VLM assigns tags later). Also extract the already-computed masks from LangSAM and SAM3 (they currently discard them). Thread RLE-encoded masks through all pipeline stages. Blur background outside masks before VLM cropping. Only random crops are mask-free.

## Technical Diagram

```
                        ┌─────────── .env ──────────────┐
                        │ OBJECT_DETECTOR=seg       │
                        │ MASK_BLUR_RADIUS=15            │
                        └───────────┬───────────────────┘
                                    ▼
  Step 2: crops_to_json.py
  ┌────────────────────────────────────────────────────────────────┐
  │ seg: run ONCE per image → all (class, bbox, mask)              │
  │           assign ALL detections to every tag (tag-free)       │
  │           VLM will later determine which tag fits             │
  │                                                                │
  │ langsam/sam3: run per tag (existing) but NOW also return masks │
  │                                                                │
  │ random: no masks (seg_mask = null)                             │
  │                                                                │
  │ JSON per image:                                                │
  │   detections_xyxy: [[x1,y1,x2,y2], ...]                      │
  │   detections_masks_rle: [{size,counts}, ...]   ← NEW          │
  │   random_crops: [[x1,y1,x2,y2], ...]          (no masks)     │
  └───────────────────────┬────────────────────────────────────────┘
                          ▼
  Step 3: save_features.py  (+ image_text_dataset.py, utils.py)
  ┌────────────────────────────────────────────────────────────────┐
  │ Dataset item: bbox + seg_mask (RLE-decoded to numpy)           │
  │ If seg_mask: GaussianBlur background → crop by bbox → VLM     │
  │ .pth keys: {..., seg_mask}  ← NEW                             │
  └───────────────────────┬────────────────────────────────────────┘
                          ▼
  Step 4: analyse → multimodal_grounding → combine_concepts
  ┌────────────────────────────────────────────────────────────────┐
  │ grounding_dict: + image_grounding_masks  ← NEW                │
  │ combined_data:  + image_grounding_masks  ← NEW                │
  │ All saved into combined_concept_<method>_raw.pth               │
  └────────────────────────────────────────────────────────────────┘
```

## Steps

### 1. Create `src/yolo_utils.py` — tag-free YOLO detector

- Use `ultralytics` with `YOLO("yolo11x-seg")` (latest YOLOv11 instance segmentation).
- `load_yolo(device, confidence_threshold)` → returns model dict.
- `predict_all_objects(model_dict, images, batch_size, topn)` → returns `List[List[Dict]]` per image, each dict: `{"class_name": str, "bbox_xywh": [x,y,w,h], "mask": np.ndarray(H,W,bool), "confidence": float}`. Tag-free: returns **every** detected object.
- `encode_masks_rle(detections)` → converts numpy masks to pycocotools RLE dicts for JSON storage.
- No tag filtering — all YOLO detections are assigned to whichever tag is being processed. The VLM in Step 5 determines semantic relevance.

### 2. Update `src/langsam_utils.py` — return masks alongside bboxes

- In `_extract_bboxes_from_result()` (line 200), also return the numpy mask array from `result.get("masks")` (already read at line 202 but discarded).
- Change return type of `predict_bboxes_for_tag_batched()` from `List[List[bbox]]` to `List[List[Tuple[bbox, np.ndarray]]]` — each detection now pairs a bbox with its binary mask.
- Same change for `predict_bboxes_for_tag()` single-image variant.

### 3. Update `src/sam3_utils.py` — return masks alongside bboxes

- In `_extract_bboxes_from_inference_state()` (line 305), also read `inference_state["masks"]` (boolean tensors, already computed but ignored).
- In `_extract_bboxes_from_postprocessed_result()` (line 405), also read `result["masks"]` (boolean tensors, already in result dict but ignored).
- Change return types of `predict_bboxes_for_tag_sam3_batched()` to match LangSAM's new signature: `List[List[Tuple[bbox, np.ndarray]]]`.

### 4. Update `preprocessing/crops_to_json.py` — integrate YOLO + store RLE masks

- Add `_load_yolo_model()` and `run_yolo_all_objects()` (near line 93).
- Update `run_detector_batched()` (line 166): change return type to `List[List[Tuple[bbox, Optional[mask]]]]`. For `"yolo_seg"`, `"langsam"`, `"sam3"` — all now return `(bbox, mask)` pairs.
- **YOLO flow in `process_json_mapping_to_json()` / `concept_process_json_mapping_to_json()`:** When `detector == "yolo_seg"`, collect all unique images across all tags, run YOLO **once**, cache `yolo_cache[img_path] = [(bbox, mask), ...]`. Inside the per-tag loop, assign **all** cached detections to the current tag (tag-free assignment).
- Update `_record_for_image()` (line 477) to accept `detections_masks_rle: Optional[List[dict]]` and write `"detections_masks_rle"` in JSON. Encode masks using `pycocotools.mask.encode()`. Random crops have no mask entries.
- Add `"yolo_seg"` to `--object_detector` CLI choices (line 1071).
- store masks RLE in a sidecar .npz file instead of inline in JSON, or is inline RLE. 
### 5. Update `src/datasets/image_text_dataset.py` — decode RLE masks into dataset items

- In `JSONDataset.create_dataset()` (line 300), when iterating `detections_xyxy` boxes, also read the parallel `detections_masks_rle[i]` if present, decode with `pycocotools.mask.decode()` → binary numpy `(H,W)`.
- Add `"seg_mask": decoded_mask_or_None` to each item dict. For `random_crops` items: `"seg_mask": None`.

### 6. Update `src/save_features.py` — blur outside mask before VLM crop

- In `inference()` (lines 60–113), after opening/resizing the image, read `seg_mask = item.get("seg_mask", None)`. If present:
  - Resize mask to match image size.
  - Create blurred copy via `PIL.ImageFilter.GaussianBlur(radius=MASK_BLUR_RADIUS)` (default 15, from env).
  - Composite: `Image.composite(original, blurred, mask_as_pil)` → sharp object, blurred background.
  - Then crop by bbox as before.
- Read `MASK_BLUR_RADIUS` from `os.environ.get("MASK_BLUR_RADIUS", "15")`.

### 7. Update `src/helpers/utils.py` — save masks in `.pth` feature files

- In `hooks_postprocessing()` (line 874), add `"seg_mask"` to `data_keys`: change `data_keys = ["hidden_states", "image", "model_predictions", "bbox"]` to also include `"seg_mask"`.

### 8. Update `src/analysis/multimodal_grounding.py` — masks in grounding dict

- In `get_multimodal_grounding()` (line 142), extract `concept_seg_masks = metadata.get("seg_mask", [])` alongside `concept_bbox`. Build `all_concept_seg_masks` in the per-concept loop. Store as `grounding_dict["image_grounding_masks"]`.
- In `refine_ground_activations()` at `feature_decomposition.py` line 99, propagate `image_grounding_masks` from `concept_dict` into the results dict.

### 9. Update `src/combine_concepts.py` — merge masks in combined `.pth`

- Add `'image_grounding_masks': []` to `combined_data` init (line 72).
- In the per-eligible-index loop (line 95), append `model_data.get('image_grounding_masks', [])[idx]`.
- Same in `combine_concepts_()` (line 210).

### 10. Update `.env` and `scripts/run_full_pipeline.py` — wire everything up

- Add to `.env`: `OBJECT_DETECTOR="yolo_seg"` (replacing `"sam3"`) and `MASK_BLUR_RADIUS=15`.
- In `PipelineConfig.__init__()` (line 87), update comment to `# 'none', 'langsam', 'sam3', 'yolo_seg'`, and add `self.mask_blur_radius = self._get_int("MASK_BLUR_RADIUS", 15)`.
- In `step_2_build_crops_json()` (line 288), change `if config.object_detector in ("langsam", "sam3"):` to `if config.object_detector in ("langsam", "sam3", "yolo_seg"):`.
- Add `ultralytics>=8.3` and `pycocotools>=2.0` to `requirements.txt`.

## Decisions Taken

1. **YOLO is tag-free:** Assigns all detections to every tag. The VLM explainer (Step 5) handles semantic relevance naturally.
2. **LangSAM/SAM3 also return masks now:** Both models already compute masks internally but currently discard them. We extract and propagate them.
3. **Only random crops are mask-free:** All detector-based crops (YOLO, LangSAM, SAM3) carry segmentation masks.
4. **Background blur (not crop-mask):** Pixels outside the mask are Gaussian-blurred (radius=15 from `MASK_BLUR_RADIUS` env var), not zeroed. This avoids distribution shift while still suppressing background.
5. **RLE encoding in JSON:** Masks stored as pycocotools RLE dicts in the crops JSON for compactness. Decoded to numpy in the dataset loader.
6. **No backward compatibility needed:** All detector modes produce masks. Only random crops have `null` masks.

## Files Changed (Summary)

| File | Action |
|------|--------|
| `src/yolo_utils.py` | **NEW** — YOLO detector module |
| `src/langsam_utils.py` | MODIFY — return masks alongside bboxes |
| `src/sam3_utils.py` | MODIFY — return masks alongside bboxes |
| `preprocessing/crops_to_json.py` | MODIFY — integrate YOLO, store RLE masks |
| `src/datasets/image_text_dataset.py` | MODIFY — decode RLE masks into items |
| `src/save_features.py` | MODIFY — blur outside mask before crop |
| `src/helpers/utils.py` | MODIFY — add `seg_mask` to saved `.pth` keys |
| `src/analysis/multimodal_grounding.py` | MODIFY — propagate masks in grounding dict |
| `src/analysis/feature_decomposition.py` | MODIFY — propagate masks in refine step |
| `src/combine_concepts.py` | MODIFY — merge masks in combined output |
| `scripts/run_full_pipeline.py` | MODIFY — wire `yolo_seg` + `MASK_BLUR_RADIUS` |
| `.env` | MODIFY — add `OBJECT_DETECTOR`, `MASK_BLUR_RADIUS` |
| `requirements.txt` | MODIFY — add `ultralytics`, `pycocotools` |
