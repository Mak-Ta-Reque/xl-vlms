import argparse
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image

from datasets import get_dataset_loader
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from helpers.utils import (clear_forward_hooks, clear_hooks_variables,
                           compute_time_left, set_seed, setup_hooks,
                           update_dict_of_list)
from models import get_model_class
from models.image_text_model import ImageTextModel
from helpers.logger import log_num_transformer_layers

@torch.no_grad()
def inference(
    loader: Callable,
    model_class: ImageTextModel,
    hook_return_functions: Optional[List[Callable]],
    device: torch.device,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Tuple[List[Dict[str, Any]], List[bool]]:

    num_iterations = len(loader)
    hook_data = {}
    model = model_class.get_model()
    # Disable KV cache globally (may reduce memory, slight slowdown in autoregressive generation)
    try:
        if hasattr(model, "config") and getattr(model.config, "use_cache", True):
            model.config.use_cache = True
            if logger:
                logger.info("Disabled model.config.use_cache")
    except Exception as e:
        if logger:
            logger.warning(f"Could not disable KV cache: {e}")
    log_num_transformer_layers(model, model_name=args.model_name_or_path)

    start_time = time.time()
    for i, item in enumerate(loader):
        # Batchify: gather all texts and images in the batch
        texts = item["text"] if isinstance(item["text"], list) else [item["text"]]
        # Replace placeholder [concept] with --token_of_interest value if provided
        toi = getattr(args, "token_of_interest", None)
        if toi is not None and "cgdl" in getattr(args, "prompt_template", None):
            toi_str = str(toi).strip()
            texts = [t.replace("[concept]", toi_str) if isinstance(t, str) else t for t in texts]
        item["text"] = texts
        image_paths = item["image"] if isinstance(item["image"], list) else [item["image"]]
        crop_locations = item.get("bbox", None)
        image_sizes = item.get("image_size", None)
        seg_mask_rles = item.get("seg_mask_rle", None)
        is_concept_flags = item.get("is_concept", None)
        patch_sizes = item.get("patch_size", None)

        # Load blur radius from env (default 15)
        _blur_radius = int(os.environ.get("MASK_BLUR_RADIUS", "15"))

        # Debug: save VLM input images (blurred full + final crop) to disk
        _debug_save = os.environ.get("DEBUG_SAVE_VLM_INPUTS", "0") == "1"
        _debug_dir = None
        if _debug_save:
            _debug_dir = os.path.join(
                os.environ.get("OUTPUT_DIR", "outputs"), "debug_vlm_inputs"
            )
            os.makedirs(_debug_dir, exist_ok=True)

        # Build per-sample inputs. If bboxes provided, crop and pass PIL Image; else pass paths.
        per_sample_inputs: List[Dict[str, Any]] = []

        def _clip(val, lo, hi):
            return max(lo, min(hi, val))

        def _ensure_per_image_bbox_list(crops, n_imgs):
            # Accept: None, single 4-elt tuple/list, list of 4-elt per image
            if crops is None:
                return [None] * n_imgs
            if isinstance(crops, (list, tuple)) and len(crops) == 4 and all(isinstance(v, (int, float)) for v in crops):
                return [list(crops)] * n_imgs
            # if it's a list of per-image bboxes
            if isinstance(crops, (list, tuple)) and len(crops) == n_imgs:
                return list(crops)
            # Fallback: broadcast None
            return [None] * n_imgs

        def _ensure_per_image_size_list(sizes, n_imgs):
            # Accept: None, single [w,h], list of [w,h] per image
            if sizes is None:
                return [None] * n_imgs
            if isinstance(sizes, (list, tuple)) and len(sizes) == 2 and all(isinstance(v, (int, float)) for v in sizes):
                return [list(sizes)] * n_imgs
            if isinstance(sizes, (list, tuple)) and len(sizes) == n_imgs:
                return list(sizes)
            return [None] * n_imgs
        
        per_image_bboxes = _ensure_per_image_bbox_list(crop_locations, len(image_paths))
        per_image_sizes = _ensure_per_image_size_list(image_sizes, len(image_paths))

        # Per-image seg mask RLE (can be None, a single dict, or a list parallel to images)
        def _ensure_per_image_mask_rle(masks, n_imgs):
            if masks is None:
                return [None] * n_imgs
            if isinstance(masks, dict):
                return [masks] * n_imgs
            if isinstance(masks, (list, tuple)) and len(masks) == n_imgs:
                return list(masks)
            return [None] * n_imgs

        per_image_mask_rles = _ensure_per_image_mask_rle(seg_mask_rles, len(image_paths))

        # Per-image is_concept flag (bool)
        def _ensure_per_image_is_concept(flags, n_imgs):
            if flags is None:
                return [False] * n_imgs
            if isinstance(flags, bool):
                return [flags] * n_imgs
            if isinstance(flags, (list, tuple)) and len(flags) == n_imgs:
                return [bool(f) for f in flags]
            return [False] * n_imgs

        per_image_is_concept = _ensure_per_image_is_concept(is_concept_flags, len(image_paths))

        # Per-image patch_size (scalar int or list of ints parallel to images)
        def _ensure_per_image_patch_size(ps, n_imgs):
            if ps is None:
                _env_ps = int(os.environ.get("PATCH_SIZE", "0"))
                return [_env_ps if _env_ps > 0 else None] * n_imgs
            if isinstance(ps, (int, float)):
                return [int(ps)] * n_imgs
            if isinstance(ps, (list, tuple)) and len(ps) == n_imgs:
                return [int(v) if v is not None else None for v in ps]
            if isinstance(ps, (list, tuple)) and len(ps) == 1:
                return [int(ps[0])] * n_imgs
            return [None] * n_imgs

        per_image_patch_sizes = _ensure_per_image_patch_size(patch_sizes, len(image_paths))

        # Context pixels beyond the mask boundary to keep for spatial context
        _context_pixels = int(os.environ.get("MASK_CONTEXT_PIXELS", "10"))

        for idx in range(len(image_paths)):
            img_ref = image_paths[idx]
            bbox = per_image_bboxes[idx]
            mask_rle = per_image_mask_rles[idx]
            _is_concept = per_image_is_concept[idx]
            _ps = per_image_patch_sizes[idx]

            # ================================================================
            # MASK-CENTRIC PATH (preferred): use segmentation mask directly
            #
            # Pipeline:
            #   1. Decode RLE mask
            #   2. Find tight bounding box of the mask
            #   3. Expand bbox by MASK_CONTEXT_PIXELS for spatial context
            #   4. Crop image + mask to expanded bbox
            #   5. Blur outside mask WITHIN the crop (sharp segment, blurred bg)
            #   6. Resize crop to patch_size × patch_size
            #   7. Pass resized crop to VLM
            #   8. Save debug images when DEBUG_SAVE_VLM_INPUTS=1
            # ================================================================
            if mask_rle is not None and isinstance(mask_rle, dict):
                if logger and idx == 0:
                    logger.info(f"[mask-centric] batch {i}, idx {idx}: mask_rle keys={list(mask_rle.keys())}, blur={_blur_radius}, ctx={_context_pixels}, patch={_ps}, debug={_debug_save}")
                img = Image.open(img_ref).convert("RGB") if not isinstance(img_ref, Image.Image) else img_ref

                # Resize to virtual size if specified
                target_size = per_image_sizes[idx]
                if isinstance(target_size, (list, tuple)) and len(target_size) >= 2:
                    try:
                        tw, th = int(target_size[0]), int(target_size[1])
                        if tw > 0 and th > 0 and img.size != (tw, th):
                            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
                            img = img.resize((tw, th), resample=resample)
                    except Exception:
                        pass

                W, H = img.size

                try:
                    from mask_utils import decode_mask_rle
                    import numpy as np
                    from PIL import ImageFilter

                    mask_np = decode_mask_rle(mask_rle)  # bool (H, W)
                    # Safety: invert non-concept masks that cover >50% of
                    # the image (background masks from point-grid auto
                    # segmentation).  Concept masks keep their original
                    # polarity because text-prompted detection is correct
                    # even for large segments (ground, sky, water, etc.).
                    if not _is_concept:
                        _total_px = mask_np.shape[0] * mask_np.shape[1]
                        if _total_px > 0 and mask_np.sum() / _total_px > 0.50:
                            mask_np = ~mask_np
                    # Resize mask to image size if needed
                    if mask_np.shape != (H, W):
                        mask_pil = Image.fromarray(mask_np.astype("uint8") * 255, mode="L")
                        mask_pil = mask_pil.resize((W, H), resample=Image.NEAREST)
                        mask_np = np.array(mask_pil) > 127

                    # Step 2: Find tight bounding box of the mask
                    ys, xs = np.where(mask_np)
                    if len(ys) == 0:
                        # Empty mask — fall back to full image
                        raise ValueError("Mask is empty (0 foreground pixels)")

                    mx1, my1 = int(xs.min()), int(ys.min())
                    mx2, my2 = int(xs.max()) + 1, int(ys.max()) + 1

                    # Step 3: Expand bbox by context buffer
                    cx1 = _clip(mx1 - _context_pixels, 0, W)
                    cy1 = _clip(my1 - _context_pixels, 0, H)
                    cx2 = _clip(mx2 + _context_pixels, 0, W)
                    cy2 = _clip(my2 + _context_pixels, 0, H)

                    # Step 4: Crop image and mask to expanded bbox
                    crop_img = img.crop((cx1, cy1, cx2, cy2))
                    crop_mask = mask_np[cy1:cy2, cx1:cx2]

                    # Step 5: Blur outside mask within the crop
                    if _blur_radius > 0:
                        blurred_crop = crop_img.filter(
                            ImageFilter.GaussianBlur(radius=_blur_radius)
                        )
                        crop_mask_pil = Image.fromarray(
                            crop_mask.astype("uint8") * 255, mode="L"
                        )
                        vlm_img = Image.composite(crop_img, blurred_crop, crop_mask_pil)
                    else:
                        vlm_img = crop_img

                    # Step 6: Resize to patch_size × patch_size
                    if _ps is not None and _ps > 0:
                        _resample = getattr(
                            getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC
                        )
                        vlm_img = vlm_img.resize((_ps, _ps), resample=_resample)

                except Exception as _mask_exc:
                    # Mask decode failed — fall back to full image (no blur)
                    if logger:
                        logger.warning(f"[mask-centric] Mask decode FAILED for idx {idx}: {_mask_exc}")
                    vlm_img = img

                # Step 8: Debug save — VLM input crop + binary mask
                if _debug_save and _debug_dir:
                    try:
                        _img_name = os.path.splitext(os.path.basename(
                            img_ref if isinstance(img_ref, str) else f"batch{i}_idx{idx}"
                        ))[0]
                        _tag = getattr(args, "token_of_interest", "unknown")
                        _tag_dir = os.path.join(_debug_dir, str(_tag))
                        os.makedirs(_tag_dir, exist_ok=True)
                        vlm_img.save(os.path.join(
                            _tag_dir, f"{_img_name}_mask_vlm_input.jpg"
                        ), quality=90)
                        # Save the raw binary mask (cropped region) so we can inspect it
                        try:
                            mask_vis = Image.fromarray(
                                crop_mask.astype("uint8") * 255, mode="L"
                            )
                            mask_vis.save(os.path.join(
                                _tag_dir, f"{_img_name}_mask_binary.png"
                            ))
                        except Exception:
                            pass
                    except Exception:
                        pass

                per_sample_inputs.append(
                    model_class.preprocessor(
                        instruction=texts[idx],
                        image_file=vlm_img,
                        response="",
                        generation_mode=args.generation_mode,
                    )
                )

            # ================================================================
            # LEGACY BBOX PATH (fallback when no mask available)
            # ================================================================
            elif bbox is not None:
                img = Image.open(img_ref).convert("RGB") if not isinstance(img_ref, Image.Image) else img_ref

                target_size = per_image_sizes[idx]
                if isinstance(target_size, (list, tuple)) and len(target_size) >= 2:
                    try:
                        tw, th = int(target_size[0]), int(target_size[1])
                        if tw > 0 and th > 0 and img.size != (tw, th):
                            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
                            img = img.resize((tw, th), resample=resample)
                    except Exception:
                        pass

                W, H = img.size
                x1, y1, x2, y2 = bbox
                x1 = int(_clip(x1, 0, W - 1))
                y1 = int(_clip(y1, 0, H - 1))
                x2 = int(_clip(x2, x1 + 1, W))
                y2 = int(_clip(y2, y1 + 1, H))
                crop_img = img.crop((x1, y1, x2, y2))

                if _ps is not None and _ps > 0:
                    _resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
                    crop_img = crop_img.resize((_ps, _ps), resample=_resample)

                if _debug_save and _debug_dir:
                    try:
                        _img_name = os.path.splitext(os.path.basename(
                            img_ref if isinstance(img_ref, str) else f"batch{i}_idx{idx}"
                        ))[0]
                        _tag = getattr(args, "token_of_interest", "unknown")
                        _tag_dir = os.path.join(_debug_dir, str(_tag))
                        os.makedirs(_tag_dir, exist_ok=True)
                        crop_img.save(os.path.join(
                            _tag_dir, f"{_img_name}_crop_vlm_input.jpg"
                        ), quality=90)
                    except Exception:
                        pass

                per_sample_inputs.append(
                    model_class.preprocessor(
                        instruction=texts[idx],
                        image_file=crop_img,
                        response="",
                        generation_mode=args.generation_mode,
                    )
                )
            else:
                # no bbox -> default behavior
                per_sample_inputs.append(
                    model_class.preprocessor(
                        instruction=texts[idx],
                        image_file= image_paths[idx],
                        response="",
                        generation_mode=args.generation_mode,
                    )
                )
        #count the number image token per sample
        image_token_id = model_class.processor_.image_token_id
        num_image_tokens_per_sample = []
        for sample in per_sample_inputs:
            input_ids = sample["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()[0] if input_ids.ndim == 2 and input_ids.shape[0] == 1 else input_ids.tolist()
            count = sum(1 for id in input_ids if id == image_token_id)
            num_image_tokens_per_sample.append(count)

        
        # Collate per-sample dicts into a batch dict
        def _collate_inputs(per_sample_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
            if len(per_sample_inputs) == 1:
                return per_sample_inputs[0]
            batched = {}
            keys = per_sample_inputs[0].keys()
            pad_id = getattr(model_class.get_tokenizer(), "pad_token_id", 0) or 0
            for k in keys:
                vals = [d[k] for d in per_sample_inputs]
                first = vals[0]
                if isinstance(first, torch.Tensor):
                    if k in ("input_ids", "attention_mask"):
                        seqs = []
                        for v in vals:
                            if v.ndim == 2 and v.shape[0] == 1:
                                seqs.append(v.squeeze(0))
                            else:
                                seqs.append(v)
                        padding_value = pad_id if k == "input_ids" else 0
                        batched[k] = torch.nn.utils.rnn.pad_sequence(
                            seqs, batch_first=True, padding_side='left', padding_value=padding_value
                        ) 
                    else:
                        arrs = []
                        for v in vals:
                            if v.ndim == 0:
                                v = v.unsqueeze(0)
                            arrs.append(v)
                        try:
                            batched[k] = torch.cat(arrs, dim=0)
                        except Exception:
                            batched[k] = vals
                else:
                    batched[k] = vals
            return batched

        inputs = _collate_inputs(per_sample_inputs)

        if args.generation_mode:
            # Explicitly pass use_cache=False for models honoring this kwarg
            gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
            out = model.generate(
                **inputs, **gen_kwargs
            )
        else:
            out = model(**inputs).logits

        # Debatch outputs: store per-sample outputs as a list for downstream
        if isinstance(out, torch.Tensor) and out.dim() >= 1 and out.size(0) > 1:
            item["model_output"] = [out[b] for b in range(out.size(0))]
        else:
            item["model_output"] = out
        # Keep using `out` locally for subsequent computations


        # Compute per-sample input lengths (no attention mask): first pad or full length
        pad_id = getattr(model_class.get_tokenizer(), "pad_token_id", 0) or 0
        input_ids_tensor = inputs["input_ids"]
        if input_ids_tensor.ndim == 1:
            input_ids_tensor = input_ids_tensor.unsqueeze(0)
        B, L = input_ids_tensor.shape
        input_lens: List[int] = []
        for b in range(B):
            row = input_ids_tensor[b]
            input_lens.append(row.shape[0])
        # Slice generated tokens per sample
        generated_ids = [out[b, input_lens[b]:] for b in range(out.size(0))]
        # Debatch to plain Python lists of token ids for portability
        model_generated_output_list = [t.tolist() if torch.is_tensor(t) else list(t) for t in generated_ids]
        item["model_generated_output"] = model_generated_output_list
        item["model_predictions"] = model_class.get_tokenizer( ).batch_decode(
            model_generated_output_list, skip_special_tokens=True
        )

        if hook_return_functions is not None:
            for func in hook_return_functions:
                if func is not None:
                    hook_output = func(**item)
                    if hook_output:
                        item.update(hook_output)

        hook_data = update_dict_of_list(item, hook_data)
        clear_hooks_variables()
        # Iteration-end cleanup to reduce GPU memory usage
        try:
            del out  # model outputs
        except Exception:
            pass
        try:
            del inputs  # batched inputs
        except Exception:
            pass
        # Drop potentially large GPU-backed fields from item now that data is aggregated
        try:
            if isinstance(item, dict) and "model_output" in item:
                item["model_output"] = None
        except Exception:
            pass
        if torch.cuda.is_available():
            pass
            #torch.cuda.empty_cache()
        if (i + 1) % 100 == 0:
            time_left = compute_time_left(start_time, i, num_iterations)
            logger.info(
                f"Iteration: {i}/{num_iterations},  Estimated time left: {time_left:.2f} mins"
            )
    return hook_data


def main():
    """Main entry point for save_features script."""
    args = get_arguments()

    logger = setup_logger(log_file=os.path.join(args.save_dir, f"logs.log"))

    set_seed(args.seed)

    logger.info(f"Loading model: {args.model_name_or_path}")
    log_args(args, logger)

    device = torch.device(args.device)

    model_class = get_model_class(
        args.model_name_or_path,
        args.processor_name,
        device=device,
        logger=logger,
        args=args,
    )

    hook_return_functions, hook_postprocessing_functions = setup_hooks(
        model=model_class.model_,
        modules_to_hook=args.modules_to_hook,
        hook_names=args.hook_names,
        tokenizer=model_class.get_tokenizer(),
        logger=logger,
        args=args,
    )
    loader = get_dataset_loader(
        dataset_name=args.dataset_name, logger=logger, args=args
    )
    # if loader is a list run inference on each and aggregate the results and update the args accordingly
    if args.dataset_name == "json_crop_map":
        all_hook_data = {}
        
        for key, ld in loader.items():


            hook_return_functions, hook_postprocessing_functions = setup_hooks(
                model=model_class.model_,
                modules_to_hook=args.modules_to_hook,
                hook_names=args.hook_names,
                tokenizer=model_class.get_tokenizer(),
                logger=logger,
                args=args,
            )
            logger.info(f"Running inference on dataset split: {key}")
            #add args"--token_of_interest",  value as key
            #--save_filename 
            args.token_of_interest = key
            args.save_filename = f"qwen2_patched_image_cat_token_of_interest_concept_generation_split_train_{key}"
            hook_data = inference(
                loader=ld,
                model_class=model_class,
                device=device,
                hook_return_functions=hook_return_functions,
                logger=logger,
                args=args,
            )
            clear_forward_hooks(model_class.model_)
            if hook_postprocessing_functions is not None:
                for func in hook_postprocessing_functions:
                    if func is not None:
                        func(data=hook_data, args=args, logger=logger)

            
    
    else:

        hook_data = inference(
            loader=loader,
            model_class=model_class,
            device=device,
            hook_return_functions=hook_return_functions,
            logger=logger,
            args=args,
        )

        clear_forward_hooks(model_class.model_)
        if hook_postprocessing_functions is not None:
            for func in hook_postprocessing_functions:
                if func is not None:
                    func(data=hook_data, args=args, logger=logger)


if __name__ == "__main__":
    main()