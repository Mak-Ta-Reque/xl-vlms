import argparse
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw

from datasets import get_dataset_loader
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from helpers.utils import (clear_forward_hooks, clear_hooks_variables,
                           compute_time_left, set_seed, setup_hooks,
                           update_dict_of_list)
from models import get_model_class
from models.image_text_model import ImageTextModel
from helpers.logger import log_num_transformer_layers


# ---------------------------------------------------------------------------
# Background masking helper — replaces pixels outside the segment mask
# ---------------------------------------------------------------------------

def _apply_background_masking(
    crop_img: Image.Image,
    crop_mask_np,  # bool ndarray (H, W)
    method: str = "gaussian_blur",
    blur_radius: int = 15,
) -> Image.Image:
    """Apply background masking to a cropped image.

    Args:
        crop_img:      PIL RGB image (the tight-bbox crop).
        crop_mask_np:  Boolean numpy array, True = foreground (keep sharp).
        method:        One of ``'gaussian_blur'``, ``'ns'`` (Navier-Stokes
                       inpainting), ``'telea'`` (Telea inpainting),
                       ``'noisy_linear'`` (sparse linear interpolation),
                       or ``'blackout'`` (zero-fill).
        blur_radius:   Radius for Gaussian blur *or* inpainting radius.

    Returns:
        PIL RGB image with background masked according to *method*.
    """
    import numpy as np
    from PIL import ImageFilter

    crop_mask_pil = Image.fromarray(
        crop_mask_np.astype("uint8") * 255, mode="L"
    )

    if method == "gaussian_blur":
        blurred = crop_img.filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        )
        return Image.composite(crop_img, blurred, crop_mask_pil)

    elif method in ("ns", "telea"):
        # OpenCV Navier-Stokes or Telea inpainting
        import cv2

        img_array = np.array(crop_img)  # RGB uint8 (H, W, 3)
        # Inpainting mask: 255 where we want to inpaint (i.e. outside fg)
        inpaint_mask = (~crop_mask_np).astype("uint8") * 255
        flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
        # cv2.inpaint expects BGR input
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(
            img_bgr, inpaint_mask,
            inpaintRadius=max(3, blur_radius // 5),
            flags=flag,
        )
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        # Composite: keep original foreground pixels, use inpainted for bg
        inpainted_pil = Image.fromarray(inpainted_rgb)
        return Image.composite(crop_img, inpainted_pil, crop_mask_pil)

    elif method == "noisy_linear":
        # Noisy linear imputation (Rong et al. / ROAD framework)
        from imputers import noisy_linear_inpaint

        inpainted_pil = noisy_linear_inpaint(crop_img, crop_mask_np, noise=0.01)
        # Composite: keep original fg sharp, use imputed values for bg
        return Image.composite(crop_img, inpainted_pil, crop_mask_pil)

    elif method == "blackout":
        black = Image.new("RGB", crop_img.size, (0, 0, 0))
        return Image.composite(crop_img, black, crop_mask_pil)

    else:
        # Unknown method — fall back to Gaussian blur
        blurred = crop_img.filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        )
        return Image.composite(crop_img, blurred, crop_mask_pil)


def _apply_boundary_smoothing(
    image: Image.Image,
    mask_np,  # bool ndarray (H, W)
    boundary_pixels: int,
) -> Image.Image:
    """Smooth the mask-edge band to reduce silhouette leakage."""
    if boundary_pixels <= 0:
        return image

    import numpy as np
    from PIL import ImageFilter
    from scipy.ndimage import binary_dilation, binary_erosion

    struct = np.ones((2 * boundary_pixels + 1, 2 * boundary_pixels + 1), dtype=bool)
    outer = binary_dilation(mask_np, structure=struct, iterations=1)
    inner = binary_erosion(mask_np, structure=struct, iterations=1)
    boundary_band = np.logical_and(outer, np.logical_not(inner))
    if not boundary_band.any():
        return image

    src = np.array(image, dtype=np.float32)
    blurred = np.array(
        image.filter(ImageFilter.GaussianBlur(radius=max(1, boundary_pixels // 2))),
        dtype=np.float32,
    )

    alpha = np.zeros(mask_np.shape, dtype=np.float32)
    alpha[boundary_band] = 1.0
    alpha = np.expand_dims(alpha, axis=-1)

    out = src * (1.0 - alpha) + blurred * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _run_inpainting_step(
    crop_img: Image.Image,
    crop_mask_np,  # bool ndarray (H, W)
    method: str,
    blur_radius: int,
    boundary_pixels: int,
) -> Image.Image:
    """Single inpainting entry point used by save_features mask pipeline."""
    if blur_radius > 0 or method != "gaussian_blur":
        vlm_img = _apply_background_masking(
            crop_img,
            crop_mask_np,
            method=method,
            blur_radius=blur_radius,
        )
    else:
        vlm_img = crop_img

    return _apply_boundary_smoothing(vlm_img, crop_mask_np, boundary_pixels)


def _semanticsegments_sam3_fg_only_mode() -> bool:
    return (
        os.environ.get("CROP_MODE", "").strip().lower() == "semanticsegments_sam3"
        and int(os.environ.get("POSITIVE_NEGATIVE_SEGMENT", os.environ.get("POSITIVE_NEGATVE_SEGMENT", "0"))) == 0
    )

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
            model.config.use_cache = False
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
        toi_str = str(toi).strip() if toi is not None else ""
        if toi is not None and getattr(args, "prompt_template", None) in ["cgdl", "yn"]:
            texts = [t.replace("[concept]", toi_str) if isinstance(t, str) else t for t in texts]
        item["text"] = texts
        image_paths = item["image"] if isinstance(item["image"], list) else [item["image"]]
        img_ids = item.get("img_id", None)
        crop_locations = item.get("bbox", None)
        image_sizes = item.get("image_size", None)
        seg_mask_rles = item.get("seg_mask_rle", None)
        is_concept_flags = item.get("is_concept", None)
        patch_sizes = item.get("patch_size", None)

        # Load blur radius from env (default from .env MASK_BLUR_RADIUS)
        _blur_radius = int(os.environ.get("MASK_BLUR_RADIUS", "10"))

        # Shared masking method for FG and BG views.
        _inpainting_method = os.environ.get("INPAINTING_METHOD", "gaussian_blur")
        _semantic_fg_only_mode = _semanticsegments_sam3_fg_only_mode()

        # Debug overlays are generated after crops.json is written in preprocessing/crops_to_json.py.
        _debug_save = False
        _debug_dir = None

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

        def _ensure_per_image_str_list(values, n_imgs):
            if values is None:
                return [None] * n_imgs
            if isinstance(values, str):
                return [values] * n_imgs
            if isinstance(values, (list, tuple)) and len(values) == n_imgs:
                return [None if v is None else str(v) for v in values]
            return [None] * n_imgs
        
        per_image_img_ids = _ensure_per_image_str_list(img_ids, len(image_paths))
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
        if _semantic_fg_only_mode:
            per_image_is_concept = [True] * len(image_paths)

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

        # Shared context/boundary pixels for both FG and BG views.
        _context_pixels = int(os.environ.get("MASK_CONTEXT_PIXELS", "0"))
        _boundary_pixels = _context_pixels

        for idx in range(len(image_paths)):
            img_ref = image_paths[idx]
            sample_id = per_image_img_ids[idx] or f"batch{i}_idx{idx}"
            sample_id = os.path.basename(str(sample_id)).replace(os.sep, "_")
            bbox = per_image_bboxes[idx]
            mask_rle = per_image_mask_rles[idx]
            _is_concept = per_image_is_concept[idx]
            _ps = per_image_patch_sizes[idx]
            debug_overlay_bbox = None

            # ================================================================
            # MASK-CENTRIC PATH (preferred): use segmentation mask directly
            #
            # Pipeline:
            #   1. Decode RLE mask
            #   2. Find tight bounding box of the mask
            #   3. Expand bbox by MASK_CONTEXT_PIXELS (fg) or
            #      MASK_CONTEXT_PIXELS_BG (bg) for spatial context
            #   4. Crop image + mask to expanded bbox
            #   5. Inpaint outside mask WITHIN the crop
            #   6. Resize crop to patch_size × patch_size
            #   7. Pass resized crop to VLM
            #   8. Save debug images when DEBUG_SAVE_VLM_INPUTS=1
            # ================================================================
            if mask_rle is not None and isinstance(mask_rle, dict):
                # Select masking parameters based on fg / bg role
                _eff_blur     = _blur_radius
                _eff_method   = _inpainting_method

                if logger and idx == 0:
                    logger.info(f"[mask-centric] batch {i}, idx {idx}: is_concept={_is_concept}, mask_rle keys={list(mask_rle.keys())}, blur={_eff_blur}, method={_eff_method}, ctx={_context_pixels}, boundary={_boundary_pixels}, patch={_ps}, debug={_debug_save}")
                img = Image.open(img_ref).convert("RGB") if not isinstance(img_ref, Image.Image) else img_ref
                source_size = img.size

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
                resize_scale_x = (W / float(source_size[0])) if source_size and source_size[0] else 1.0
                resize_scale_y = (H / float(source_size[1])) if source_size and source_size[1] else 1.0

                try:
                    from mask_utils import decode_mask_rle
                    import numpy as np

                    mask_np = decode_mask_rle(mask_rle)  # bool (H, W)
                    # Mask polarity is already normalised in
                    # crops_to_json.py (_normalize_mask_polarity /
                    # positive_negative_segment).  No alteration here —
                    # save_features uses masks exactly as produced.

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

                    # Step 3: Expand bbox by context + boundary pixels
                    # so that dilated FG mask still fits inside the crop.
                    _crop_pad = max(max(_context_pixels, 0), _boundary_pixels)
                    cx1 = _clip(mx1 - _crop_pad, 0, W)
                    cy1 = _clip(my1 - _crop_pad, 0, H)
                    cx2 = _clip(mx2 + _crop_pad, 0, W)
                    cy2 = _clip(my2 + _crop_pad, 0, H)
                    debug_overlay_bbox = (int(cx1), int(cy1), int(cx2), int(cy2))

                    # Step 4: Crop image and mask to expanded bbox
                    crop_img = img.crop((cx1, cy1, cx2, cy2))
                    crop_mask = mask_np[cy1:cy2, cx1:cx2]

                    # Step 4b: Apply the shared boundary-width adjustment.
                    # FG (is_concept=True)  → dilate: expand the sharp
                    #     foreground region so a ring of real surrounding
                    #     pixels is kept — gives the VLM spatial context.
                    # BG (is_concept=False) → erode: shrink the sharp
                    #     background region inward so boundary pixels
                    #     near the object edge are removed — prevents
                    #     the object silhouette from leaking into the
                    #     background view.
                    # The same width is also used to randomize pixels
                    # around the final mask edge to weaken boundary leakage.
                    if _boundary_pixels > 0:
                        _struct = np.ones(
                            (2 * _boundary_pixels + 1,
                             2 * _boundary_pixels + 1), dtype=bool,
                        )
                        if _is_concept:
                            from scipy.ndimage import binary_dilation
                            crop_mask = binary_dilation(
                                crop_mask, structure=_struct, iterations=1,
                            ).astype(bool)
                        else:
                            from scipy.ndimage import binary_erosion
                            eroded = binary_erosion(
                                crop_mask, structure=_struct, iterations=1,
                            )
                            # Guard: if erosion empties the mask keep original
                            if eroded is not None and eroded.any():
                                crop_mask = eroded.astype(bool)

                    # Step 5: Run the standalone inpainting step.
                    vlm_img = _run_inpainting_step(
                        crop_img,
                        crop_mask,
                        method=_eff_method,
                        blur_radius=_eff_blur,
                        boundary_pixels=_boundary_pixels,
                    )

                    # Step 6: Resize to patch_size keeping aspect ratio,
                    # pad remaining space with white pixels.
                    if _ps is not None and _ps > 0:
                        _resample = getattr(
                            getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC
                        )
                        cw, ch = vlm_img.size
                        scale = min(_ps / cw, _ps / ch)
                        new_w = max(1, int(round(cw * scale)))
                        new_h = max(1, int(round(ch * scale)))
                        vlm_img = vlm_img.resize((new_w, new_h), resample=_resample)
                        if new_w != _ps or new_h != _ps:
                            padded = Image.new("RGB", (_ps, _ps), (255, 255, 255))
                            padded.paste(vlm_img, ((_ps - new_w) // 2, (_ps - new_h) // 2))
                            vlm_img = padded

                except Exception as _mask_exc:
                    # Mask decode failed — fall back to full image (no blur)
                    if logger:
                        logger.warning(f"[mask-centric] Mask decode FAILED for idx {idx}: {_mask_exc}")
                    vlm_img = img

                # Step 8: Debug save — crop rectangle overlay + binary mask
                if _debug_save and _debug_dir:
                    try:
                        _tag = getattr(args, "token_of_interest", "unknown")
                        _tag_dir = os.path.join(_debug_dir, str(_tag))
                        os.makedirs(_tag_dir, exist_ok=True)
                        # Label: fg (concept) vs bg (background) for clarity
                        _role = "fg" if _is_concept else "bg"
                        overlay = img.copy()
                        draw = ImageDraw.Draw(overlay)
                        if debug_overlay_bbox is not None:
                            bx1, by1, bx2, by2 = debug_overlay_bbox
                            line_w = max(2, min(6, int(round(min(W, H) * 0.004))))
                            draw.rectangle([bx1, by1, bx2, by2], outline=(255, 0, 0), width=line_w)
                            _bbox_suffix = f"{bx1}_{by1}_{bx2}_{by2}"
                        else:
                            _bbox_suffix = "nobox"
                        overlay.save(os.path.join(
                            _tag_dir, f"{sample_id}_{idx}_{_role}_{_bbox_suffix}_crop_overlay.jpg"
                        ), quality=90)
                        # Save the raw binary mask (cropped region) so we can inspect it
                        try:
                            mask_vis = Image.fromarray(
                                crop_mask.astype("uint8") * 255, mode="L"
                            )
                            mask_vis.save(os.path.join(
                                _tag_dir, f"{sample_id}_{idx}_{_role}_{_bbox_suffix}_mask_binary.png"
                            ))
                        except Exception:
                            pass
                        # Save a metadata text file with masking info
                        try:
                            with open(os.path.join(
                                _tag_dir, f"{sample_id}_{idx}_{_role}_{_bbox_suffix}_mask_method.txt"
                            ), "w") as _mf:
                                _mf.write(f"method={_eff_method}\n")
                                _mf.write(f"blur_radius={_eff_blur}\n")
                                _mf.write(f"context_pixels={_context_pixels}\n")
                                _mf.write(f"boundary_pixels={_boundary_pixels}\n")
                                _mf.write(f"is_concept={_is_concept}\n")
                                _mf.write(f"role={_role}\n")
                                _mf.write(f"source_size={source_size}\n")
                                _mf.write(f"resized_size={(W, H)}\n")
                                _mf.write(f"resize_scale_x={resize_scale_x:.6f}\n")
                                _mf.write(f"resize_scale_y={resize_scale_y:.6f}\n")
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
                source_size = img.size

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
                resize_scale_x = (W / float(source_size[0])) if source_size and source_size[0] else 1.0
                resize_scale_y = (H / float(source_size[1])) if source_size and source_size[1] else 1.0
                x1, y1, x2, y2 = bbox
                x1 = int(_clip(x1, 0, W - 1))
                y1 = int(_clip(y1, 0, H - 1))
                x2 = int(_clip(x2, x1 + 1, W))
                y2 = int(_clip(y2, y1 + 1, H))
                debug_overlay_bbox = (x1, y1, x2, y2)
                crop_img = img.crop((x1, y1, x2, y2))

                if _ps is not None and _ps > 0:
                    _resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
                    cw, ch = crop_img.size
                    scale = min(_ps / cw, _ps / ch)
                    new_w = max(1, int(round(cw * scale)))
                    new_h = max(1, int(round(ch * scale)))
                    crop_img = crop_img.resize((new_w, new_h), resample=_resample)
                    if new_w != _ps or new_h != _ps:
                        padded = Image.new("RGB", (_ps, _ps), (255, 255, 255))
                        padded.paste(crop_img, ((_ps - new_w) // 2, (_ps - new_h) // 2))
                        crop_img = padded

                if _debug_save and _debug_dir:
                    try:
                        _tag = getattr(args, "token_of_interest", "unknown")
                        _tag_dir = os.path.join(_debug_dir, str(_tag))
                        os.makedirs(_tag_dir, exist_ok=True)
                        overlay = img.copy()
                        draw = ImageDraw.Draw(overlay)
                        bx1, by1, bx2, by2 = debug_overlay_bbox
                        line_w = max(2, min(6, int(round(min(W, H) * 0.004))))
                        draw.rectangle([bx1, by1, bx2, by2], outline=(255, 0, 0), width=line_w)
                        overlay.save(os.path.join(
                            _tag_dir, f"{sample_id}_{idx}_{bx1}_{by1}_{bx2}_{by2}_crop_overlay.jpg"
                        ), quality=90)
                        try:
                            with open(os.path.join(
                                _tag_dir, f"{sample_id}_{idx}_{bx1}_{by1}_{bx2}_{by2}_crop_overlay.txt"
                            ), "w") as _mf:
                                _mf.write(f"source_size={source_size}\n")
                                _mf.write(f"resized_size={(W, H)}\n")
                                _mf.write(f"resize_scale_x={resize_scale_x:.6f}\n")
                                _mf.write(f"resize_scale_y={resize_scale_y:.6f}\n")
                                _mf.write(f"bbox={(bx1, by1, bx2, by2)}\n")
                        except Exception:
                            pass
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
            gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=False)
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

        item["concept_name"] = toi_str
        if args.generation_mode:
            # `out` holds generated token ids only in generation mode; in
            # teacher-forcing mode it is logits and cannot be decoded.
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
            item["model_predictions"] = model_class.get_tokenizer().batch_decode(
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

    from device_utils import get_device_config  # type: ignore
    device_config = get_device_config(args.device)
    device = device_config.primary_device
    logger.info(f"Device config: {device_config.raw} -> primary={device}, gpu_ids={device_config.gpu_ids}")

    model_class = get_model_class(
        args.model_name_or_path,
        args.processor_name,
        device=device,
        logger=logger,
        args=args,
        device_config=device_config,
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
        base_save_filename = getattr(args, "save_filename", None)

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
            if base_save_filename:
                args.save_filename = f"{base_save_filename}_{key}"
            else:
                args.save_filename = f"token_of_interest_concept_split_{key}"
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