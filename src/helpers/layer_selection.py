"""Per-tag decoder-layer selection via logit lens.

Before feature extraction for a concept tag, sweep a set of language-model
decoder layers on a few of the tag's own regions (the bbox/mask patches that
steps 1-2 stored in crops.json) and pick the layer whose hidden states are
most likely to produce the tag token itself. The score is the *relative*
probability p(tag token) / p(layer top-1 token), which is more comparable
across layers than the raw probability.

The probe inputs are built exactly like the extraction inputs in
``save_features.inference`` (mask tight-bbox crop + background inpainting,
or bbox crop, then patch-size resize), so the selected layer matches what
the extraction will actually see.

Environment variables (all optional; feature is off by default):
    LOGIT_LENS_LAYER_SELECTION  "1" enables per-tag layer selection.
    LOGIT_LENS_MODE             "patch" (default) scores the visual-token
                                positions of a single forward pass;
                                "text" scores the answer-producing positions
                                of a full generation.
    LOGIT_LENS_LAYERS           Layers to sweep: "auto" (default, all
                                language-model decoder layers), a range
                                "0-27", a list "5,10,20", or full dotted
                                module paths.
    LOGIT_LENS_NUM_PATCHES      Number of tag regions to probe (default 8);
                                sampled uniformly from the tag's crops.json
                                entries.

All other knobs (INPAINTING_METHOD, MASK_BLUR_RADIUS, MASK_CONTEXT_PIXELS,
PATCH_SIZE, max_new_tokens, ...) follow the original implementation.

Intermediate results (per-layer scores, probed regions, selected layer) are
written to <save_dir>/logitlens/<tag>/.
"""

import argparse
import os
import re
import zlib
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image

from helpers.logit_lens import (
    resolve_layer_modules,
    score_hidden_state_with_logit_lens,
    write_layer_selection_debug,
)

__all__ = [
    "layer_selection_enabled",
    "discover_decoder_layers",
    "resolve_sweep_layers",
    "get_visual_token_ids",
    "select_layer_for_tag",
]


def layer_selection_enabled() -> bool:
    return os.environ.get("LOGIT_LENS_LAYER_SELECTION", "0").strip() == "1"


def _read_config() -> Dict[str, Any]:
    return {
        "mode": os.environ.get("LOGIT_LENS_MODE", "patch").strip().lower(),
        "layers_spec": os.environ.get("LOGIT_LENS_LAYERS", "auto").strip(),
        "num_patches": max(1, int(os.environ.get("LOGIT_LENS_NUM_PATCHES", "8"))),
    }


def discover_decoder_layers(model: torch.nn.Module) -> List[str]:
    """Return the full module names of the language decoder blocks, in order.

    Matches names ending in ``layers.<idx>`` and prefers those under a
    ``language_model`` path; vision-tower modules are excluded.
    """
    end_pattern = re.compile(r"(?:^|\.)layers\.(\d+)$")
    language, generic = [], []
    for name, _ in model.named_modules():
        match = end_pattern.search(name)
        if not match:
            continue
        if "visual" in name or "vision" in name:
            continue
        entry = (int(match.group(1)), name)
        if "language_model" in name:
            language.append(entry)
        else:
            generic.append(entry)
    chosen = language if language else generic
    chosen.sort(key=lambda item: item[0])
    return [name for _, name in chosen]


def resolve_sweep_layers(
    model: torch.nn.Module,
    spec: str,
    logger: Optional[Callable] = None,
) -> List[str]:
    """Resolve LOGIT_LENS_LAYERS into concrete module names."""
    spec = (spec or "auto").strip()
    if spec.lower() in {"auto", ""}:
        layers = discover_decoder_layers(model)
        if not layers:
            raise ValueError("Could not auto-discover decoder layers for the sweep.")
        return layers
    # Bare range "0-27" -> bracketed form understood by expand_layer_specs.
    if re.fullmatch(r"\d+\s*-\s*\d+", spec):
        spec = f"[{spec}]"
    layers = resolve_layer_modules(model, spec, logger=logger)
    named = dict(model.named_modules())
    missing = [name for name in layers if name not in named]
    if missing:
        raise ValueError(f"LOGIT_LENS_LAYERS entries not found in model: {missing}")
    return layers


def sample_tag_regions(
    dataset: Any,
    num_regions: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], str]:
    """Sample region items (bbox/mask entries from crops.json) for the sweep.

    Samples uniformly from all of the tag's regions (no is_concept filter —
    e.g. sliding-window crops carry no such flag). Returns the sampled items
    and the tag's probe prompt text.
    """
    items = [dataset[i] for i in range(len(dataset))]
    prompt_text = ""
    for item in items:
        text = item.get("text", "")
        prompt_text = text[0] if isinstance(text, list) else text
        if prompt_text:
            break
    pool = items
    if len(pool) > num_regions:
        pool = rng.sample(pool, num_regions)
    return pool, prompt_text


def _clip(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def _resize_to_patch(image: Image.Image, patch: Optional[int]) -> Image.Image:
    """Aspect-preserving resize to patch x patch with white padding (same as
    step 6 of the mask-centric path in ``save_features.inference``)."""
    if not patch or patch <= 0:
        return image
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    width, height = image.size
    scale = min(patch / width, patch / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    image = image.resize((new_w, new_h), resample=resample)
    if new_w != patch or new_h != patch:
        padded = Image.new("RGB", (patch, patch), (255, 255, 255))
        padded.paste(image, ((patch - new_w) // 2, (patch - new_h) // 2))
        image = padded
    return image


def build_region_image(
    item: Dict[str, Any],
    logger: Optional[Callable] = None,
) -> Optional[Tuple[Image.Image, List[int]]]:
    """Build the probe image for one crops.json region item.

    Mirrors the per-sample input construction of ``save_features.inference``:
    mask-centric path (tight mask bbox + MASK_CONTEXT_PIXELS pad + background
    inpainting) when an RLE mask exists, bbox crop otherwise, then the
    PATCH_SIZE resize. Returns (image, [x1, y1, x2, y2]) or None on failure.
    """
    image_path = item.get("image", None)
    if not isinstance(image_path, str) or not image_path:
        return None
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        if logger is not None:
            logger.warning(f"[layer-selection] Could not open {image_path}: {exc}")
        return None

    target_size = item.get("image_size", None)
    if isinstance(target_size, (list, tuple)) and len(target_size) >= 2:
        try:
            tw, th = int(target_size[0]), int(target_size[1])
            if tw > 0 and th > 0 and img.size != (tw, th):
                resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
                img = img.resize((tw, th), resample=resample)
        except Exception:
            pass

    width, height = img.size
    patch = item.get("patch_size", None)
    if not patch:
        env_patch = int(os.environ.get("PATCH_SIZE", "0"))
        patch = env_patch if env_patch > 0 else None
    context_pixels = int(os.environ.get("MASK_CONTEXT_PIXELS", "0"))
    blur_radius = int(os.environ.get("MASK_BLUR_RADIUS", "10"))
    inpainting_method = os.environ.get("INPAINTING_METHOD", "gaussian_blur")

    mask_rle = item.get("seg_mask_rle", None)
    bbox = item.get("bbox", None)

    if isinstance(mask_rle, dict):
        try:
            import numpy as np
            from mask_utils import decode_mask_rle
            # Lazy import to avoid a circular import at module load time
            # (save_features imports this module).
            from save_features import _run_inpainting_step

            mask_np = decode_mask_rle(mask_rle)
            if mask_np.shape != (height, width):
                mask_pil = Image.fromarray(mask_np.astype("uint8") * 255, mode="L")
                mask_pil = mask_pil.resize((width, height), resample=Image.NEAREST)
                mask_np = np.array(mask_pil) > 127

            ys, xs = np.where(mask_np)
            if len(ys) == 0:
                raise ValueError("Mask is empty (0 foreground pixels)")
            x1 = _clip(int(xs.min()) - context_pixels, 0, width)
            y1 = _clip(int(ys.min()) - context_pixels, 0, height)
            x2 = _clip(int(xs.max()) + 1 + context_pixels, 0, width)
            y2 = _clip(int(ys.max()) + 1 + context_pixels, 0, height)
            crop_img = img.crop((x1, y1, x2, y2))
            crop_mask = mask_np[y1:y2, x1:x2]
            if context_pixels > 0 and item.get("is_concept", False):
                try:
                    from scipy.ndimage import binary_dilation
                    struct = np.ones(
                        (2 * context_pixels + 1, 2 * context_pixels + 1), dtype=bool
                    )
                    crop_mask = binary_dilation(
                        crop_mask, structure=struct, iterations=1
                    ).astype(bool)
                except Exception:
                    pass
            region = _run_inpainting_step(
                crop_img,
                crop_mask,
                method=inpainting_method,
                blur_radius=blur_radius,
                boundary_pixels=context_pixels,
            )
            return _resize_to_patch(region, patch), [x1, y1, x2, y2]
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    f"[layer-selection] Mask path failed for {image_path}: {exc}; "
                    "falling back to bbox/full image."
                )

    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = _clip(x1, 0, width - 1)
        y1 = _clip(y1, 0, height - 1)
        x2 = _clip(x2, x1 + 1, width)
        y2 = _clip(y2, y1 + 1, height)
        region = img.crop((x1, y1, x2, y2))
        return _resize_to_patch(region, patch), [x1, y1, x2, y2]

    return _resize_to_patch(img, patch), [0, 0, width, height]


def get_visual_token_ids(model: torch.nn.Module, tokenizer: Any) -> List[int]:
    """Token ids that mark image-patch positions in the input sequence."""
    ids: List[int] = []
    configs = [getattr(model, "config", None)]
    if configs[0] is not None:
        configs.append(getattr(configs[0], "text_config", None))
    for config in configs:
        if config is None:
            continue
        for attr in ("image_token_id", "image_token_index"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value >= 0 and value not in ids:
                ids.append(value)
    if not ids and tokenizer is not None:
        for token in ("<|image_pad|>", "<image>"):
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                continue
            if isinstance(token_id, int) and token_id >= 0 and token_id not in ids:
                unk_id = getattr(tokenizer, "unk_token_id", None)
                if token_id != unk_id:
                    ids.append(token_id)
    return ids


@torch.no_grad()
def select_layer_for_tag(
    model_class: Any,
    tag: str,
    dataset: Any,
    args: argparse.Namespace,
    logger: Optional[Callable] = None,
) -> Optional[Dict[str, Any]]:
    """Sweep decoder layers with the logit lens and pick the best one for *tag*.

    Returns None when disabled or when no usable sample exists; otherwise a
    dict with ``selected_layer`` and a JSON-serializable ``summary``.
    """
    if not layer_selection_enabled():
        return None

    from helpers.utils import clear_forward_hooks, clear_hooks_variables

    config = _read_config()
    model = model_class.get_model()
    tokenizer = model_class.get_tokenizer()
    lm_head = model_class.get_lm_head()
    language_model = model_class.get_language_model()

    # Remove any previously registered feature-extraction hooks so the sweep
    # forwards do not pollute the global HIDDEN_STATES store (and vice versa).
    clear_forward_hooks(model)
    clear_hooks_variables()

    seed = int(getattr(args, "seed", 42)) ^ zlib.crc32(str(tag).encode("utf-8"))
    rng = random.Random(seed)

    regions, prompt_text = sample_tag_regions(dataset, config["num_patches"], rng)
    if not regions:
        if logger is not None:
            logger.warning(f"[layer-selection] No regions found for tag '{tag}'; skipping.")
        return None
    if getattr(args, "prompt_template", None) in ["cgdl", "yn"] and prompt_text:
        prompt_text = prompt_text.replace("[concept]", str(tag))

    layer_names = resolve_sweep_layers(model, config["layers_spec"], logger=logger)

    visual_token_ids = get_visual_token_ids(model, tokenizer)
    if config["mode"] == "patch" and not visual_token_ids:
        if logger is not None:
            logger.warning(
                "[layer-selection] Could not determine visual token ids; "
                "falling back to LOGIT_LENS_MODE=text."
            )
        config["mode"] = "text"

    # Local forward hooks; each capture overwrites so that after generation
    # the stored tensor is the final full-sequence forward (use_cache=False).
    named_modules = dict(model.named_modules())
    captured: Dict[str, torch.Tensor] = {}

    def _make_hook(name: str):
        def hook(module, hook_in, hook_out):
            out = hook_out[0] if isinstance(hook_out, tuple) else hook_out
            captured[name] = out.detach()
        return hook

    handles = [named_modules[name].register_forward_hook(_make_hook(name)) for name in layer_names]

    per_layer_rel: Dict[str, List[float]] = {name: [] for name in layer_names}
    per_layer_abs: Dict[str, List[float]] = {name: [] for name in layer_names}
    region_records: List[Dict[str, Any]] = []

    try:
        for item in regions:
            built = build_region_image(item, logger=logger)
            if built is None:
                continue
            crop, region_bbox = built
            inputs = model_class.preprocessor(
                instruction=prompt_text,
                image_file=crop,
                response="",
                generation_mode=True,
            )
            input_ids = inputs["input_ids"]
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            prompt_len = int(input_ids.shape[1])

            captured.clear()
            record: Dict[str, Any] = {
                "image": item.get("image", None),
                "img_id": item.get("img_id", None),
                "bbox": region_bbox,
                "is_concept": bool(item.get("is_concept", False)),
                "has_mask": isinstance(item.get("seg_mask_rle", None), dict),
                "layer_scores": {},
            }
            if config["mode"] == "patch":
                model(**inputs)
                visual_ids = torch.tensor(visual_token_ids, device=input_ids.device)
                position_mask = torch.isin(input_ids[0], visual_ids)
                positions = torch.nonzero(position_mask, as_tuple=False).squeeze(-1)
                if positions.numel() == 0:
                    if logger is not None:
                        logger.warning(
                            f"[layer-selection] No visual-token positions for "
                            f"{item.get('image', None)}; skipping region."
                        )
                    continue
            else:
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=False,
                )
                record["generated_text"] = tokenizer.batch_decode(
                    generated[:, prompt_len:], skip_special_tokens=True
                )[0]
                positions = None  # answer-producing tail, resolved per layer

            for layer_name in layer_names:
                hidden = captured.get(layer_name, None)
                if hidden is None:
                    continue
                if hidden.ndim == 4:
                    # AltUp-style decoder layers (e.g. Gemma3n's
                    # Gemma3nTextDecoderLayer) return a stacked tensor of
                    # shape (altup_num_inputs, batch, seq, hidden) instead of
                    # the conventional (batch, seq, hidden) hidden state.
                    # Select the "active" prediction stream so downstream
                    # position-based indexing operates on the real
                    # batch/seq dims instead of mistaking altup_num_inputs's
                    # batch-sized second axis for the sequence length.
                    altup_idx = getattr(
                        getattr(language_model, "config", None), "altup_active_idx", None
                    )
                    if altup_idx is None:
                        altup_idx = getattr(getattr(model, "config", None), "altup_active_idx", 0)
                    hidden = hidden[altup_idx]
                if hidden.ndim == 2:
                    hidden = hidden.unsqueeze(0)
                if config["mode"] == "patch":
                    valid = positions[positions < hidden.shape[1]]
                    scored = hidden[:, valid, :]
                else:
                    # Positions >= prompt_len - 1 are the states that produce
                    # the answer tokens (last prompt token predicts the first).
                    start = min(max(prompt_len - 1, 0), hidden.shape[1] - 1)
                    scored = hidden[:, start:, :]
                result = score_hidden_state_with_logit_lens(
                    hidden_state=scored,
                    lm_head=lm_head,
                    tokenizer=tokenizer,
                    concept_text=str(tag),
                    language_model=language_model,
                    position_aggregation="max",
                )
                per_layer_rel[layer_name].append(result["relative_probability"])
                per_layer_abs[layer_name].append(result["concept_token_probability"])
                record["layer_scores"][layer_name] = {
                    "relative_probability": result["relative_probability"],
                    "concept_token_probability": result["concept_token_probability"],
                }
            region_records.append(record)
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()
        clear_hooks_variables()

    scored_layers = [name for name in layer_names if per_layer_rel[name]]
    if not scored_layers:
        if logger is not None:
            logger.warning(f"[layer-selection] No layer could be scored for tag '{tag}'.")
        return None

    layer_candidates = [
        {
            "layer_name": name,
            "relative_probability": sum(per_layer_rel[name]) / len(per_layer_rel[name]),
            "concept_token_probability": sum(per_layer_abs[name]) / len(per_layer_abs[name]),
            "per_region_relative": per_layer_rel[name],
        }
        for name in scored_layers
    ]
    selected = max(layer_candidates, key=lambda item: item["relative_probability"])
    selected_layer = selected["layer_name"]

    summary = {
        "concept": str(tag),
        "mode": config["mode"],
        "layers_spec": config["layers_spec"],
        "num_regions": len(region_records),
        "region_source": "crops.json regions (uniform sample), same preprocessing as extraction",
        "seed": seed,
        "selection_metric": "relative_probability = p(tag) / p(top-1), max over positions, mean over regions",
        "selected_layer": selected_layer,
        "selected_layer_relative_probability": selected["relative_probability"],
        "selected_layer_concept_probability": selected["concept_token_probability"],
        "layer_scores": [
            {k: v for k, v in candidate.items()} for candidate in layer_candidates
        ],
        "regions": region_records,
    }

    save_dir = Path(getattr(args, "save_dir", ".")) / "logitlens"
    safe_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(tag))
    write_layer_selection_debug(
        debug_dir=save_dir / safe_tag,
        sample_id=safe_tag,
        concept_text=str(tag),
        layer_candidates=layer_candidates,
        selected_layer=selected_layer,
        selected_token=str(tag),
        selected_token_probability=selected["relative_probability"],
        logger=logger,
        score_key="relative_probability",
        score_label="Relative concept probability p(tag) / p(top-1)",
    )
    import json as _json

    with open(save_dir / safe_tag / "layer_selection.json", "w", encoding="utf-8") as handle:
        _json.dump(summary, handle, indent=2, ensure_ascii=False)

    if logger is not None:
        logger.info(
            f"[layer-selection] tag='{tag}' mode={config['mode']} "
            f"regions={len(region_records)} -> {selected_layer} "
            f"(rel_prob={selected['relative_probability']:.4f})"
        )

    # Keep the stored summary light: region details live in the JSON file.
    stored_summary = {
        k: v for k, v in summary.items() if k not in {"regions"}
    }
    return {"selected_layer": selected_layer, "summary": stored_summary}
