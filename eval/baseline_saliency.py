#!/usr/bin/env python3
"""
Attention-map and GradCAM saliency for the TGCL rebuttal's baseline
conditions (reviewer-facing controls, evaluated once each against the fixed
apple/cat/bird eval set -- not part of the 45-run concept-decomposition
grid, since neither method uses a learned concept dictionary).

Design (verified against google/gemma-3n-E4B-it directly, not assumed):
- Gemma-3n is a self-attention-only decoder (no encoder-decoder cross
  attention), so glimpse/attention.py's `outputs.cross_attentions` approach
  does not apply here. Both baselines instead hook
  `model.language_model.layers[i].self_attn` directly with
  attn_implementation="eager" (required -- sdpa/flash attention backends
  don't materialize attention weight tensors to hook).
- A full forward+backward pass through all 35 decoder layers plus the
  262k-vocab lm_head does not fit in one 24GB GPU alongside the attention
  weights this needs to retain (confirmed empirically: OOM at ~23GB even
  forward-only with output_attentions=True). Sharded across 2 GPUs via this
  repo's existing device_utils.get_device_config("cuda:i,j") (device_map -
  the same mechanism scripts/run_full_pipeline.py already uses for
  multi-GPU configs), it fits comfortably.
- Attention-map baseline: forward pass only, average raw attention weights
  (from the target token's position, over the 256 image-token positions)
  across heads and layers.
- GradCAM baseline: adds one backward pass from the target token's logit,
  and weights each layer's attention by ReLU(grad) before averaging --
  standard gradient-weighted attention relevance (Chefer et al.-style), not
  glimpse/glimpse_explainer.py's more elaborate (and, per direct code
  reading, less battle-tested against this model) adaptive-layer-weighting
  scheme. This is a deliberate, simpler, independently-implemented
  formulation, chosen because it's the version actually verified end-to-end
  against Gemma-3n here, rather than inheriting an untested 2000-line module.
- Vision tokens are 256 per image (config.vision_soft_tokens_per_image),
  laid out as a contiguous 16x16 raster block in input_ids wherever
  config.image_token_id appears -- confirmed via direct inspection, not
  assumed from any generic model default.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import gc

import numpy as np
import torch

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

MODEL_NAME = "google/gemma-3n-E4B-it"
VISION_GRID = 16  # sqrt(256) soft tokens/image for gemma-3n-E4B-it


def load_model_and_processor(devices: str = "cuda:0,1"):
    from device_utils import get_device_config
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration

    device_config = get_device_config(devices)
    model = Gemma3nForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        device_map=device_config.device_map,
        max_memory=device_config.max_memory,
        token=os.getenv("HF_TOKEN", None),
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, token=os.getenv("HF_TOKEN", None))
    return model, processor, device_config


def _language_model(model):
    return model.model.language_model if hasattr(model.model, "language_model") else model.model


def compute_saliency(
    model, processor, image, prompt: str, method: str = "gradcam"
) -> Tuple[np.ndarray, int, str]:
    """Returns (saliency_16x16, target_token_id, target_token_text).

    method: "attention" (no grad) or "gradcam" (gradient-weighted).
    """
    assert method in ("attention", "gradcam")
    lang_model = _language_model(model)
    device = next(model.parameters()).device

    conversation = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]
    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(device)

    image_token_id = model.config.image_token_id
    image_mask = (inputs["input_ids"][0] == image_token_id)
    n_image_tokens = int(image_mask.sum().item())
    if n_image_tokens != VISION_GRID * VISION_GRID:
        raise RuntimeError(
            f"Expected {VISION_GRID * VISION_GRID} image tokens, found {n_image_tokens} "
            f"-- vision token layout assumption (contiguous {VISION_GRID}x{VISION_GRID} "
            "raster block) may not hold for this input; aborting rather than silently "
            "producing a meaningless saliency map."
        )
    image_positions = image_mask.nonzero(as_tuple=True)[0]

    captured = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                attn_w = out[1]
                if method == "gradcam":
                    attn_w.retain_grad()
                captured[layer_idx] = attn_w
        return hook

    handles = [
        lang_model.layers[li].self_attn.register_forward_hook(make_hook(li))
        for li in range(len(lang_model.layers))
    ]

    try:
        grad_ctx = torch.enable_grad() if method == "gradcam" else torch.no_grad()
        with grad_ctx:
            out = model(**inputs, output_attentions=True)
            logits = out.logits[0, -1]
            target_id = int(torch.argmax(logits).item())
            if method == "gradcam":
                target_logit = logits[target_id]
                target_logit.backward()
    finally:
        for h in handles:
            h.remove()
        if method == "gradcam":
            # backward() populates .grad on every trainable parameter in
            # the whole ~8-16GB model (nothing is frozen), not just the
            # attn_w tensors retain_grad() was called on -- confirmed via
            # direct measurement this is what was driving GPU memory from
            # ~12GB to ~22.5GB after a single image and never being freed by
            # gc.collect()/empty_cache() alone. Must explicitly drop it.
            for p in model.parameters():
                p.grad = None

    target_text = processor.tokenizer.decode([target_id])

    # Aggregate over layers/heads: attention FROM the last (query) position
    # TO each image-token (key) position. Move each layer's (small) row to
    # CPU immediately -- device_map shards layers across GPUs, so stacking
    # in-place would mix devices, and holding every layer's full (heads,
    # seq_len, seq_len) attention/grad tensor on-GPU across the whole loop
    # is what was driving the OOM/leak seen in initial testing.
    image_positions_cpu = image_positions.cpu()
    layer_maps = []
    for li, attn_w in captured.items():
        # attn_w: (1, num_heads, seq_len, seq_len)
        row = attn_w[0, :, -1, :].detach().float()  # (heads, seq_len)
        if method == "gradcam":
            grad = attn_w.grad
            if grad is None:
                continue
            weight = torch.relu(grad[0, :, -1, :].detach().float())  # (heads, seq_len)
            row = weight * row
        row = row.mean(dim=0).cpu()  # (seq_len,) -- averaged over heads, off-GPU
        layer_maps.append(row)

    if not layer_maps:
        raise RuntimeError("No attention captured from any layer -- eager hooks did not fire.")

    agg = torch.stack(layer_maps, dim=0).mean(dim=0)  # (seq_len,) averaged over layers, on CPU
    image_saliency = agg[image_positions_cpu].numpy()  # (256,)
    saliency_grid = image_saliency.reshape(VISION_GRID, VISION_GRID)

    # Drop every captured tensor/graph reference explicitly rather than
    # relying on scope-exit GC timing -- confirmed necessary empirically
    # (GPU memory climbed across successive calls without this).
    captured.clear()
    del out
    if method == "gradcam":
        del target_logit
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return saliency_grid, target_id, target_text


def pool_to_patch_grid(saliency_16x16: np.ndarray, patch_grid: int = 8) -> np.ndarray:
    """Average-pool the 16x16 vision-token saliency map down to patch_grid x
    patch_grid (default 8x8) for pixel-space insertion/deletion masking."""
    assert VISION_GRID % patch_grid == 0, f"{VISION_GRID} not divisible by {patch_grid}"
    factor = VISION_GRID // patch_grid
    t = torch.from_numpy(saliency_16x16).float().unsqueeze(0).unsqueeze(0)
    pooled = torch.nn.functional.avg_pool2d(t, kernel_size=factor, stride=factor)
    return pooled.squeeze(0).squeeze(0).numpy()
