#!/usr/bin/env python3
"""
Token-level blur evaluation for prompt-grounding outputs.

Reads the JSON produced by inference/vlm_prompt_grounding_explainer.py and aggregates over all
grounded objects across images:
- For each {object name, bbox} in grounded_objects, blur only that bbox at multiple strengths.
- For each blurred input, compute the probability of the FIRST token of the object name at the
    first generation step (i.e., next token after the prompt) using lm_head over hidden states
    followed by softmax.

Outputs:
- Per-image JSON with per-token probability curves.
- Aggregate CSV of blur_strength vs mean probability across all tokens (for quick plotting).

Notes:
- We obtain the target token id by tokenizing the object name and taking its first token id.
- The measured probability is p(first_token(object_name) | prompt, image) at the prompt end.
- Blur strength is mapped to GaussianBlur radius via: radius = strength * radius_scale (default 0.20).
"""
from __future__ import annotations

import os
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from PIL import Image, ImageFilter


def _clip_bbox(bb: List[float]) -> List[float]:
    return [max(0.0, min(1.0, float(v))) for v in bb[:4]]


def _blur_box(img: Image.Image, bbox_norm: List[float], radius: float) -> Image.Image:
    w, h = img.size
    x0 = int(round(bbox_norm[0] * w))
    y0 = int(round(bbox_norm[1] * h))
    x1 = int(round(bbox_norm[2] * w))
    y1 = int(round(bbox_norm[3] * h))
    x0, y0 = max(0, min(x0, w - 1)), max(0, min(y0, h - 1))
    x1, y1 = max(x0 + 1, min(x1, w)), max(y0 + 1, min(y1, h))
    roi = img.crop((x0, y0, x1, y1))
    if radius > 0:
        roi = roi.filter(ImageFilter.GaussianBlur(radius=radius))
    out = img.copy()
    out.paste(roi, (x0, y0))
    return out


def _load_model_and_tools(model_name: str):
    # Reuse the repo's model loader for consistency with explainer
    import sys
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from models import get_model_class  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = argparse.Namespace(
        local_files_only=False,
        cache_dir=os.environ.get("HF_HOME", "/mnt/abka03/huggingface/hub"),
    )
    model_class = get_model_class(
        model_name_or_path=model_name,
        processor_name=model_name,
        device=device,
        logger=None,
        args=args,
    )
    model = model_class.get_model()
    processor = model_class.get_processor()
    tokenizer = None
    if hasattr(model_class, "get_tokenizer"):
        try:
            tokenizer = model_class.get_tokenizer()
        except Exception:
            tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(processor, "tokenizer", None)
    return model, processor, tokenizer, model_class


def _build_instruction_for_object(obj_name: str) -> str:
    # Minimal prompt per user request; keep wording concise and restrictive.
    return (
        f"If there is '{obj_name}' in the image, predict '{obj_name}', otherwise say 'UNK'. "
        "Respond with a single word only and no extra output."
    )


def _prepare_inputs_single(processor, model_class, image: Image.Image, instruction: str) -> Dict[str, Any]:
    if getattr(model_class, "preprocessor", None) is not None:
        return model_class.preprocessor(
            instruction=instruction,
            image_file=image,
            response="",
            generation_mode=True,
        )
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    return processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def _collate_inputs(batch_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack a list of preprocessed input dicts into a batch along dim 0 for tensor values.
    Non-tensor values are taken from the first element.
    Assumes all dicts have the same keys.
    """
    if not batch_inputs:
        return {}
    keys = batch_inputs[0].keys()
    out: Dict[str, Any] = {}
    text_keys = {"input_ids", "attention_mask", "position_ids"}
    for k in keys:
        vals = [bi[k] for bi in batch_inputs]
        if isinstance(vals[0], torch.Tensor):
            # Skip variable-length text keys; we'll build and pad these ourselves later
            if k in text_keys:
                continue
            try:
                out[k] = torch.cat(vals, dim=0)
            except Exception:
                # Fallback: stack if all shapes match exactly
                try:
                    if all(tuple(v.shape) == tuple(vals[0].shape) for v in vals):
                        out[k] = torch.stack(vals, dim=0)
                    else:
                        # As a safety, skip keys that can't be stacked cleanly
                        continue
                except Exception:
                    continue
        else:
            out[k] = vals[0]
    return out


def _logits_via_lm_head(model, out) -> torch.Tensor:
    """Return [B, T, V] logits computed strictly as lm_head(hidden_states[-1]).

    Requires the forward call to include output_hidden_states=True.
    """
    hs = getattr(out, "hidden_states", None)
    if not isinstance(hs, (list, tuple)) or len(hs) == 0:
        raise RuntimeError("hidden_states not available; pass output_hidden_states=True to model forward")
    last_h = hs[-1]
    if not (hasattr(model, "lm_head") and callable(getattr(model, "lm_head"))):
        raise RuntimeError("Model has no callable lm_head to compute logits")
    return model.lm_head(last_h)


def _tokenize_output(tokenizer, processor, text: str) -> List[int]:
    if tokenizer is not None:
        enc = tokenizer(text, add_special_tokens=False)
        return enc["input_ids"] if isinstance(enc, dict) else list(enc)
    # fallback
    enc = processor.tokenizer(text, add_special_tokens=False)
    return enc["input_ids"] if isinstance(enc, dict) else list(enc)


def _gather_positions_for_tokens(gen_ids: List[int], targets: List[int]) -> List[Optional[int]]:
    """Return first unmatched index in gen_ids for each target id; None if not found."""
    used: set = set()
    pos: List[Optional[int]] = []
    for tid in targets:
        found = None
        for i, gid in enumerate(gen_ids):
            if i in used:
                continue
            if gid == tid:
                found = i
                used.add(i)
                break
        pos.append(found)
    return pos


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute token probabilities under Gaussian blur over grounded boxes")
    ap.add_argument("--results_json", required=True, help="Path to vlm_groundings.json")
    ap.add_argument("--out_dir", required=True, help="Directory to write outputs")
    ap.add_argument("--strength_start", type=int, default=0)
    ap.add_argument("--strength_end", type=int, default=100)
    ap.add_argument("--strength_step", type=int, default=10)
    ap.add_argument("--radius_scale", type=float, default=0.20, help="Gaussian blur radius = strength * radius_scale")
    ap.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluating blurred images")
    ap.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16", "none"], help="Autocast dtype for forward pass")
    ap.add_argument("--max_text_tokens", type=int, default=50, help="Cap the number of text tokens (input_ids) per sample")
    args = ap.parse_args()

    results_payload = json.loads(Path(args.results_json).read_text())
    model_name = results_payload.get("model_card") or results_payload.get("model_name") or "google/gemma-3n-E4B-it"
    results: List[Dict[str, Any]] = results_payload.get("results", results_payload)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Outputs are aggregated across all tokens; per-image files are not required.

    model, processor, tokenizer, model_class = _load_model_and_tools(str(model_name))
    device = next(model.parameters()).device
    model.eval()

    strengths = list(range(int(args.strength_start), int(args.strength_end) + 1, max(1, int(args.strength_step))))

    # Build mapping: token_id -> list of (image_path, bbox, optional_name)
    from collections import defaultdict
    token_groups: Dict[int, List[Tuple[str, List[float], Optional[str]]]] = defaultdict(list)
    for item in results:
        img_path = item.get("image_path")
        if not img_path or not os.path.exists(img_path):
            continue
        # Map token id to object name using grounding_over_sequence if available
        tid2name: Dict[int, str] = {}
        gos = item.get("grounding_over_sequence") or {}
        names = gos.get("name") or []
        tid_lists = gos.get("token_id") or []
        if isinstance(names, list) and isinstance(tid_lists, list):
            for nm, tids in zip(names, tid_lists):
                if isinstance(tids, list):
                    for tid in tids:
                        if isinstance(tid, int):
                            tid2name[tid] = nm
        # Collect token explanations (token_id + bbox)
        token_entries = item.get("per_token_explantion") or []
        for te in token_entries:
            if not isinstance(te, dict):
                continue
            tid = te.get("token_id")
            bb = te.get("explantion")
            if not isinstance(tid, int) or not (isinstance(bb, list) and len(bb) == 4):
                continue
            token_groups[int(tid)].append((img_path, _clip_bbox(bb), tid2name.get(int(tid))))

    # Aggregate over all tokens across all images
    agg_probs: Dict[int, List[float]] = {s: [] for s in strengths}

    for s in strengths:
        radius = float(s) * float(args.radius_scale)
        for tid, entries in token_groups.items():
            # Preprocess each blurred image for this token; record prompt lengths
            pre_list: List[Dict[str, Any]] = []
            prompt_lens: List[int] = []
            target_tid_list: List[int] = []
            keep_mask: List[bool] = []
            for (img_path, bb, obj_name) in entries:
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    keep_mask.append(False)
                    # placeholders
                    pre_list.append({})
                    prompt_lens.append(0)
                    target_tid_list.append(0)
                    continue
                img_blur = _blur_box(img, bb, radius)
                instruction = _build_instruction_for_object(obj_name) if obj_name else "Respond with a single word only and no extra output."
                inp = _prepare_inputs_single(processor, model_class, img_blur, instruction)
                # Truncate text tokens safely: preserve all image tokens (large repeated run) and keep only
                # the first max_text_tokens after the last image token. This avoids image-token/embedding mismatch.
                if isinstance(inp.get("input_ids"), torch.Tensor):
                    ids = inp["input_ids"]  # [1, T]
                    T = ids.shape[-1]
                    if args.max_text_tokens and T > args.max_text_tokens:
                        arr = ids[0].tolist()
                        # Detect the longest contiguous run; treat it as image token run if long enough
                        best_len = 0
                        best_end = -1
                        i = 0
                        while i < len(arr):
                            j = i + 1
                            while j < len(arr) and arr[j] == arr[i]:
                                j += 1
                            run_len = j - i
                            if run_len > best_len:
                                best_len = run_len
                                best_end = j - 1
                            i = j
                        # Heuristic threshold: image token run is usually very long (>= 64)
                        if best_len >= 64 and best_end >= 0:
                            first_text_end = min(T, (best_end + 1) + int(args.max_text_tokens))
                            inp["input_ids"] = ids[:, :first_text_end]
                            if isinstance(inp.get("attention_mask"), torch.Tensor) and inp["attention_mask"].shape[-1] == T:
                                inp["attention_mask"] = inp["attention_mask"][:, :first_text_end]
                            if isinstance(inp.get("position_ids"), torch.Tensor) and inp["position_ids"].shape[-1] == T:
                                inp["position_ids"] = inp["position_ids"][:, :first_text_end]
                pre_list.append(inp)
                prompt_lens.append(int(inp["input_ids"].shape[-1]))
                # Use provided token id directly as target
                target_tid_list.append(int(tid))
                keep_mask.append(True)

            # Filter out invalid entries
            valid_idx = [i for i, ok in enumerate(keep_mask) if ok]
            if not valid_idx:
                # nothing valid in this mini-batch
                continue
            # Dynamic sub-batching to avoid OOM
            sub_bs = len(valid_idx)
            start = 0
            while start < len(valid_idx):
                end = min(len(valid_idx), start + sub_bs)
                sub_idx = valid_idx[start:end]
                try:
                    # Collate only this sub-batch
                    batch_inputs = _collate_inputs([pre_list[i] for i in sub_idx])
                    batch_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch_inputs.items()}

                    # Build per-sample ctx ids and masks
                    pad_id = getattr(tokenizer, 'pad_token_id', 0) if tokenizer is not None else 0
                    ctx_ids_list: List[torch.Tensor] = []
                    attn_list: List[torch.Tensor] = []
                    for i in sub_idx:
                        prompt_len = prompt_lens[i]
                        base_prompt = pre_list[i]["input_ids"][0]
                        ctx_ids = base_prompt.to(device)
                        ctx_ids_list.append(ctx_ids)
                        attn_list.append(torch.ones_like(ctx_ids, dtype=torch.long, device=device))
                    max_len = max(x.size(0) for x in ctx_ids_list)
                    ctx_pad = []
                    attn_pad = []
                    for x, a in zip(ctx_ids_list, attn_list):
                        if x.size(0) < max_len:
                            pad_len = max_len - x.size(0)
                            x = torch.cat([x, torch.full((pad_len,), pad_id, dtype=torch.long, device=device)], dim=0)
                            a = torch.cat([a, torch.zeros((pad_len,), dtype=torch.long, device=device)], dim=0)
                        ctx_pad.append(x.unsqueeze(0))
                        attn_pad.append(a.unsqueeze(0))
                    ctx_batch = torch.cat(ctx_pad, dim=0)
                    attn_batch = torch.cat(attn_pad, dim=0)

                    model_inputs = dict(batch_inputs)
                    model_inputs["input_ids"] = ctx_batch
                    model_inputs["attention_mask"] = attn_batch
                    # Ensure hidden states are returned so we can compute logits via lm_head explicitly
                    model_inputs["output_hidden_states"] = True

                    # Mixed precision autocast if requested
                    amp_enabled = (device.type == 'cuda' and args.amp_dtype != 'none')
                    amp_dtype = torch.float16 if args.amp_dtype == 'float16' else (torch.bfloat16 if args.amp_dtype == 'bfloat16' else None)
                    with torch.inference_mode():
                        if amp_enabled:
                            # Prefer new API
                            with torch.amp.autocast("cuda", dtype=amp_dtype):  # type: ignore[arg-type]
                                out = model(**model_inputs)
                                logits = _logits_via_lm_head(model, out)
                        else:
                            out = model(**model_inputs)
                            logits = _logits_via_lm_head(model, out)

                    # Extract probabilities for this token
                    for vb, i in enumerate(sub_idx):
                        prompt_len = prompt_lens[i]
                        t = prompt_len - 1
                        if 0 <= t < logits.shape[1]:
                            tid = int(target_tid_list[i])
                            logit = logits[vb, t]
                            prob = torch.softmax(logit.float(), dim=-1)[tid].item()
                            agg_probs[s].append(float(prob))
                    # Advance window
                    start = end
                    # Free intermediates
                    del logits, out, ctx_batch, attn_batch
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    # Halve sub-batch and retry
                    torch.cuda.empty_cache()
                    if sub_bs <= 1:
                        # give up on this slice; advance by 1 to avoid infinite loop
                        start = end
                        continue
                    sub_bs = max(1, sub_bs // 2)

    # Write aggregate curve CSV: blur_strength, mean_prob
    import csv
    curve_csv = out_dir / "token_blur_curve.csv"
    with curve_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blur_strength", "mean_prob"])
        for s in strengths:
            vals = [v for v in agg_probs[s] if (v == v)]  # filter NaNs
            mean_prob = (sum(vals) / max(1, len(vals))) if vals else float("nan")
            w.writerow([s, mean_prob])

    print(f"Wrote: {curve_csv}")


if __name__ == "__main__":
    main()
