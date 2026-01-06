import os
import random
import re
import time
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

try:
    from captum.attr import IntegratedGradients
except ImportError as exc:  # pragma: no cover - ensures clear guidance when dependency missing
    raise ImportError(
        "save_cam.py now depends on Captum for Integrated Gradients. Please install it via `pip install captum`."
    ) from exc

from datasets import get_dataset_loader
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from models import get_model_class
from models.image_text_model import ImageTextModel


CAM_ACTIVATIONS: Dict[str, torch.Tensor] = {}


def compute_time_left(start_time: float, completed: int, total: int) -> float:
    if completed <= 0:
        return 0.0
    elapsed = time.time() - start_time
    avg_time = elapsed / completed
    remaining = max(total - completed, 0)
    return (avg_time * remaining) / 60.0


def set_seed(seed_value: int = 42) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _capture_activations(module_name: str):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        CAM_ACTIVATIONS[module_name] = output
        return None

    return hook


def _match_module_name(name: str, patterns: Sequence[str], exact: bool) -> bool:
    if exact:
        return name in patterns
    for pattern in patterns:
        regex = re.compile(re.sub(r"\\*", ".*", pattern))
        if regex.search(name):
            return True
    return False


def _register_cam_hooks(
    model: torch.nn.Module,
    module_patterns: Sequence[str],
    exact_match: bool,
) -> Tuple[List[torch.utils.hooks.RemovableHandle], List[str]]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    matched: List[str] = []
    for name, module in model.named_modules():
        if _match_module_name(name, module_patterns, exact_match):
            handles.append(module.register_forward_hook(_capture_activations(name)))
            matched.append(name)
    if not matched:
        raise ValueError(
            f"No modules matched the requested patterns: {module_patterns}."
        )
    return handles, matched


def _clear_cam_hooks(handles: Sequence[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


def _clear_cam_activations() -> None:
    CAM_ACTIVATIONS.clear()


def _clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _ensure_per_image_bbox_list(crops: Any, n_imgs: int) -> List[Optional[List[float]]]:
    if crops is None:
        return [None] * n_imgs
    if isinstance(crops, (list, tuple)) and len(crops) == 4 and all(
        isinstance(v, (int, float)) for v in crops
    ):
        return [list(crops)] * n_imgs
    if isinstance(crops, (list, tuple)) and len(crops) == n_imgs:
        return list(crops)
    return [None] * n_imgs


def _pad_tensor_sequences(
    tensors: List[torch.Tensor],
    padding_value: float = 0,
) -> torch.Tensor:
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)


def _collate_inputs(per_sample_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(per_sample_inputs) == 1:
        return per_sample_inputs[0]
    batched: Dict[str, Any] = {}
    keys = per_sample_inputs[0].keys()
    pad_id = 0
    for k in keys:
        vals = [d[k] for d in per_sample_inputs]
        first = vals[0]
        if isinstance(first, torch.Tensor):
            if k == "input_ids":
                seqs = []
                for v in vals:
                    if v.ndim == 2 and v.shape[0] == 1:
                        seqs.append(v.squeeze(0))
                    else:
                        seqs.append(v)
                padding_value = pad_id
                batched[k] = _pad_tensor_sequences(seqs, padding_value=padding_value)
            elif k == "attention_mask":
                seqs = []
                for v in vals:
                    if v.ndim == 2 and v.shape[0] == 1:
                        seqs.append(v.squeeze(0))
                    else:
                        seqs.append(v)
                batched[k] = _pad_tensor_sequences(seqs, padding_value=0)
            else:
                arrs = []
                for v in vals:
                    if v.ndim == 0:
                        arrs.append(v.unsqueeze(0))
                    else:
                        arrs.append(v)
                try:
                    batched[k] = torch.cat(arrs, dim=0)
                except Exception:
                    batched[k] = vals
        else:
            batched[k] = vals
    return batched


def _extract_item_value(item: Dict[str, Any], key: str, idx: int) -> Any:
    value = item.get(key)
    if isinstance(value, list):
        if idx < len(value):
            return value[idx]
        return None
    return value


def _prepare_inputs_for_batch(
    item: Dict[str, Any],
    model_class: ImageTextModel,
    args: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    texts = item["text"] if isinstance(item["text"], list) else [item["text"]]
    toi = getattr(args, "token_of_interest", None)
    if toi is not None and "cgdl" in getattr(args, "prompt_template", ""):
        toi_str = str(toi).strip()
        texts = [t.replace("[concept]", toi_str) if isinstance(t, str) else t for t in texts]
    image_paths = item["image"] if isinstance(item["image"], list) else [item["image"]]
    crop_locations = item.get("bbox")
    per_image_bboxes = _ensure_per_image_bbox_list(crop_locations, len(image_paths))

    per_sample_inputs: List[Dict[str, Any]] = []
    for idx, img_ref in enumerate(image_paths):
        bbox = per_image_bboxes[idx]
        if bbox is not None:
            img = Image.open(img_ref).convert("RGB") if not isinstance(img_ref, Image.Image) else img_ref
            width, height = img.size
            x1, y1, x2, y2 = bbox
            x1 = int(_clip(x1, 0, width - 1))
            y1 = int(_clip(y1, 0, height - 1))
            x2 = int(_clip(x2, x1 + 1, width))
            y2 = int(_clip(y2, y1 + 1, height))
            crop_img = img.crop((x1, y1, x2, y2))
            per_sample_inputs.append(
                model_class.preprocessor(
                    instruction=texts[idx],
                    image_file=crop_img,
                    response="",
                    generation_mode=args.generation_mode,
                )
            )
        else:
            per_sample_inputs.append(
                model_class.preprocessor(
                    instruction=texts[idx],
                    image_file=image_paths[idx],
                    response="",
                    generation_mode=args.generation_mode,
                )
            )

    inputs = _collate_inputs(per_sample_inputs)

    metadata: List[Dict[str, Any]] = []
    for idx in range(len(image_paths)):
        metadata.append(
            {
                "image_path": image_paths[idx],
                "bbox": per_image_bboxes[idx],
                "instruction": _extract_item_value(item, "instruction", idx),
                "response": _extract_item_value(item, "response", idx),
                "targets": _extract_item_value(item, "targets", idx),
                "concept_label": _extract_item_value(item, "concept", idx),
                "text": texts[idx],
            }
        )
    return inputs, metadata


def _find_image_tensor(inputs: Dict[str, Any]) -> Tuple[str, torch.Tensor]:
    candidate_keys = [
        "pixel_values",
        "image_values",
        "vision_inputs",
        "images",
        "image",
    ]
    for key in candidate_keys:
        tensor = inputs.get(key)
        if torch.is_tensor(tensor):
            return key, tensor
    for key, value in inputs.items():
        if torch.is_tensor(value) and value.dtype.is_floating_point and value.ndim >= 3:
            return key, value
    raise RuntimeError("Could not find an image-like tensor in the model inputs.")


def _reduce_to_heatmap(grad_tensor: torch.Tensor) -> torch.Tensor:
    grad = grad_tensor.detach().float()
    while grad.dim() > 2:
        grad = grad.mean(dim=0)
    if grad.numel() == 0:
        return grad.cpu()
    grad = grad - grad.min()
    if grad.max() > 0:
        grad = grad / grad.max()
    return grad.cpu()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_heatmap_png(heatmap: torch.Tensor, image_path: str, output_path: str) -> None:
    heatmap_np = heatmap.detach().cpu().numpy()
    if heatmap_np.ndim == 1:
        heatmap_np = heatmap_np[None, :]
    heatmap_np = heatmap_np - heatmap_np.min()
    if heatmap_np.max() > 0:
        heatmap_np = heatmap_np / heatmap_np.max()
    heatmap_img = Image.fromarray((heatmap_np * 255).astype(np.uint8))
    with Image.open(image_path) as base_img:
        base_img = base_img.convert("RGB")
        heatmap_img = heatmap_img.resize(base_img.size, Image.BILINEAR)
    _ensure_dir(os.path.dirname(output_path))
    heatmap_img.save(output_path)


def _save_text_heatmaps(entries: List[Dict[str, Any]], args: Any, logger: Any) -> Optional[str]:
    if not entries:
        return None
    output_path = os.path.join(args.save_dir, f"{args.save_filename}_text_heatmaps.pth")
    _ensure_dir(args.save_dir)
    torch.save(entries, output_path)
    if logger is not None:
        logger.info(f"Saved text heatmap data to {output_path}")
    return output_path

def _build_baseline(image_tensor: torch.Tensor, baseline_type: str = "zero") -> torch.Tensor:
    if baseline_type == "zero":
        return torch.zeros_like(image_tensor)
    if baseline_type == "mean":
        baseline = image_tensor.mean(dim=tuple(range(1, image_tensor.dim())), keepdim=True)
        return baseline.expand_as(image_tensor)
    if baseline_type == "random":
        return torch.rand_like(image_tensor)
    raise ValueError(f"Unsupported ig_baseline '{baseline_type}'. Choose from ['zero', 'mean', 'random'].")


def _attribution_to_heatmap(attr: torch.Tensor) -> torch.Tensor:
    map_ = attr.detach()
    if map_.dim() == 4:  # (B, C, H, W)
        map_ = map_.mean(dim=1)
    elif map_.dim() == 3:  # (C, H, W)
        map_ = map_.mean(dim=0)
    map_ = map_ - map_.min()
    if map_.max() > 0:
        map_ = map_ / map_.max()
    return map_.cpu()


def _aggregate_activations(module_names: Sequence[str]) -> torch.Tensor:
    activations: List[torch.Tensor] = []
    for name in module_names:
        if name not in CAM_ACTIVATIONS:
            raise RuntimeError(f"Activation for module {name} was not captured.")
        act = CAM_ACTIVATIONS[name]
        if isinstance(act, tuple):
            act = act[0]
        if act.dim() == 2:
            act = act.unsqueeze(1)
        activations.append(act)
    if len(activations) == 1:
        return activations[0]
    aligned = torch.stack(activations, dim=0)
    return aligned.mean(dim=0)


def _compute_concept_scores(
    activations: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    image_token_id: Optional[int],
    concept_vector: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor], List[Optional[torch.Tensor]]]:
    concept_vec = concept_vector.to(device=activations.device, dtype=activations.dtype)
    concept_vec = concept_vec.view(1, 1, -1)
    concept_vec = F.normalize(concept_vec, p=2, dim=-1)
    activations_norm = F.normalize(activations, p=2, dim=-1)
    cosine_full = (activations_norm * concept_vec).sum(dim=-1)

    if input_ids is not None and input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids is not None and input_ids.shape[1] != cosine_full.shape[1]:
        input_ids = input_ids[:, -cosine_full.shape[1] :]

    sample_scores: List[torch.Tensor] = []
    selected_scores: List[torch.Tensor] = []
    token_masks: List[Optional[torch.Tensor]] = []

    for b in range(cosine_full.shape[0]):
        mask: Optional[torch.Tensor] = None
        if input_ids is not None and image_token_id is not None:
            token_ids = input_ids[b]
            token_ids = token_ids.to(cosine_full.device)
            if isinstance(image_token_id, (list, tuple, set)):
                mask = torch.zeros_like(token_ids, dtype=torch.bool, device=cosine_full.device)
                for tid in image_token_id:
                    mask |= (token_ids == tid)
            else:
                mask = (token_ids == image_token_id)
            if mask.dim() > 1:
                mask = mask.any(dim=-1)
            if mask.any():
                token_values = cosine_full[b][mask]
            else:
                mask = None
                token_values = cosine_full[b]
        else:
            token_values = cosine_full[b]
        sample_scores.append(token_values.mean())
        selected_scores.append(token_values.detach().cpu())
        token_masks.append(mask.detach().cpu() if mask is not None else None)

    return sample_scores, cosine_full, selected_scores, token_masks


def _concept_score_forward(
    image_tensor: torch.Tensor,
    base_inputs: Dict[str, Any],
    model: torch.nn.Module,
    module_names: Sequence[str],
    concept_vector: torch.Tensor,
    args: Any,
    image_key: str,
    image_token_id: Optional[int] = None,
    detail_store: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    _clear_cam_activations()
    forward_inputs: Dict[str, Any] = dict(base_inputs)
    forward_inputs[image_key] = image_tensor

    outputs = model(**forward_inputs)
    del outputs

    activations = _aggregate_activations(module_names)
    input_ids = forward_inputs.get("input_ids")
    if torch.is_tensor(input_ids):
        input_ids = input_ids.to(image_tensor.device)

    sample_scores, cosine_full, selected_scores, token_masks = _compute_concept_scores(
        activations, input_ids, image_token_id, concept_vector
    )
    scores_tensor = torch.stack(sample_scores)

    if detail_store is not None:
        detail_store["scores"] = scores_tensor.detach().cpu()
        detail_store["cosine_full"] = cosine_full.detach().cpu()
        detail_store["selected_scores"] = [score.clone() if torch.is_tensor(score) else score for score in selected_scores]
        detail_store["token_masks"] = token_masks
        detail_store["input_ids"] = input_ids.detach().cpu() if input_ids is not None else None

    return scores_tensor


def _load_concept_vector(
    path: str,
    index: int,
    aggregation: str,
    normalize: bool,
    device: torch.device,
) -> torch.Tensor:
    if not path:
        raise ValueError("--concept_vector must be provided for CAM extraction.")
    concept_data = torch.load(path, map_location=device)
    if "concepts" not in concept_data:
        raise KeyError(
            f"Expected key 'concepts' in concept vector file {path}, found: {list(concept_data.keys())}."
        )
    concepts = concept_data["concepts"]
    if not torch.is_tensor(concepts):
        concepts = torch.tensor(concepts, device=device, dtype=torch.float32)
    concepts = concepts.to(device=device, dtype=torch.float32)

    if concepts.dim() == 1:
        vector = concepts
    elif aggregation == "mean":
        vector = concepts.mean(dim=0)
    else:
        if index < 0 or index >= concepts.shape[0]:
            raise IndexError(
                f"concept_index {index} is out of range for {concepts.shape[0]} concepts."
            )
        vector = concepts[index]
    if normalize:
        vector = F.normalize(vector.unsqueeze(0), p=2, dim=-1).squeeze(0)
    return vector


def _process_batch(
    model_class: ImageTextModel,
    model: torch.nn.Module,
    inputs: Dict[str, Any],
    module_names: Sequence[str],
    concept_vector: torch.Tensor,
    metadata: List[Dict[str, Any]],
    tokenizer: Optional[Any],
    args: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    image_key, image_tensor = _find_image_tensor(inputs)
    image_tensor = image_tensor.clone().detach().requires_grad_(True)

    base_inputs = {k: v for k, v in inputs.items() if k != image_key}
    image_token_id = getattr(getattr(model_class, "processor_", None), "image_token_id", None)

    score_forward = partial(
        _concept_score_forward,
        base_inputs=base_inputs,
        model=model,
        module_names=module_names,
        concept_vector=concept_vector,
        args=args,
        image_key=image_key,
        image_token_id=image_token_id,
    )

    baseline = _build_baseline(image_tensor, baseline_type=args.ig_baseline)
    ig = IntegratedGradients(score_forward)
    ig_kwargs: Dict[str, Any] = {}
    if getattr(args, "ig_internal_batch_size", None):
        ig_kwargs["internal_batch_size"] = args.ig_internal_batch_size

    targets: Optional[List[int]] = None
    if image_tensor.shape[0] > 1:
        targets = list(range(image_tensor.shape[0]))

    attributions = ig.attribute(
        image_tensor,
        baselines=baseline,
        n_steps=args.ig_steps,
        method="riemann_trapezoid",
        target=targets,
        **ig_kwargs,
    )

    detail_store: Dict[str, Any] = {}
    with torch.no_grad():
        scores_tensor = score_forward(image_tensor.detach(), detail_store)

    heatmaps_per_sample: List[torch.Tensor] = []
    if attributions.dim() == 3:  # single example without batch dim
        attributions = attributions.unsqueeze(0)
    for idx in range(attributions.shape[0]):
        heatmaps_per_sample.append(_attribution_to_heatmap(attributions[idx]))

    cosine_full = detail_store.get("cosine_full")
    selected_scores = detail_store.get("selected_scores", [])
    token_masks = detail_store.get("token_masks", [])
    stored_input_ids = detail_store.get("input_ids")
    scores_cpu = detail_store.get("scores", scores_tensor.detach().cpu())

    records: List[Dict[str, Any]] = []
    text_entries: List[Dict[str, Any]] = []
    for idx, meta in enumerate(metadata):
        record: Dict[str, Any] = {
            "image_path": meta.get("image_path"),
            "instruction": meta.get("instruction"),
            "response": meta.get("response"),
            "targets": meta.get("targets"),
            "concept_label": meta.get("concept_label"),
            "bbox": meta.get("bbox"),
            "text": meta.get("text"),
            "score": float(scores_cpu[idx].item()),
            "selected_token_scores": selected_scores[idx] if idx < len(selected_scores) else None,
            "cosine_per_token": cosine_full[idx] if cosine_full is not None else None,
            "heatmap": heatmaps_per_sample[idx],
        }
        if token_masks and token_masks[idx] is not None:
            record["image_token_mask"] = token_masks[idx]
        if stored_input_ids is not None:
            record["input_ids"] = stored_input_ids[idx]
        records.append(record)

        text_entry: Dict[str, Any] = {
            "image_path": meta.get("image_path"),
            "text": meta.get("text"),
            "score_per_token": cosine_full[idx] if cosine_full is not None else None,
        }
        if stored_input_ids is not None:
            token_ids = stored_input_ids[idx]
            text_entry["input_ids"] = token_ids
            if tokenizer is not None and token_ids is not None:
                try:
                    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
                    text_entry["tokens"] = tokens
                except Exception:
                    text_entry["tokens"] = None
        text_entries.append(text_entry)

    return records, text_entries


def _process_loader(
    loader: Any,
    args: Any,
    model_class: ImageTextModel,
    concept_vector: torch.Tensor,
    module_patterns: Sequence[str],
    logger: Any,
    image_heatmap_dir: str,
    start_index: int = 0,
    concept_bucket: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]], int]:
    model = model_class.get_model()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if hasattr(model, "config") and getattr(model.config, "use_cache", True):
        model.config.use_cache = False

    handles, matched_names = _register_cam_hooks(
        model=model,
        module_patterns=module_patterns,
        exact_match=args.exact_match_modules_to_hook,
    )
    all_records: List[Dict[str, Any]] = []
    all_text_entries: List[Dict[str, Any]] = []

    start_time = time.time()
    total_batches = len(loader)
    tokenizer = None
    if hasattr(model_class, "get_tokenizer"):
        try:
            tokenizer = model_class.get_tokenizer()
        except Exception:
            tokenizer = None
    _ensure_dir(image_heatmap_dir)
    sample_counter = start_index

    try:
        for batch_idx, item in enumerate(loader):
            inputs, metadata = _prepare_inputs_for_batch(item, model_class, args)
            batch_records, batch_text_entries = _process_batch(
                model_class=model_class,
                model=model,
                inputs=inputs,
                module_names=matched_names,
                concept_vector=concept_vector,
                metadata=metadata,
                tokenizer=tokenizer,
                args=args,
            )
            if concept_bucket is not None:
                for record in batch_records:
                    record["concept_bucket"] = concept_bucket
                for entry in batch_text_entries:
                    entry["concept_bucket"] = concept_bucket

            if len(batch_records) != len(batch_text_entries):
                raise RuntimeError(
                    "Mismatch between CAM records and text heatmap entries."
                )
            for record, text_entry in zip(batch_records, batch_text_entries):
                filename_base = f"{args.save_filename}_{sample_counter:06d}"
                heatmap_path = os.path.join(image_heatmap_dir, f"{filename_base}.png")
                try:
                    _save_heatmap_png(record["heatmap"], record["image_path"], heatmap_path)
                    record["heatmap_path"] = heatmap_path
                except Exception as exc:
                    if logger is not None:
                        logger.warning(
                            f"Failed to save heatmap for {record.get('image_path', 'unknown')}: {exc}"
                        )
                text_entry["heatmap_path"] = record.get("heatmap_path")
                sample_counter += 1

            all_records.extend(batch_records)
            all_text_entries.extend(batch_text_entries)

            if logger is not None and (batch_idx + 1) % 10 == 0:
                time_left = compute_time_left(start_time, batch_idx + 1, total_batches)
                logger.info(
                    f"Processed {batch_idx + 1}/{total_batches} batches. Estimated time left: {time_left:.2f} mins"
                )
    finally:
        _clear_cam_hooks(handles)

    return all_records, matched_names, all_text_entries, sample_counter


def _save_results(
    records: List[Dict[str, Any]],
    matched_modules: Sequence[str],
    args: Any,
    logger: Any,
    image_heatmap_dir: str,
) -> str:
    os.makedirs(args.save_dir, exist_ok=True)
    output_path = os.path.join(args.save_dir, f"{args.save_filename}.pth")

    payload = {
        "records": records,
        "matched_modules": list(matched_modules),
        "concept_vector_path": args.concept_vector,
        "concept_index": args.concept_index,
        "concept_aggregation": args.concept_aggregation,
        "normalize_concept": args.normalize_concept,
        "model_name": args.model_name_or_path,
        "dataset_name": args.dataset_name,
        "prompt_template": args.prompt_template,
        "generation_mode": args.generation_mode,
        "image_heatmap_directory": image_heatmap_dir,
    }

    torch.save(payload, output_path)
    if logger is not None:
        logger.info(f"Saved CAM results to {output_path}")
    return output_path


def main() -> None:
    args = get_arguments()
    if args.concept_vector is None:
        raise ValueError("--concept_vector must be provided when running save_cam.py")

    logger = setup_logger(log_file=os.path.join(args.save_dir, "logs.log"))
    log_args(args, logger)

    set_seed(args.seed)
    device = torch.device(args.device)

    model_class = get_model_class(
        args.model_name_or_path,
        args.processor_name,
        device=device,
        logger=logger,
        args=args,
    )

    concept_vector = _load_concept_vector(
        path=args.concept_vector,
        index=args.concept_index,
        aggregation=args.concept_aggregation,
        normalize=args.normalize_concept,
        device=device,
    )

    loader = get_dataset_loader(
        dataset_name=args.dataset_name,
        logger=logger,
        args=args,
    )

    if args.modules_to_hook is None:
        raise ValueError("--modules_to_hook must be provided to specify where to capture activations.")
    module_patterns: List[str] = []
    for group in args.modules_to_hook:
        if isinstance(group, str):
            module_patterns.append(group)
        else:
            module_patterns.extend(group)
    if not module_patterns:
        raise ValueError("No module patterns resolved from --modules_to_hook.")

    image_heatmap_dir = os.path.join(args.save_dir, f"{args.save_filename}_image_heatmaps")
    all_records: List[Dict[str, Any]] = []
    all_text_entries: List[Dict[str, Any]] = []
    matched_modules: List[str] = []
    sample_counter = 0

    if isinstance(loader, dict):
        for concept_key, dl in loader.items():
            if logger is not None:
                logger.info(f"Running CAM extraction for concept split: {concept_key}")
            batch_records, matched, text_entries, sample_counter = _process_loader(
                loader=dl,
                args=args,
                model_class=model_class,
                concept_vector=concept_vector,
                module_patterns=module_patterns,
                logger=logger,
                image_heatmap_dir=image_heatmap_dir,
                start_index=sample_counter,
                concept_bucket=concept_key,
            )
            all_records.extend(batch_records)
            all_text_entries.extend(text_entries)
            matched_modules = matched
    else:
        all_records, matched_modules, text_entries, sample_counter = _process_loader(
            loader=loader,
            args=args,
            model_class=model_class,
            concept_vector=concept_vector,
            module_patterns=module_patterns,
            logger=logger,
            image_heatmap_dir=image_heatmap_dir,
            start_index=sample_counter,
        )
        all_text_entries.extend(text_entries)

    _save_results(all_records, matched_modules, args, logger, image_heatmap_dir)
    _save_text_heatmaps(all_text_entries, args, logger)


if __name__ == "__main__":
    main()
