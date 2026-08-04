import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def split_top_level_patterns(raw: str) -> List[str]:
    """Split a layer spec by commas/pipes while preserving bracketed ranges."""
    if not raw:
        return []

    patterns: List[str] = []
    current: List[str] = []
    bracket_depth = 0
    for char in raw:
        if char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth > 0:
            bracket_depth -= 1

        if char in {",", "|"} and bracket_depth == 0:
            token = "".join(current).strip()
            if token:
                patterns.append(token)
            current = []
            continue

        current.append(char)

    token = "".join(current).strip()
    if token:
        patterns.append(token)
    return patterns


def _expand_bracket_expression(pattern: str) -> List[str]:
    match = re.fullmatch(r"(?P<prefix>.*)\[(?P<body>.+)\](?P<suffix>.*)", pattern.strip())
    if not match:
        return [pattern.strip()]

    prefix = match.group("prefix")
    body = match.group("body").strip()
    suffix = match.group("suffix")

    if "," in body:
        values = [part.strip() for part in body.split(",") if part.strip()]
    elif "-" in body:
        start_str, end_str = [part.strip() for part in body.split("-", 1)]
        if not start_str or not end_str:
            return [pattern.strip()]
        start_idx = int(start_str)
        end_idx = int(end_str)
        step = 1 if end_idx >= start_idx else -1
        values = [str(idx) for idx in range(start_idx, end_idx + step, step)]
    else:
        values = [body]

    return [f"{prefix}{value}{suffix}" for value in values]


def expand_layer_specs(layer_spec: str) -> List[str]:
    """Expand a layer spec into concrete patterns or numeric indices."""
    # Allow inline comments in .env values, e.g. "[20,23] # optional note".
    layer_spec = (layer_spec or "").split("#", 1)[0].strip()
    expanded: List[str] = []
    for token in split_top_level_patterns(layer_spec):
        expanded.extend(_expand_bracket_expression(token))

    uniq: List[str] = []
    for token in expanded:
        stripped = token.strip()
        if stripped and stripped not in uniq:
            uniq.append(stripped)
    return uniq


def _module_name_priority(name: str) -> Tuple[int, int, str]:
    lowered = name.lower()
    if "language_model" in lowered:
        prefix_rank = 0
    elif re.search(r"(^|\.)model(\.|$)", lowered):
        prefix_rank = 1
    elif ".layers." in lowered:
        prefix_rank = 2
    else:
        prefix_rank = 3
    return prefix_rank, len(name), name


def resolve_layer_modules(model: torch.nn.Module, layer_spec: str, logger: Optional[Any] = None) -> List[str]:
    """Resolve a layer spec into concrete module names from a model.

    Supports:
    - plain module names, which are kept as-is when they match exactly;
    - bracket expressions like ``[5-10]`` or ``[5,7,9]``;
    - plain numeric indices, which are resolved against module names ending in
      ``layers.<index>``.
    """
    named_modules = dict(model.named_modules())
    names = list(named_modules.keys())

    resolved: List[str] = []
    for token in expand_layer_specs(layer_spec):
        if token in named_modules:
            candidate = token
        elif re.fullmatch(r"\d+", token):
            index = int(token)
            candidates = [
                name
                for name in names
                if re.search(rf"(^|\.)layers\.{index}(\.|$)", name)
            ]
            if not candidates:
                raise ValueError(
                    f"Could not resolve layer index {index} in model modules."
                )
            candidates.sort(key=_module_name_priority)
            candidate = candidates[0]
            if logger is not None and len(candidates) > 1:
                logger.warning(
                    f"Layer index {index} matched multiple modules {candidates}; using {candidate}"
                )
        elif token.startswith("layers.") and re.fullmatch(r"layers\.\d+", token):
            index = int(token.split(".")[-1])
            candidates = [
                name
                for name in names
                if re.search(rf"(^|\.)layers\.{index}(\.|$)", name)
            ]
            if not candidates:
                raise ValueError(
                    f"Could not resolve layer token '{token}' in model modules."
                )
            candidates.sort(key=_module_name_priority)
            candidate = candidates[0]
            if logger is not None and len(candidates) > 1:
                logger.warning(
                    f"Layer token {token} matched multiple modules {candidates}; using {candidate}"
                )
        else:
            candidate = token

        if candidate not in resolved:
            resolved.append(candidate)

    return resolved


def resolve_final_norm_module(language_model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Best-effort resolution of the final normalization module."""
    direct_attrs = ["norm", "ln_f", "final_layer_norm", "layer_norm"]
    for attr in direct_attrs:
        module = getattr(language_model, attr, None)
        if isinstance(module, torch.nn.Module):
            return module

    nested_paths = [
        "model.norm",
        "model.ln_f",
        "transformer.ln_f",
        "decoder.final_layer_norm",
        "language_model.norm",
        "language_model.model.norm",
    ]
    for path in nested_paths:
        current: Any = language_model
        found = True
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                found = False
                break
        if found and isinstance(current, torch.nn.Module):
            return current

    for name, module in language_model.named_modules():
        if name and (name.endswith("norm") or name.endswith("ln_f") or name.endswith("final_layer_norm")):
            if ".layers." not in name:
                return module

    return None


def get_concept_token_ids(tokenizer: Any, concept_text: str) -> List[int]:
    """Collect candidate token ids for a concept string using common variants."""
    if tokenizer is None:
        return []

    raw = (concept_text or "").strip()
    if not raw:
        return []

    variants = [raw, raw.lower(), raw.upper(), raw.capitalize(), f" {raw}", f" {raw.lower()}"]
    token_ids: List[int] = []
    for variant in variants:
        try:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
        except Exception:
            continue
        for token_id in encoded:
            if token_id not in token_ids:
                token_ids.append(token_id)
    return token_ids


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=True).strip()
    except Exception:
        return str(int(token_id))


def score_hidden_state_with_logit_lens(
    hidden_state: torch.Tensor,
    lm_head: torch.nn.Module,
    tokenizer: Any,
    concept_text: str,
    language_model: Optional[torch.nn.Module] = None,
    top_k: int = 10,
    valid_token_count: Optional[int] = None,
    position_aggregation: str = "max",
) -> Dict[str, Any]:
    """Apply final norm + lm_head and score concept tokens with logit lens.

    This follows the common "logit-lens" pattern: compute LM logits from
    intermediate hidden states, convert to probabilities, then score the
    concept tokens across all sequence positions and aggregate by
    *position_aggregation* (default: "max").
    """
    if hidden_state.dim() == 2:
        hidden_state = hidden_state.unsqueeze(0)
    if hidden_state.dim() != 3:
        raise ValueError(f"Expected hidden state rank 2 or 3, got shape {tuple(hidden_state.shape)}")

    device = next(lm_head.parameters()).device
    dtype = next(lm_head.parameters()).dtype
    hidden_state = hidden_state.to(device=device, dtype=dtype)

    norm_module = resolve_final_norm_module(language_model) if language_model is not None else None
    if norm_module is not None:
        try:
            hidden_state = norm_module(hidden_state)
        except TypeError:
            hidden_state = norm_module(hidden_state.to(device=device))
    elif language_model is not None:
        warnings.warn(
            "Could not resolve the final norm module of the language model; "
            "applying lm_head to unnormalized hidden states, logit-lens "
            "probabilities may be unreliable."
        )

    logits = lm_head(hidden_state)
    probs = torch.softmax(logits, dim=-1)

    # Use only the first `valid_token_count` positions (may be prefix of full seq)
    seq_len = probs.shape[1]
    if valid_token_count is not None:
        valid_token_count = int(valid_token_count)
        valid_token_count = max(1, min(seq_len, valid_token_count))
    else:
        valid_token_count = seq_len

    probs = probs[:, :valid_token_count, :]

    concept_token_ids = get_concept_token_ids(tokenizer, concept_text)

    if position_aggregation not in {"max", "mean", "last"}:
        position_aggregation = "max"

    def _aggregate_positions(x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T)
        if position_aggregation == "last":
            return x[:, -1]
        if position_aggregation == "mean":
            return x.mean(dim=1)
        return x.max(dim=1).values

    # Top-1 probability per position over the full vocabulary — used to
    # compute the concept probability *relative* to the layer's strongest
    # token, which is more comparable across layers than raw probability.
    per_pos_top1_probs, _per_pos_top1_ids = torch.max(probs, dim=-1)  # (B, T)

    if concept_token_ids:
        concept_probs = probs[:, :, concept_token_ids]
        # (B, T, K) -> best token-id per position
        per_pos_best_probs, per_pos_best_token_idx = torch.max(concept_probs, dim=-1)
        # Aggregate over positions
        concept_prob = _aggregate_positions(per_pos_best_probs)
        per_pos_relative = per_pos_best_probs / per_pos_top1_probs.clamp(min=1e-12)
        relative_prob = _aggregate_positions(per_pos_relative)

        # best_position / top_tokens always refer to a single position: the
        # last one for "last", otherwise the strongest position — even when
        # concept_token_probability is a positional mean.
        if position_aggregation == "last":
            best_pos = int(per_pos_best_probs.shape[1] - 1)
        else:
            best_pos = int(torch.argmax(per_pos_best_probs[0]).item())

        best_token_idx = int(per_pos_best_token_idx[0, best_pos].item())
        concept_token_id = int(concept_token_ids[best_token_idx])
        per_pos_array = per_pos_best_probs[0].tolist()
    else:
        max_per_pos_probs, max_per_pos_token_ids = torch.max(probs, dim=-1)
        concept_prob = _aggregate_positions(max_per_pos_probs)
        # Without concept tokens the "concept" is the top token itself.
        relative_prob = torch.ones_like(concept_prob)
        if position_aggregation == "last":
            best_pos = int(max_per_pos_probs.shape[1] - 1)
        else:
            best_pos = int(torch.argmax(max_per_pos_probs[0]).item())
        concept_token_id = int(max_per_pos_token_ids[0, best_pos].item())
        per_pos_array = max_per_pos_probs[0].tolist()

    k = min(int(top_k), probs.shape[-1])
    top_probs, top_token_ids = torch.topk(probs[:, best_pos, :], k=k, dim=-1)

    top_tokens: List[Dict[str, Any]] = []
    for prob, token_id in zip(top_probs[0].tolist(), top_token_ids[0].tolist()):
        top_tokens.append(
            {
                "token_id": int(token_id),
                "token": _decode_token(tokenizer, int(token_id)),
                "probability": float(prob),
            }
        )

    return {
        "concept_token_ids": [int(token_id) for token_id in concept_token_ids],
        "concept_token_id": concept_token_id,
        "concept_token": _decode_token(tokenizer, concept_token_id),
        "concept_token_probability": float(concept_prob[0].item()),
        "relative_probability": float(relative_prob[0].item()),
        "top1_token_probability": float(per_pos_top1_probs[0, best_pos].item()),
        "best_position": int(best_pos),
        "num_positions_scored": int(valid_token_count),
        "position_aggregation": position_aggregation,
        "per_position_concept_probs": [float(v) for v in per_pos_array],
        "top_tokens": top_tokens,
    }


def _to_serializable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(val) for val in value]
    return value


def write_layer_selection_debug(
    debug_dir: Path,
    sample_id: str,
    concept_text: str,
    layer_candidates: List[Dict[str, Any]],
    selected_layer: str,
    selected_token: str,
    selected_token_probability: float,
    logger: Optional[Any] = None,
    score_key: str = "concept_token_probability",
    score_label: str = "Concept token probability",
) -> None:
    """Write a JSON summary and a histogram for layer selection."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_sample_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", sample_id)

    summary = {
        "sample_id": sample_id,
        "concept_text": concept_text,
        "selected_layer": selected_layer,
        "selected_token": selected_token,
        "selected_token_probability": float(selected_token_probability),
        "layer_candidates": layer_candidates,
    }
    json_path = debug_dir / f"{safe_sample_id}_layer_selection.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(_to_serializable(summary), handle, indent=2, ensure_ascii=False)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency fallback
        if logger is not None:
            logger.warning(f"Could not import matplotlib for layer debug plot: {exc}")
        return

    layer_names = [item["layer_name"] for item in layer_candidates]
    scores = [float(item[score_key]) for item in layer_candidates]
    colors = ["#4C78A8" if layer != selected_layer else "#F58518" for layer in layer_names]

    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(layer_names)), 4.5))
    ax.bar(range(len(layer_names)), scores, color=colors)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.set_ylabel(score_label)
    ax.set_title(f"Layer selection for {concept_text} -> {selected_token}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    png_path = debug_dir / f"{safe_sample_id}_layer_selection.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
