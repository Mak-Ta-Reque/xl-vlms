
import argparse
import os
import random
import re
import time
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from datasets.constants import WORDS

__all__ = [
    "register_hooks",
    "clear_forward_hooks",
    "clear_hooks_variables",
    "hooks_postprocessing",
    "set_seed",
    "setup_hooks",
    "resolve_layer_path",
]


# Dictionary to store hidden states
HIDDEN_STATES = {}

def get_batch_lengths(model_output):
    """
    Given model_output as a list of lists (batch dimension), return a list of lengths for each batch item.
    Handles both tensors and lists.
    """
    lengths = []
    for item in model_output:
        if torch.is_tensor(item):
            lengths.append(item.shape[0])
        else:
            lengths.append(len(item))
    return lengths
def mean_skip_special_tokens(v: torch.Tensor, pred_tokens: torch.Tensor, special_ids: set) -> torch.Tensor:
    """
    Compute mean of hidden states per batch, skipping embeddings for tokens in special_ids.
    Args:
        v: (B, L, D) hidden states
        pred_tokens: (B, L) token ids
        special_ids: set of token ids to skip
    Returns:
        (B, 1, D) mean hidden states per batch
    """
    mean_vecs = []
    for b in range(v.shape[0]):
        row_hidden = v[b]
        row_pred = pred_tokens[b]
        keep_mask = torch.ones(row_pred.shape[0], dtype=torch.bool, device=row_pred.device)
        for sid in special_ids:
            keep_mask &= (row_pred != sid)
        kept = row_hidden[keep_mask]
        if kept.shape[0] > 0:
            mean_vecs.append(kept.mean(dim=0, keepdim=True))
        else:
            mean_vecs.append(torch.zeros((1, v.shape[2]), dtype=v.dtype, device=v.device))
    return torch.cat(mean_vecs, dim=0).unsqueeze(1)

def set_seed(seed_value=42):
    # Python random seed
    random.seed(seed_value)

    # NumPy random seed
    np.random.seed(seed_value)

    # PyTorch random seed
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def append_item_to_dict_of_list(key: str, value: Any, dictionary: Dict[str, Any]):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]
    return dictionary


def update_dict_of_list(item: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a batched item dict into an accumulator of lists.
    - If item[k] is a list/tuple (batched), extend the accumulator list.
    - Otherwise, append the single value.
    Also ensures that any torch.Tensors are detached and moved to CPU before storing.
    """

    def _to_cpu(x: Any) -> Any:
        return x.detach().cpu() if torch.is_tensor(x) else x

    for k, v in item.items():
        # Case 1: batched sequences
        if isinstance(v, (list, tuple)):
            values = [_to_cpu(x) for x in v]
            if k in data:
                data[k].extend(values)
            else:
                data[k] = list(values)

        # Case 2: nested mapping (e.g., dict of lists/sequences)
        elif isinstance(v, dict):
            for ek, ev in v.items():
                if k not in data:
                    data[k] = {}
                if isinstance(ev, (list, tuple)):
                    proc = [_to_cpu(x) for x in ev]
                else:
                    # Fallback: try to iterate (e.g., 1D tensor) else treat as single
                    if torch.is_tensor(ev):
                        try:
                            proc = [_to_cpu(x) for x in list(ev)]
                        except TypeError:
                            proc = [_to_cpu(ev)]
                    else:
                        try:
                            proc = [_to_cpu(x) for x in ev]  # type: ignore
                        except Exception:
                            proc = [_to_cpu(ev)]

                if ek in data[k]:
                    data[k][ek].extend(proc)
                else:
                    data[k][ek] = list(proc)

        # Case 3: single value
        else:
            val = _to_cpu(v)
            if k in data:
                data[k].append(val)
            else:
                data[k] = [val]
    return data



def fmatch(name: str, patterns: List[str], exact_match: bool = False) -> bool:
    if exact_match:
        return name in patterns
    else:
        # Convert patterns with '*' to proper regex expressions (where * means "any sequence of characters")
        regex_patterns = [
            re.compile(re.sub(r"\*", ".*", pattern)) for pattern in patterns
        ]
        return any([regex.search(name) for regex in regex_patterns])


def compute_time_left(start_time, iteration: int, num_iterations: int):
    elapsed_time = time.time() - start_time  # Time spent so far
    avg_time_per_iter = elapsed_time / iteration  # Average time per iteration
    remaining_iters = num_iterations - iteration
    time_left = avg_time_per_iter * remaining_iters  # Estimated time left
    return time_left / 60


def get_start_idx_generated_tokens(tokens: List[torch.Tensor]) -> int:
    if isinstance(tokens, list) and len(tokens) > 1:
        total_len = torch.cat(tokens, dim=1).shape[1]
        idx = tokens[0].shape[1] - total_len
    else:
        # teacher forcing mode
        v = v[0]
        idx = 0
    return idx  # generated tokens start after the prompt, count from last


def save_hidden_states(module_name: str = "", **kwargs: Any):
    """
    Save module output hidden states. In case of autoregressive, make sure the kv caching is enabled.
    """
    global HIDDEN_STATES

    def hook(module, input, output):
        if isinstance(output, tuple):  # e.g residual streams output is a tuple
            output = output[0]
        output = output.detach().cpu()
        if module_name in HIDDEN_STATES:
            HIDDEN_STATES[module_name].append(output)
        else:
            HIDDEN_STATES[module_name] = [output]

    return hook


def apply_steering_vector(
    x: torch.Tensor,
    vector: torch.Tensor,
    alpha: float = 1,
    only_generated_tokens: bool = False,
    include_last_prompt_token: bool = False,
    start_prompt_token_idx: int = 0,
) -> torch.Tensor:
    if x.shape[1] > 1:
        if only_generated_tokens:
            return x
        if include_last_prompt_token:
            start_prompt_token_idx = -1
        if start_prompt_token_idx > 0 or start_prompt_token_idx == -1:
            x_ = x[:, start_prompt_token_idx:, :]
            x_ = x_ + alpha * vector.to(x_.device).to(x_.dtype)
            x[:, start_prompt_token_idx:, :] = x_
            return x
    x = x + alpha * vector.to(x.device).to(x.dtype)
    # import ipdb; ipdb.set_trace()
    return x


def shift_hidden_states(
    vector: torch.Tensor = None,
    operation: str = "add",
    alpha: float = 1,
    only_generated_tokens: bool = False,
    include_last_prompt_token: bool = False,
    start_prompt_token_idx: int = 0,
    **kwargs: Any,
):
    """
    Shift features in the vector's direction.
    """
    if "add" in operation:

        def hook(module, input, output):
            if isinstance(output, tuple):  # e.g. in the residual stream
                output_ = apply_steering_vector(
                    output[0],
                    vector,
                    alpha=alpha,
                    only_generated_tokens=only_generated_tokens,
                    include_last_prompt_token=include_last_prompt_token,
                    start_prompt_token_idx=start_prompt_token_idx,
                )
                return (output_,) + output[1:]
            else:
                output = apply_steering_vector(
                    output,
                    vector,
                    alpha=alpha,
                    only_generated_tokens=only_generated_tokens,
                    include_last_prompt_token=include_last_prompt_token,
                    start_prompt_token_idx=start_prompt_token_idx,
                )
                return output

    else:
        raise NotImplementedError(
            f"Only the following steering operation are supported: add, got {operation}"
        )

    return hook


def extract_token_of_interest_states(
    tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    token_of_interest_idx: Union[int, torch.Tensor] = None,
    token_of_interest_start_token: Union[int, List[int], Tuple[int, ...], torch.Tensor] = 0,
) -> Tuple[torch.Tensor]:

    # Optionally crop sequences starting from a given token position. If a batch of
    # start indices is provided, slice per-sample and pad back to a uniform length.
    if isinstance(token_of_interest_start_token, (list, tuple, torch.Tensor)):
        # Align lengths first so per-sample slicing uses the same base lengths
        if pred_tokens.shape[1] > tokens.shape[1]:
            pred_tokens = pred_tokens[:, -tokens.shape[1] :]
        elif pred_tokens.shape[1] < tokens.shape[1]:
            tokens = tokens[:, -pred_tokens.shape[1] :]

        B, L, D = tokens.shape
        # Convert to tensor of shape (B,)
        starts = (
            token_of_interest_start_token
            if isinstance(token_of_interest_start_token, torch.Tensor)
            else torch.tensor(token_of_interest_start_token)
        )
        assert (
            starts.numel() == B
        ), f"token_of_interest_start_token must have length {B}, got {starts.numel()}"
        starts = starts.to(torch.long)

        # Slice per-sample, allowing negative indexing like python: -k means L-k
        tokens_list: List[torch.Tensor] = []
        preds_list: List[torch.Tensor] = []
        for b in range(B):
            s = starts[b].item()
            s = s if s >= 0 else max(0, L + s)
            s = min(max(0, s), L)  # clamp to [0, L]
            tokens_list.append(tokens[b, s:, :])
            preds_list.append(pred_tokens[b, s:])

        # Pad back to uniform length
        max_len = max(t.shape[0] for t in tokens_list) if tokens_list else 0
        if max_len == 0:
            # Degenerate case; keep shapes consistent
            tokens = tokens[:, :0, :]
            pred_tokens = pred_tokens[:, :0]
        else:
            padded_tokens = tokens.new_zeros((B, max_len, D))
            padded_preds = pred_tokens.new_full((B, max_len), fill_value=-1)
            for b in range(B):
                l = tokens_list[b].shape[0]
                if l > 0:
                    padded_tokens[b, :l, :] = tokens_list[b]
                    padded_preds[b, :l] = preds_list[b]
            tokens, pred_tokens = padded_tokens, padded_preds
    elif token_of_interest_start_token != 0:
        # Scalar start index: simple slice for all samples
        tokens = tokens[:, token_of_interest_start_token:]
        pred_tokens = pred_tokens[:, token_of_interest_start_token:]

    # Concider only text, no preds tokens for image tokens
    if pred_tokens.shape[1] > tokens.shape[1]:
        pred_tokens = pred_tokens[
            :, -tokens.shape[1] :
        ]  # e.g. in case of language_model.lm_head only the hidden states for generated tokens are saved
    elif pred_tokens.shape[1] < tokens.shape[1]:
        tokens = tokens[:, -pred_tokens.shape[1] :]

    assert (
        token_of_interest_idx is not None
    ), f"Please provide the token_of_interest_idx, got {token_of_interest_idx}"

    # If the token_of_interest splits into different ids, we consider the first one (while skipping eos/bos tokens)
    if not isinstance(token_of_interest_idx, torch.Tensor):
        token_of_interest_idx = torch.tensor([token_of_interest_idx])
    token_of_interest_idx = token_of_interest_idx.to(pred_tokens.device)

    # Step 1: Find where the tokens of interest exist in the batch (B, L)
    token_of_interest_batch_presence = torch.isin(
        pred_tokens, token_of_interest_idx
    )  # (B, L)
    # Step 2: Get the first occurrence index for each sequence
    token_of_interest_batch_first_pos = torch.argmax(
        token_of_interest_batch_presence.long(), dim=1
    )  # (B,)

    # Step 3: Mask for sequences with no token of interest
    no_token_found_mask = ~token_of_interest_batch_presence.any(dim=1)

    # Set the position to -1 if no token of interest is found
    token_of_interest_batch_first_pos[no_token_found_mask] = -1

    # Step 4: Now handle indexing into `v` based on the first position
    # Extract v at the first position for each batch (B,)
    # Select only valid positions in `v`
    v_selected = tokens[
        range(tokens.shape[0]),
        token_of_interest_batch_first_pos.clamp(min=0).to(tokens.device),
    ].unsqueeze(1)
    return v_selected, ~no_token_found_mask


def extract_states_before_special_tokens(
    tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    end_special_tokens: List[str],
    tokenizer: Callable,
    token_of_interest_start_token: Union[int, List[int], Tuple[int, ...], torch.Tensor] = 0,
) -> Tuple[torch.Tensor]:
    # Allow scalar or per-sample start indices; slice accordingly
    if isinstance(token_of_interest_start_token, (list, tuple, torch.Tensor)):
        # Align lengths first so per-sample slicing uses the same base lengths
        if pred_tokens.shape[1] > tokens.shape[1]:
            pred_tokens = pred_tokens[:, -tokens.shape[1] :]
        elif pred_tokens.shape[1] < tokens.shape[1]:
            tokens = tokens[:, -pred_tokens.shape[1] :]

        B, L, D = tokens.shape
        starts = (
            token_of_interest_start_token
            if isinstance(token_of_interest_start_token, torch.Tensor)
            else torch.tensor(token_of_interest_start_token)
        )
        assert (
            starts.numel() == B
        ), f"token_of_interest_start_token must have length {B}, got {starts.numel()}"
        starts = starts.to(torch.long)

        tokens_list: List[torch.Tensor] = []
        preds_list: List[torch.Tensor] = []
        for b in range(B):
            s = starts[b].item()
            s = s if s >= 0 else max(0, L + s)
            s = min(max(0, s), L)
            tokens_list.append(tokens[b, s:, :])
            preds_list.append(pred_tokens[b, s:])

        max_len = max(t.shape[0] for t in tokens_list) if tokens_list else 0
        if max_len == 0:
            tokens = tokens[:, :0, :]
            pred_tokens = pred_tokens[:, :0]
        else:
            padded_tokens = tokens.new_zeros((B, max_len, D))
            padded_preds = pred_tokens.new_full((B, max_len), fill_value=-1)
            for b in range(B):
                l = tokens_list[b].shape[0]
                if l > 0:
                    padded_tokens[b, :l, :] = tokens_list[b]
                    padded_preds[b, :l] = preds_list[b]
            tokens, pred_tokens = padded_tokens, padded_preds
    elif token_of_interest_start_token != 0:
        # e.g. consider only the answers
        tokens = tokens[:, token_of_interest_start_token:]
        pred_tokens = pred_tokens[:, token_of_interest_start_token:]

    # Concider only text, no preds tokens for image tokens
    if pred_tokens.shape[1] > tokens.shape[1]:
        pred_tokens = pred_tokens[
            :, -tokens.shape[1] :
        ]  # e.g. in case of language_model.lm_head only the hidden states for generated tokens are saved
    elif pred_tokens.shape[1] < tokens.shape[1]:
        tokens = tokens[:, -pred_tokens.shape[1] :]

    assert end_special_tokens is not None and isinstance(
        end_special_tokens, list
    ), f"Please provide the list of token_of_interest, got {end_special_tokens}"

    # If the token_of_interest splits into different ids, we consider the first one (while skipping eos/bos tokens)
    end_special_tokens_idx = torch.tensor(
        [
            tokenizer.encode(tok, add_special_tokens=False)[0]
            for tok in end_special_tokens
        ]
    ).to(pred_tokens.device)

    # Step 1: Find where the tokens of interest exist in the batch (B, L)
    token_of_interest_batch_presence = torch.isin(
        pred_tokens, end_special_tokens_idx
    )  # (B, L)
    # Step 2: Get the first occurrence index for each sequence
    token_of_interest_batch_first_pos = torch.argmax(
        token_of_interest_batch_presence.long(), dim=1
    )  # (B,)

    # Step 3: Mask for sequences with no token of interest
    no_token_found_mask = ~token_of_interest_batch_presence.any(dim=1)

    # Set the position to -1 if no token of interest is found
    token_of_interest_batch_first_pos[no_token_found_mask] = -1

    # Step 4: Now handle indexing into `v` based on the first position
    # Extract v at the first position for each batch (B,)
    # Select only valid positions in `v`
    v_selected = (
        tokens[
            range(tokens.shape[0]),
            : token_of_interest_batch_first_pos.to(tokens.device),
        ]
        .mean(1)
        .unsqueeze(1)
    )
    return v_selected, no_token_found_mask


def get_hidden_states(
    token_idx: int = None,
    token_start_end_idx: List[List[int]] = None,
    extract_token_of_interest: bool = False,
    token_of_interest_start_token: int = 0,
    extract_before_special_tokens: bool = False,
    save_only_generated_tokens: bool = False,
    mean: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    hidden_states = {}
    output = {}
    for k, v in HIDDEN_STATES.items():
        if isinstance(v, list) and len(v) > 1:
            v = torch.cat(v, dim=1)
            #v = v / v.norm(p=2, dim=1, keepdim=True)
            # Extra added for balancing distance avoiding neighbour tokens
        else:
            v = v[0]
        if token_idx is not None:
            v = v[:, token_idx, :].unsqueeze(1)
        elif token_start_end_idx is not None:
            v = v[:, int(token_start_end_idx[0]) : int(token_start_end_idx[1]), :]
        elif extract_token_of_interest:

            if save_only_generated_tokens:
                token_of_interest_start_token_batch = []

                batch_lengths = get_batch_lengths(kwargs["model_generated_output"])
                for length in batch_lengths:
                    start_idx_generated_tokens = -length
                    token_of_interest_start_token_batch.append(start_idx_generated_tokens)
                toi_start = token_of_interest_start_token_batch
            else:
                # Use the provided scalar or batch value from function args
                toi_start = token_of_interest_start_token

            # Normalize model_output to 2D LongTensor (B, L), padding lists with -1
            pred_tokens_input = kwargs["model_output"]
            if isinstance(pred_tokens_input, list):
                seqs = []
                for p in pred_tokens_input:
                    if not torch.is_tensor(p):
                        p = torch.tensor(p, dtype=torch.long)
                    else:
                        p = p.to(dtype=torch.long)
                    seqs.append(p)
                pred_tokens_tensor = torch.nn.utils.rnn.pad_sequence(
                    seqs, batch_first=True, padding_value=-1
                )
            else:
                pred_tokens_tensor = pred_tokens_input
                if pred_tokens_tensor.dim() == 1:
                    pred_tokens_tensor = pred_tokens_tensor.unsqueeze(0)
                pred_tokens_tensor = pred_tokens_tensor.to(dtype=torch.long)

            v, token_of_interest_mask = extract_token_of_interest_states(
                tokens=v,
                pred_tokens=pred_tokens_tensor,
                token_of_interest_idx=kwargs.get("token_of_interest_idx", None),
                token_of_interest_start_token=toi_start,
            )
            output["token_of_interest_mask"] = token_of_interest_mask
            output["image"] = kwargs["image"]
        elif extract_before_special_tokens:

            if save_only_generated_tokens:
                # Build scalar or per-sample start indices for generated tokens
                mgo = kwargs.get("model_generated_output", None)
                if isinstance(mgo, list):
                    token_of_interest_start_token = []
                    for model_output in mgo:
                        length = (
                            model_output.shape[0]
                            if torch.is_tensor(model_output)
                            else len(model_output)
                        )
                        token_of_interest_start_token.append(-int(length))
                elif torch.is_tensor(mgo):
                    token_of_interest_start_token = -int(mgo.shape[1])

            # Normalize model_output to 2D LongTensor (B, L), padding lists with -1
            pred_tokens_input = kwargs["model_output"]
            if isinstance(pred_tokens_input, list):
                seqs = []
                for p in pred_tokens_input:
                    if not torch.is_tensor(p):
                        p = torch.tensor(p, dtype=torch.long)
                    else:
                        p = p.to(dtype=torch.long)
                    seqs.append(p)
                pred_tokens_tensor = torch.nn.utils.rnn.pad_sequence(
                    seqs, batch_first=True, padding_value=-1
                )
            else:
                pred_tokens_tensor = pred_tokens_input
                if pred_tokens_tensor.dim() == 1:
                    pred_tokens_tensor = pred_tokens_tensor.unsqueeze(0)
                pred_tokens_tensor = pred_tokens_tensor.to(dtype=torch.long)

            v, token_of_interest_mask = extract_states_before_special_tokens(
                tokens=v,
                pred_tokens=pred_tokens_tensor,
                end_special_tokens=kwargs["end_special_tokens"],
                tokenizer=kwargs["tokenizer"],
                token_of_interest_start_token=token_of_interest_start_token,
            )
            output["token_of_interest_mask"] = torch.ones_like(
                token_of_interest_mask
            ).bool()
            output["image"] = kwargs["image"]

        else:
            if save_only_generated_tokens:
                # Slice v to only generated tokens using per-sample starts
                mgo = kwargs.get("model_generated_output", None)
                
                B, L, D = v.shape
                if isinstance(mgo, list):
                    batch_lengths = get_batch_lengths(mgo)
                    starts = torch.tensor([-int(length) for length in batch_lengths], dtype=torch.long)
                elif torch.is_tensor(mgo):
                    starts = torch.full((B,), -int(mgo.shape[1]))
                else:
                    starts = torch.zeros((B,), dtype=torch.long)

                tokens_list: List[torch.Tensor] = []
                for b in range(B):
                    s = starts[b].item()
                    s = s if s >= 0 else max(0, L + s)
                    s = min(max(0, s), L)
                    tokens_list.append(v[b, s:, :])

                max_len = max(t.shape[0] for t in tokens_list) if tokens_list else 0
                if max_len == 0:
                    v = v[:, :0, :]
                else:
                    padded_tokens = v.new_zeros((B, max_len, D))
                    for b in range(B):
                        l = tokens_list[b].shape[0]
                        if l > 0:
                            padded_tokens[b, :l, :] = tokens_list[b]
                    v = padded_tokens

                if mean:
                    tokenizer = kwargs["tokenizer"]
                    special_ids = set(tokenizer.all_special_ids)
                    special_ids.add(106)  # 106 is end of sentence for qween model
                    pred_tokens_input = kwargs.get("model_generated_output", None)
                    # Normalize model_output to 2D LongTensor (B, Lp)
                    if isinstance(pred_tokens_input, list):
                        seqs = []
                        for p in pred_tokens_input:
                            p = torch.tensor(p, dtype=torch.long) if not torch.is_tensor(p) else p.to(dtype=torch.long)
                            seqs.append(p)
                        pred_tokens_tensor = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=-1)
                    else:
                        pred_tokens_tensor = pred_tokens_input
                        if pred_tokens_tensor.dim() == 1:
                            pred_tokens_tensor = pred_tokens_tensor.unsqueeze(0)
                        pred_tokens_tensor = pred_tokens_tensor.to(dtype=torch.long)
                    pred_tokens_tensor = pred_tokens_tensor.to(v.device)
                    # Align pred tokens to v
                    if pred_tokens_tensor.shape[1] > v.shape[1]:
                        pred_tokens_tensor = pred_tokens_tensor[:, -v.shape[1]:]
                    elif pred_tokens_tensor.shape[1] < v.shape[1]:
                        v = v[:, -(v.shape[1] - (v.shape[1] - pred_tokens_tensor.shape[1])):, :]
                    v = mean_skip_special_tokens(v, pred_tokens_tensor, special_ids)
            else:
                if mean:
                    tokenizer = kwargs["tokenizer"]
                    special_ids = set(tokenizer.all_special_ids)
                    pred_tokens_input = kwargs.get("model_output", None)
                    # Normalize model_output to 2D LongTensor (B, Lp)
                    if isinstance(pred_tokens_input, list):
                        seqs = []
                        for p in pred_tokens_input:
                            p = torch.tensor(p, dtype=torch.long) if not torch.is_tensor(p) else p.to(dtype=torch.long)
                            seqs.append(p)
                        pred_tokens_tensor = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=-1)
                    else:
                        pred_tokens_tensor = pred_tokens_input
                        if pred_tokens_tensor.dim() == 1:
                            pred_tokens_tensor = pred_tokens_tensor.unsqueeze(0)
                        pred_tokens_tensor = pred_tokens_tensor.to(dtype=torch.long)
                    pred_tokens_tensor = pred_tokens_tensor.to(v.device)
                    # Align pred tokens to v
                    if pred_tokens_tensor.shape[1] > v.shape[1]:
                        pred_tokens_tensor = pred_tokens_tensor[:, -v.shape[1]:]
                    elif pred_tokens_tensor.shape[1] < v.shape[1]:
                        v = v[:, -(v.shape[1] - (v.shape[1] - pred_tokens_tensor.shape[1])):, :]
                    v = mean_skip_special_tokens(v, pred_tokens_tensor, special_ids)

        hidden_states[k] = [v[i] for i in range(v.shape[0])]
    output["hidden_states"] = hidden_states
    return output


def save_hidden_states_to_file(
    data: Dict[str, Any],
    data_keys: List[str] = ["hidden_states"],
    hook_name: str = "",
    args: argparse.Namespace = None,
    logger: Callable = None,
) -> None:
    saved_data = {}

    for data_key in data.keys():
        if data_key in data_keys:
            assert (
                data_key in data
            ), f"{data_key} not found in data, there is only: {data.keys()}"

            saved_data[data_key] = data[data_key]  # List[Any]
    file_name = os.path.join(
        args.save_dir, "features", f"{hook_name}_{args.save_filename}.pth"
    )
    # Ensure parent directory exists before saving
    parent_dir = os.path.dirname(file_name)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        msg = f"Created directory {parent_dir} for saving hidden states."
        logger.info(msg)

    torch.save(saved_data, file_name)
    if logger is not None:
        logger.info(f"Saving data to: {file_name}")


def save_analysis_to_file(
    data: Dict[str, Any],
    analysis_saving_path: str,
    data_keys: List[str] = ["text_grounding"],
    logger: Callable = None,
) -> None:
    saved_data = {}

    for data_key in data_keys:
        assert (
            data_key in data
        ), f"{data_key} not found in data, there is only: {data.keys()}"

        saved_data[data_key] = data[data_key]  # List[Any]
    file_name = f"{analysis_saving_path}.pth"
    torch.save(saved_data, file_name)
    if logger is not None:
        logger.info(f"Saving analysis data to: {file_name}")


def register_hooks(
    model: Callable,
    modules_to_hook: List[str],
    hook_name: str = "save_hidden_states",
    tokenizer: Callable = None,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Callable:
    hook_function, hook_return_function = None, None
    if "save_hidden_states" == hook_name:
        # Save the hidden states of all tokens in the sequence
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            save_only_generated_tokens=args.save_only_generated_tokens,
            tokenizer = tokenizer,
        )
    elif "save_hidden_states_mean" == hook_name:
        # Save the hidden states of all tokens in the sequence
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            save_only_generated_tokens=args.save_only_generated_tokens,
            tokenizer = tokenizer,
            mean = True,
        )
    elif "save_hidden_states_logit_lens" == hook_name:
        # Keep per-position hidden states: logit lens scores every sequence
        # position, so mean-pooling here would discard the needed information.
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            save_only_generated_tokens=args.save_only_generated_tokens,
            tokenizer = tokenizer,
            mean = False,
        )
    elif "save_hidden_states_given_token_idx" == hook_name:
        # Save the hidden states at given token index
        hook_function = save_hidden_states
        hook_return_function = partial(get_hidden_states, token_idx=args.token_idx)
    elif "save_hidden_states_given_token_start_end_idx" == hook_name:
        # Save the hidden states of tokens between start and end index
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states, token_start_end_idx=args.token_start_end_idx
        )
    elif "save_hidden_states_for_token_of_interest" == hook_name:
        # Save the hidden states of tokens between start and end index
        token_of_interest = args.token_of_interest

        # Get index in tokenizer vocabulary for token of interest
        # Some tokenizers encode/decode space along with token, so include index of whitespace + token_of_interest
        tokens_of_interest = set(
            [
                token_of_interest,
                token_of_interest.capitalize(),
                token_of_interest.lower(),
            ]
        )
        token_of_interest_idx = args.token_of_interest_idx
        if token_of_interest_idx is None:
            token_of_interest_idx = torch.tensor(
                [
                    tokenizer.encode(tok, add_special_tokens=False)[0]
                    for tok in tokens_of_interest
                ]
            )
            #prefixed_token =  tokenizer.encode("beatiful " + token_of_interest, add_special_tokens=False)
            #suffix_token = tokenizer.encode( token_of_interest + " long", add_special_tokens=False)
            #im_start_token = tokenizer.encode("<|im_start|> "+ token_of_interest, add_special_tokens=False)
      
            check_token = tokenizer.encode(" " + token_of_interest, add_special_tokens=False)[0]
            #dcode_tok_106 = tokenizer.decode([106, 0])
            if token_of_interest in tokenizer.decode([check_token]): # Check if this check_token is only encoding whitespace
                token_of_interest_idx = torch.tensor(list(token_of_interest_idx) + [check_token])
            

        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            extract_token_of_interest=True,
            token_of_interest_idx=token_of_interest_idx,
            token_of_interest_start_token=args.token_of_interest_start_token,
            save_only_generated_tokens=args.save_only_generated_tokens,
        )
    elif "save_hidden_states_for_token_of_interest_class" == hook_name:
        # Save the hidden states of tokens between start and end index
        token_of_interest = []
        tokens = list(WORDS[args.token_of_interest_class])
        for tok in tqdm(tokens):
            toks = [
                tok,
                tok.capitalize(),
                tok.lower(),
            ]
            token_of_interest.extend(toks)
        tokens_of_interest = list(set(token_of_interest))

        token_of_interest_idx = args.token_of_interest_idx
        if token_of_interest_idx is None:
            token_of_interest_idx = torch.tensor(
                [
                    tokenizer.encode(tok, add_special_tokens=False)[0]
                    for tok in tokens_of_interest
                ]
            )
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            extract_token_of_interest=True,
            token_of_interest_idx=token_of_interest_idx,
            token_of_interest_start_token=args.token_of_interest_start_token,
            save_only_generated_tokens=args.save_only_generated_tokens,
        )
    elif "save_hidden_states_before_special_tokens" == hook_name:
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            extract_before_special_tokens=True,
            end_special_tokens=args.end_special_tokens,
            tokenizer=tokenizer,
            save_only_generated_tokens=args.save_only_generated_tokens,
        )
    elif "shift_hidden_states" in hook_name:
        operation = ""
        if "add" in hook_name:
            operation = "add"
        else:
            raise NotImplementedError(
                f"Please provide a valid operation. Got {hook_name}"
            )

        only_generated_tokens = "only_generated" in hook_name
        include_last_prompt_token = "last_prompt_token" in hook_name

        vector = torch.load(args.shift_vector_path)[args.shift_vector_key]
        hook_function = partial(
            shift_hidden_states,
            vector=vector,
            operation=operation,
            alpha=args.steering_alpha,
            only_generated_tokens=only_generated_tokens,
            include_last_prompt_token=include_last_prompt_token,
            start_prompt_token_idx=args.start_prompt_token_idx_steering,
        )
    else:
        warnings.warn(f"{hook_name} is not supported. No hooks attached to model.")
    if hook_function is not None:
        hooked_modules = []
        """
        Log all the layer of the model
        
        """
        

        for name, module in model.named_modules():
            if fmatch(
                name, modules_to_hook, exact_match=args.exact_match_modules_to_hook
            ):
                module.register_forward_hook(hook_function(module_name=name))
                hooked_modules.append(name)
        if logger is not None:
            logger.info(f"Apply {hook_name} to hooked_modules: {hooked_modules}")

    return hook_return_function


def hooks_postprocessing(
    hook_name: str = "save_hidden_states", args: argparse.Namespace = None
) -> Callable:
    hook_postprocessing_function = None
    if "save_hidden_states" in hook_name:

        data_keys = ["hidden_states", "image"]
        # temp change
        data_keys = ["concept", "hidden_states", "image", "model_predictions", "bbox", "seg_mask_rle", "is_concept"]

        if "token_of_interest" in hook_name:
            data_keys.append("token_of_interest_mask")
        hook_postprocessing_function = partial(
            save_hidden_states_to_file,
            args=args,
            data_keys=data_keys,
            hook_name=hook_name,
        )
    elif "vqav2_accuracy" in hook_name:
        import metrics
        hook_postprocessing_function = metrics.get_metric(
            metric_name="vqav2_accuracy", args=args
        )

    elif "captioning_metrics" in hook_name:
        import metrics
        hook_postprocessing_function = metrics.get_metric(
            metric_name="captioning_metrics", args=args
        )
    else:
        warnings.warn(f"{hook_name} is not supported. No hooks attached to model.")

    return hook_postprocessing_function


def clear_forward_hooks(model: Callable) -> None:
    for module in model.modules():
        module._forward_hooks.clear()


def clear_hooks_variables():
    global HIDDEN_STATES
    HIDDEN_STATES = {}


def setup_hooks(
    model: Callable,
    modules_to_hook: List[str],
    hook_names: str,
    tokenizer: Callable = None,
    logger: Callable = None,
    args: argparse.Namespace = None,
):
    hook_return_functions, hook_postprocessing_functions = [], []
    for i, hook_name in enumerate(hook_names):
        if modules_to_hook is not None and i < len(modules_to_hook):
            modules_to_hook_ = modules_to_hook[i]
            assert isinstance(
                modules_to_hook_, list
            ), f"modules_to_hook_ must be of type list. modules_to_hook_: {modules_to_hook_}"
            hook_return_function = register_hooks(
                model=model,
                modules_to_hook=modules_to_hook_,
                hook_name=hook_name,
                tokenizer=tokenizer,
                logger=logger,
                args=args,
            )
        else:
            hook_return_function = None
        hook_postprocessing_function = hooks_postprocessing(
            hook_name=hook_name, args=args
        )

        hook_return_functions.append(hook_return_function)
        hook_postprocessing_functions.append(hook_postprocessing_function)

    return hook_return_functions, hook_postprocessing_functions


def resolve_layer_path(
    raw: str,
    base: str = "model.language_model.layers",
) -> str:
    """Expand LAYER_PATH shorthands into full module path strings.

    Accepted formats
    ----------------
    model.language_model.norm   direct dotted path → returned unchanged
    27                          single index       → <base>.27
    1,2,3,4,5,6                 comma list         → <base>.1;<base>.2;...
    10-20                       inclusive range    → <base>.10;<base>.11;...;<base>.20

    Multiple resolved paths are joined with ";" which matches the
    ``parse_list_of_lists`` format expected by ``--modules_to_hook``.
    """
    raw = raw.strip()

    def _prefix(token: str) -> str:
        return f"{base}.{token}.post_attention_layernorm" if re.fullmatch(r"\d+", token) else token

    # Bare range: N-M (no commas, no dots)
    range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", raw)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        step = 1 if end >= start else -1
        return ",".join(_prefix(str(i)) for i in range(start, end + step, step))

    # Comma-separated tokens (integers or dotted paths mixed)
    if "," in raw:
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        resolved = [_prefix(t) for t in tokens]
        return resolved[0] if len(resolved) == 1 else ",".join(resolved)

    # Single token (integer or direct path)
    return _prefix(raw)