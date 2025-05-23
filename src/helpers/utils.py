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

import metrics
from datasets.constants import WORDS
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from PIL import Image

__all__ = [
    "register_hooks",
    "clear_forward_hooks",
    "clear_hooks_variables",
    "hooks_postprocessing",
    "set_seed",
    "setup_hooks",
]


# Dictionary to store hidden states
HIDDEN_STATES = {}


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
    for k, v in item.items():
        if k in data:
            data[k].append(v)
        else:
            data[k] = [v]
    return data

def _load_all_pickles(directory: str) -> Dict[str, Any]:
    """
    Loads all pickle files from a directory and merges them into a single dictionary.

    Args:
        directory (str): Path to the directory containing .pkl files.

    Returns:
        Dict[str, Any]: Merged dictionary containing all loaded data.
    """
    combined_data = {}  # Initialize empty dictionary

    # List all .pkl files in the directory
    pickle_files = [f for f in os.listdir(directory) if f.endswith(".pkl")]

    # Load and merge each file
    for file in pickle_files:
        file_path = os.path.join(directory, file)
        loaded_data = torch.load(file_path)  # Load the dictionary from file
        combined_data = update_dict_of_list(loaded_data, combined_data)  # Merge data

    return combined_data


def save_dict_as_pickle(data, save_dir):
    """
    Saves a dictionary as a pickle file using PyTorch's torch.save(),
    handling any tensor values properly.

    Args:
        data (dict): The dictionary to save.
        save_dir (str): The directory where the file should be saved.

    Returns:
        str: The path of the saved file.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    # Extract filename from "img_id"
    if "img_id" not in data or not data["img_id"]:
        raise ValueError("The dictionary must contain a non-empty 'img_id' key.")

    filename = f"{data['img_id'][0]}.pkl"
    file_path = os.path.join(save_dir, filename)

    # Save the dictionary using torch.save()
    torch.save(data, file_path)

    return


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
        m = module
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
    token_of_interest_start_token: int = 0,
) -> Tuple[torch.Tensor]:

    if token_of_interest_start_token != 0:
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
    # Token of interest could be mutiple variation, we took the first one/ last one / or anything 
    # Additional implemention: check if the tokens of interset a subset

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



def extract_tokens_of_interest_states(
    tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    tokens_of_interest_idx: Union[int, torch.Tensor] = None,
    token_of_interest_start_token: int = 0,
) -> Tuple[torch.Tensor]:
    def _ordered_subset_mask_batch(main_arr, sub_arr):
    # Get the batch size (first dimension)
        batch_size = main_arr.shape[0]
        
        # List to store masks for each batch
        masks = []
        
        # Loop through each batch and generate its corresponding mask
        for batch_idx in range(batch_size):
            mask = torch.zeros(main_arr.shape[1], dtype=torch.bool)  # Mask as a tensor of boolean values
            n, m = main_arr.shape[1], sub_arr.shape[0]
            
            # Check if the sub_arr can be found as a contiguous subsequence in this batch
            for i in range(n - m + 1):
                if torch.all(main_arr[batch_idx, i:i + m] == sub_arr):
                    # Mark the positions that match in the mask
                    mask[i:i + m] = True
                    break  # Once we find a match, no need to check further
            
            masks.append(mask)
    
        return torch.stack(masks)
    def ordered_subset_mask_batch(main_arr, sub_arr):
        batch_size = main_arr.shape[0]

        # List to store masks for each batch
        masks = []

        # Loop through each batch and generate its corresponding mask
        for batch_idx in range(batch_size):
            main_seq = main_arr[batch_idx]
            main_list = main_seq.tolist()

            # Create mask initialized to False
            mask = torch.zeros(main_seq.shape[0], dtype=torch.bool)

            # Initialize the index for searching sub_arr
            sub_idx = 0
            sub_len = len(sub_arr)

            # Loop through the main sequence and mark corresponding indices in the mask
            for i in range(len(main_list)):
                if sub_idx < sub_len and main_list[i] == sub_arr[sub_idx]:
                    mask[i] = True
                    sub_idx += 1  # Move to the next element in sub_arr

                if sub_idx == sub_len:
                    break  # Once all elements in sub_arr are matched, stop

            masks.append(mask)

        return torch.stack(masks)

    if token_of_interest_start_token != 0:
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

    assert (
        tokens_of_interest_idx is not None
    ), f"Please provide the token_of_interest_idx, got {tokens_of_interest_idx}"

    # If the token_of_interest splits into different ids, we consider the first one (while skipping eos/bos tokens)
    if not isinstance(tokens_of_interest_idx, torch.Tensor):
        tokens_of_interest_idx = torch.tensor([tokens_of_interest_idx])
    tokens_of_interest_idx = tokens_of_interest_idx.to(pred_tokens.device)

    # Step 1: Find where the tokens of interest exist in the batch (B, L)
    token_of_interest_batch_presence = torch.tensor([[item in tokens_of_interest_idx for item in large_list] for large_list in pred_tokens])
    
   # [[item in set(small_list) for item in large_list] 
   #         for large_list, small_list in zip(pred_tokens, tokens_of_interest_idx)]

    
    # Additional implemention: check if the tokens of interset a subset
    
    # Step 2: Get the first occurrence index for each sequence
    #tokens_of_interest_batch =   token_of_interest_batch_presence .sum(dim=1)

    # Step 3: Mask for sequences with no token of interest

    #print(tokens_of_interest_batch)
    # Set the position to -1 if no token of interest is found
    #token_of_interest_batch_first_pos[tokens_found_mask] = -1

    # Step 4: Now handle indexing into `v` based on the first position
    # Extract v at the first position for each batch (B,)
    # Select only valid positions in `v`
    v_selected = tokens#[token_of_interest_batch_presence].unsqueeze(0)
    return v_selected, token_of_interest_batch_presence


def extract_states_before_special_tokens(
    tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    end_special_tokens: List[str],
    tokenizer: Callable,
    token_of_interest_start_token: int = 0,
) -> Tuple[torch.Tensor]:
    if token_of_interest_start_token != 0:
        # e.g. consider only te answers
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
    extract_tokens_of_interest: bool = False,
    token_of_interest_start_token: int = 0,
    extract_before_special_tokens: bool = False,
    save_only_generated_tokens: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    hidden_states = {}
    output = {}
    for k, v in HIDDEN_STATES.items():
        if isinstance(v, list) and len(v) > 1:
            buffer_encoding = [v[0][:, -2:-1,:]]
            v = torch.cat(buffer_encoding + v[1:], dim=1)# Skip the buffer
        else:
            v = v[0]
            if v.shape[1]> 1: # if the first token geneation take the last index
                v = v [:, -2:-1,:]
        
        if token_idx is not None:
            v = v[:, token_idx, :].unsqueeze(1)
        elif token_start_end_idx is not None:
            v = v[:, int(token_start_end_idx[0]) : int(token_start_end_idx[1]), :]
        elif extract_token_of_interest:

            if save_only_generated_tokens:
                start_idx_generated_tokens = -kwargs["model_generated_output"].shape[1]
                token_of_interest_start_token = start_idx_generated_tokens

            v, token_of_interest_mask = extract_token_of_interest_states(
                tokens=v,
                pred_tokens=kwargs["model_output"],
                token_of_interest_idx=kwargs.get("token_of_interest_idx", None),
                token_of_interest_start_token=token_of_interest_start_token,
            )
            output["token_of_interest_mask"] = token_of_interest_mask
            output["image"] = kwargs["image"]
        elif extract_before_special_tokens:

            if save_only_generated_tokens:
                start_idx_generated_tokens = -kwargs["model_generated_output"].shape[1]
                token_of_interest_start_token = start_idx_generated_tokens

            v, token_of_interest_mask = extract_states_before_special_tokens(
                tokens=v,
                pred_tokens=kwargs["model_output"],
                end_special_tokens=kwargs["end_special_tokens"],
                tokenizer=kwargs["tokenizer"],
                token_of_interest_start_token=token_of_interest_start_token,
            )
            output["token_of_interest_mask"] = torch.ones_like(
                token_of_interest_mask
            ).bool()
            output["image"] = kwargs["image"]
        elif extract_tokens_of_interest:
            if save_only_generated_tokens:
                start_idx_generated_tokens = -kwargs["model_generated_output"].shape[1]
                token_of_interest_start_token = start_idx_generated_tokens
            v, token_of_interest_mask = extract_tokens_of_interest_states(
                tokens=v,
                pred_tokens=kwargs["model_output"],
                tokens_of_interest_idx=kwargs.get("tokens_of_interest_idx", None),
                token_of_interest_start_token=token_of_interest_start_token,
            )
            output["token_of_interest_mask"] = token_of_interest_mask
            output["image"] = kwargs["image"]
        
        else:
            
            pass
        hidden_states[k] = v
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

    if args.post_process_hidden:
       saved_data =  post_process_hidden(saved_data)

    file_name = os.path.join(
        args.save_dir, "features", f"{hook_name}_{args.save_filename}.pth"
    )
    torch.save(saved_data, file_name)
    if logger is not None:
        logger.info(f"Saving data to: {file_name}")


def post_process_hidden(hidden_states):
    all_hidden = []
    data = hidden_states
    for images, hiden_states, scores, predictions, token_pred in zip(data["image"], data["hidden_states"], data["scores"], data["model_predictions"], data['model_generated_output']):
        all_tokens = token_pred.tolist()[0]
        updated_hidden_states = {}
        for hidden_key, hidden_value in hiden_states.items():
            new_hidden_states = []
            for embeding, logits, token in zip(hidden_value.permute(1, 0, 2), scores, token_pred.permute(1, 0)):
                #print(token)
                #print(embeding.shape)
                #print(logits.shape)
                probabilities = torch.softmax(logits, dim=1).cpu()
                _lambdas = probabilities[:,all_tokens]
                
                l1_norm = torch.norm(_lambdas, p=1, dim=1, keepdim=True)  # Compute L1 norm along dim=1
                l1_normalized_lambda = _lambdas / l1_norm
                #print(l1_normalized_lambda)
                pos_index = all_tokens.index(token[0])
                pos_value = l1_normalized_lambda[0][pos_index]
                l1_normalized_lambda = -1 * l1_normalized_lambda
                l1_normalized_lambda[0][pos_index] = pos_value
                #negative_lambda = probabilities[:, negatve_indices]
                #positive_location = all_tokens.index(token.item())
                #lamdas = negative_lambda[0] * -1
                #lamdas = torch.cat((lamdas[:positive_location], positve_lambda[0], lamdas[positive_location:]))
                #coefficients =  lamdas.view(1, -1, 1)
                linear_combination = torch.sum(hidden_value * l1_normalized_lambda.unsqueeze(-1), dim=1) 
                new_hidden_states.append(linear_combination)  
            new_hidden_states = torch.stack(new_hidden_states)
            updated_hidden_states[hidden_key] = new_hidden_states.permute(1, 0, 2)
        all_hidden.append(updated_hidden_states)


    data["hidden_states"] = all_hidden
        
    return data


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
        hook_return_function = get_hidden_states

    elif hook_name in ["save_hidden_states_noun_phrase", "save_hidden_states_token", "save_hidden_states_sentence"]:
        # Save the hidden states of all tokens in the sequence
        hook_function = save_hidden_states
        hook_return_function = get_hidden_states

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
        token_of_interest = args.token_of_interest.strip()

        # Get index in tokenizer vocabulary for token of interest
        # Some tokenizers encode/decode space along with token, so include index of whitespace + token_of_interest
        tokens_of_interest = set(
            [
                token_of_interest,
                token_of_interest.capitalize(),
                token_of_interest.lower(),
                " " + token_of_interest,
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
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            extract_token_of_interest=True,
            token_of_interest_idx=token_of_interest_idx,
            token_of_interest_start_token=args.token_of_interest_start_token,
            save_only_generated_tokens=args.save_only_generated_tokens,
        )

    elif "save_hidden_states_for_tokens_of_interest" == hook_name:
        # Save the hidden states of tokens between start and end index
        tokens_of_interest = args.tokens_of_interest

        # Get index in tokenizer vocabulary for token of interest
        # Some tokenizers encode/decode space along with token, so include index of whitespace + token_of_interest

        #token_of_interest_idx = args.token_of_interest_idx
        #if token_of_interest_idx is None:
        tokens_of_interest_idx = torch.tensor(tokenizer.encode(tokens_of_interest, add_special_tokens=False))#[0]

        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            extract_tokens_of_interest=True,
            token_of_interest_start_token=args.token_of_interest_start_token,
            tokens_of_interest_idx=tokens_of_interest_idx,
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

        data_keys = ["hidden_states", "image", "text", 'model_predictions']
        # temp change
        #data_keys = ["hidden_states", "image", "model_predictions", "scores", "model_generated_output"]

        if "token_of_interest" in hook_name or "tokens_of_interest" in hook_name:
            data_keys.append("token_of_interest_mask")
        hook_postprocessing_function = partial(
            save_hidden_states_to_file,
            args=args,
            data_keys=data_keys,
            hook_name=hook_name,
        )
    elif "vqav2_accuracy" in hook_name:
        hook_postprocessing_function = metrics.get_metric(
            metric_name="vqav2_accuracy", args=args
        )

    elif "captioning_metrics" in hook_name:
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



def load_image_as_rgb(file_path, out_type="PIL"):
    """
    Load an image file (.dcm, .jpeg, .png) and return it as either a PIL image or a NumPy array.

    Parameters:
        file_path (str): The path to the image file.
        out_type (str): The desired output type, either "PIL" or "np".

    Returns:
        PIL.Image.Image or np.ndarray: The loaded image either as a PIL object or a NumPy array.
    """
    # Extract the file extension
    # @ remove the lavel befor the path
    ext = ""
    if isinstance(file_path, list):
     
        ext = os.path.splitext(file_path[0])[-1].lower()
    else:

        ext = os.path.splitext(file_path)[-1].lower()
        file_path = [file_path] # fixing the bug by brutforce


    if ext == ".dcm":
        # Load DICOM file
        dicom_data = pydicom.dcmread(file_path)
        image = dicom_data.pixel_array
        image_array = image
        
        # Convert grayscale to RGB (if it's single-channel)
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image] * 3, axis=-1)  # Convert to RGB

        if out_type == "PIL":
            # Convert NumPy array to PIL Image
            pil_image = Image.fromarray(image_array)
            # Ensure 3-channel image if grayscale
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            return pil_image
        elif out_type == "np":
            # Normalize to 0-255 for RGB conversion if necessary
            if np.max(image) > 255:
                image = (image / np.max(image)) * 255.0
            return image.astype(np.uint8)
        else:
            raise ValueError("Invalid out_type. Use 'PIL' or 'np'.")

    elif ext in [".jpeg", ".jpg", ".png"]:
        # Load JPEG or PNG file
        image = Image.open(file_path[0].split("@")[1]).convert("RGB")  # Ensure it's RGB

        if out_type == "PIL":
            return image
        elif out_type == "np":
            return np.array(image)
        else:
            raise ValueError("Invalid out_type. Use 'PIL' or 'np'.")

    else:
        raise ValueError(f"Unsupported file format: {ext}")