import argparse
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import torch

from datasets import get_dataset_loader
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from helpers.utils import (clear_forward_hooks, clear_hooks_variables,
                           compute_time_left, set_seed, setup_hooks,
                           update_dict_of_list)
from models import get_model_class
from models.image_text_model import ImageTextModel


@torch.no_grad()
def inference(
    loader: Callable,
    model_class: ImageTextModel,
    hook_return_function: Callable,
    device: torch.device,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Tuple[List[Dict[str, Any]], List[bool]]:

    num_iterations = len(loader)
    hook_data = {}
    model = model_class.get_model()
    start_time = time.time()
    for i, item in enumerate(loader):
        # Batchify: gather all texts and images in the batch
        texts = item["text"] if isinstance(item["text"], list) else [item["text"]]
        # Replace placeholder [concept] with --token_of_interest value if provided
        toi = getattr(args, "token_of_interest", None)
        if toi is not None and "cgdl" in getattr(args, "prompt_template", None):
            toi_str = str(toi).strip()
            texts = [t.replace("concept", toi_str) if isinstance(t, str) else t for t in texts]
        item["text"] = texts
        image_paths = item["image"] if isinstance(item["image"], list) else [item["image"]]

        # Build per-sample inputs
        per_sample_inputs = [
            model_class.preprocessor(
                instruction=texts[i],
                image_file=image_paths[i],
                response="",
                generation_mode=args.generation_mode,
            )
            for i in range(len(texts))
        ]

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
                            seqs, batch_first=True, padding_value=padding_value
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
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
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
            pad_positions = (row == pad_id).nonzero(as_tuple=False)
            if pad_positions.numel() > 0:
                input_lens.append(pad_positions[0].item())
            else:
                input_lens.append(L)
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
        if (i + 1) % 100 == 0:
            time_left = compute_time_left(start_time, i, num_iterations)
            logger.info(
                f"Iteration: {i}/{num_iterations},  Estimated time left: {time_left:.2f} mins"
            )
    return hook_data


if __name__ == "__main__":

    args = get_arguments()

    logger = setup_logger(log_file=os.path.join(args.save_dir, f"logs.log"))

    set_seed(args.seed)

    logger.info(f"Loading model: {args.model_name_or_path}")
    log_args(args, logger)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    hook_data = inference(
        loader=loader,
        model_class=model_class,
        device=device,
        hook_return_function=hook_return_functions,
        logger=logger,
        args=args,
    )

    clear_forward_hooks(model_class.model_)
    if hook_postprocessing_functions is not None:
        for func in hook_postprocessing_functions:
            if func is not None:
                func(data=hook_data, args=args, logger=logger)