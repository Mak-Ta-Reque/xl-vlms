import argparse
import os
import time
from typing import Any, Callable, Dict, List, Tuple

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


@torch.no_grad()
def inference(
    loader: Callable,
    model_class: ImageTextModel,
    hook_return_functions: List[Callable] | None,
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
            texts = [t.replace("[concept]", toi_str) if isinstance(t, str) else t for t in texts]
        item["text"] = texts
        image_paths = item["image"] if isinstance(item["image"], list) else [item["image"]]
        crop_locations = item.get("bbox", None)

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
        
        per_image_bboxes = _ensure_per_image_bbox_list(crop_locations, len(image_paths))

        for idx in range(len(image_paths)):
            img_ref = image_paths[idx]
            bbox = per_image_bboxes[idx]
            if bbox is not None:
                # open, crop (xyxy), clip to bounds
                img = Image.open(img_ref).convert("RGB") if not isinstance(img_ref, Image.Image) else img_ref
                W, H = img.size
                x1, y1, x2, y2 = bbox
                x1 = int(_clip(x1, 0, W - 1))
                y1 = int(_clip(y1, 0, H - 1))
                x2 = int(_clip(x2, x1 + 1, W))
                y2 = int(_clip(y2, y1 + 1, H))
                crop_img = img.crop((x1, y1, x2, y2))
                per_sample_inputs.append(
                    model_class.preprocessor(
                        instruction=texts[idx],
                        image_file= crop_img,
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