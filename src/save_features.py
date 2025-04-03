import argparse
import os
import time
from typing import Any, Callable, Dict, List, Tuple
import shutil
import torch

from datasets import get_dataset_loader
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from helpers.utils import (clear_forward_hooks, clear_hooks_variables,
                           compute_time_left, set_seed, setup_hooks,
                           update_dict_of_list, save_dict_as_pickle)
from models import get_model_class 
from helpers.loading_cache import load_all_pickles
from helpers.post_process_embeding import extract_phrase_embeddings
from models.image_text_model import ImageTextModel


def move_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, tuple):
        return tuple(tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor for tensor in data)
    else:
        raise TypeError("Input must be a tensor or a tuple of tensors.")

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
        
        if args.dataset_name == "text":
            text = item["text"][0]  # for now we support batch size = 1
            inputs = model_class.preprocessor(
            instruction=text,
            response="",
            generation_mode=args.generation_mode,
            )
        else:
            text = item["text"][0]  # for now we support batch size = 1
            image_path = item["image"][0]
            
            inputs = model_class.preprocessor(
                instruction=text,
                image_file=image_path,
                response="",
                generation_mode=args.generation_mode,
                
            )

        if args.generation_mode:
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                  do_sample=True,
                  output_scores=True,
                  return_dict_in_generate=True,
                  temperature=0.3,
                  top_k=1,
            )
            #move_to_cpu_and_cleanup(out)
            scores = out.scores
            scores = move_to_cpu(scores)
            out = out.sequences
            out = move_to_cpu(out)

        else:
            out = model(**inputs).logits

        item["model_output"] = out
        input_len = (
            inputs["input_ids"].shape[1]
            if inputs["input_ids"].ndim > 1
            else inputs["input_ids"].shape[0]
        )
       # This is modification from original implementation, ChexAgent model only generate prediction , no input is repeted
        if args.slice_prediction:
            item["model_generated_output"] = out[:, input_len:]
            item["model_predictions"] = model_class.get_tokenizer().batch_decode(
            out[:, input_len:], skip_special_tokens=True
            )
        else:
            item["model_generated_output"] = out
            item["model_predictions"] = model_class.get_tokenizer().batch_decode(
            out, skip_special_tokens=True
            )
        del out
        item["scores"] = scores
        #item["token_of_interest"] = args.token_of_interest # This is becase we want to pass token of interest, I added
        if hook_return_functions is not None:
            for func in hook_return_functions:
                if func is not None:
                    hook_output = func(**item)
                    if hook_output:
                        item.update(hook_output)
        """
        cache_dir = args.cache_dir
        if cache_dir is not None:
            
            os.makedirs(cache_dir, exist_ok=True)
            if len(os.listdir(cache_dir))> 1: # If cache directory has files already just load them and return
                return load_all_pickles(cache_dir)


        else:
            raise(f"Cache duirectroy is{cache_dir}. It is not possible to svae save intermidiate file")
        save_dict_as_pickle(item, cache_dir )
        """
        if "save_hidden_states_noun_phrase" in args.hook_names : # With tis hook name we only extract the phrase embeddigns of all embedding 
            item = extract_phrase_embeddings(item, model_class)
            for key, value in item.items():
                if key in hook_data:
                    hook_data[key].extend(item[key])
                else:
                    hook_data[key] = item[key]
        else:
            hook_data = update_dict_of_list(item, hook_data)
        clear_hooks_variables()
        if (i + 1) % 100 == 0:
            time_left = compute_time_left(start_time, i, num_iterations)
            logger.info(
                f"Iteration: {i}/{num_iterations},  Estimated time left: {time_left:.2f} mins"
            )
        
    #hook_data = load_all_pickles(cache_dir)
    #shutil.rmtree(cache_dir)
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
