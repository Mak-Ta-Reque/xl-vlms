# Main function to generate noun phrases from model predictions and save results



import argparse
import os
import time
from typing import Any, Callable, Dict, List, Tuple
import shutil
import torch
import spacy
from datasets import get_dataset_loader
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from helpers.utils import (clear_forward_hooks, clear_hooks_variables,
                           compute_time_left, set_seed, setup_hooks,
                           update_dict_of_list, save_dict_as_pickle)
from models import get_model_class 
from helpers.loading_cache import load_all_pickles
from helpers.post_process_embeding import extract_phrase_embeddings, clean_string, modify_string
from models.image_text_model import ImageTextModel
nlp = spacy.load("en_core_web_sm")

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
    device: torch.device,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Tuple[List[Dict[str, Any]], List[bool]]:
    all_phrases =[]
    all_images = []
    num_iterations = len(loader)
    model = model_class.get_model()
    start_time = time.time()
    for i, item in enumerate(loader):
        

        text = args.prompt
        image_path = item["image"][0]
        
        inputs = model_class.preprocessor(
            instruction=text,
            image_file=image_path,
            response="",
            generation_mode=args.generation_mode,
        )

        out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                  do_sample=True,
                  output_scores=True,
                  return_dict_in_generate=True,
                  temperature=0.1,
                  top_k=1,
            )
       

        out = out.sequences
        input_len = (
            inputs["input_ids"].shape[1]
            if inputs["input_ids"].ndim > 1
            else inputs["input_ids"].shape[0]
        )
       # This is modification from original implementation, ChexAgent model only generate prediction , no input is repeted
        if args.slice_prediction:
           prediction = model_class.get_tokenizer().batch_decode(
            out[:, input_len:], skip_special_tokens=True
            )
        else:
            prediction = model_class.get_tokenizer().batch_decode(
            out, skip_special_tokens=True
            )
        doc = nlp(prediction[0])
        phrases = []
        for np in doc.noun_chunks:
            clean_text = clean_string(np.text)
            clean_text = modify_string(clean_text)
            if len(clean_text) >3 :
                #print(f"Phrase: {clean_text}")
                phrases.append(clean_text)
            else:

                continue
        images = image_path * len(phrases)
        all_phrases.extend(phrases)
        all_images.extend(images)
        if (i + 1) % 100 == 0:
            time_left = compute_time_left(start_time, i, num_iterations)
            logger.info(
                f"Iteration: {i}/{num_iterations},  Estimated time left: {time_left:.2f} mins"
            )
    return all_phrases, all_images
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

    
    loader = get_dataset_loader(
        dataset_name=args.dataset_name, logger=logger, args=args
    )

    phrases, images = inference(
        loader=loader,
        model_class=model_class,
        device=device,
        logger=logger,
        args=args,
    )
    output_dir = os.path.join(args.save_dir, "output") 
    os.makedirs(output_dir, exist_ok=True)
    image_file = os.path.join(output_dir, "images.txt")
    phrase_file = os.path.join(output_dir, "phrases.txt")
    with open(image_file, "w") as f:
        for image in images:
            f.write(f"{image}\n")
    with open(phrase_file, "w") as f:
        for phrase in phrases:
            f.write(f"{phrase}\n")
    print("Phrases and images saved to files.")

