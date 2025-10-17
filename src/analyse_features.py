import copy
import os

import torch

from analysis import analyse_features
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from models import get_model_class

def concept_decompostion(args, model_subset, logger):
   
    analyse_features(
        analysis_name=args.analysis_name,
        logger=logger,
        model_class=model_subset,
        device=device,
        args=args,
    )


if __name__ == "__main__":

    args = get_arguments()

    logger = setup_logger(log_file=os.path.join(args.save_dir, f"logs.log"))
    log_args(args, logger)

    device = torch.device(args.device)

    model_class = get_model_class(
        args.model_name_or_path,
        args.processor_name,
        device=device,
        logger=logger,
        args=args,
    )
    lm_head = model_class.get_lm_head().float()
    lm_head = lm_head.to(device)
    tokenizer = model_class.get_tokenizer()
    # move to the device
    #tokenizer = tokenizer.to(device)
    # create a subset of the model class
    del model_class
    model_subset = {
        "lm_head": lm_head,
        "tokenizer": tokenizer,
    }


    feature_source = args.features_path
    # if feature_source len is 1 and is not None and a dir path than a file path
    if feature_source is not None and os.path.isdir(feature_source[0]) and len(feature_source) == 1:
        feature_source = feature_source[0]
        feature_files = os.listdir(feature_source)
        feature_files = [f for f in feature_files if f.endswith(".pth")]
        feature_files = [os.path.join(feature_source, f) for f in feature_files]
        concept_names = [args.save_filename +f"_"+ f.split("_")[-1].split(".")[0] for f in feature_files]
        all_args = []

        for i, arg in enumerate(concept_names):
            arg = copy.deepcopy(args)
            arg.save_filename = concept_names[i]
            arg.features_path = [feature_files[i]]
            all_args.append(arg)

        for arg in all_args:
            concept_decompostion(arg, model_subset, logger)

        #args.features_path = os.path.join(feature_source, feature_file)
        #concept_decompostion(args, logger, device)
    else:
        concept_decompostion(args, model_subset, logger)


    

