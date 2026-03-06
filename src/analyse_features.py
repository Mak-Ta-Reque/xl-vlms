import copy
import gc
import os

import torch

from analysis import analyse_features
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from models import get_model_class


def cleanup_gpu_memory():
    """Clean up GPU memory by running garbage collection and clearing CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def concept_decompostion(args, model_subset, logger, device):
   
    analyse_features(
        analysis_name=args.analysis_name,
        logger=logger,
        model_class=model_subset,
        device=device,
        args=args,
    )


def main():
    """Main entry point for analyse_features script."""
    args = get_arguments()

    logger = setup_logger(log_file=os.path.join(args.save_dir, f"logs.log"))
    log_args(args, logger)

    from device_utils import get_device_config  # type: ignore
    device_config = get_device_config(args.device)
    device = device_config.primary_device
    logger.info(f"Device config: {device_config.raw} -> primary={device}, gpu_ids={device_config.gpu_ids}")

    model_class = get_model_class(
        args.model_name_or_path,
        args.processor_name,
        device=device,
        logger=logger,
        args=args,
        device_config=device_config,
    )
    
    # Extract only the components we need (lm_head and tokenizer)
    # Clone the lm_head weights to detach from the original model graph
    lm_head_weights = model_class.get_lm_head().weight.data.clone().float()
    lm_head_bias = None
    if model_class.get_lm_head().bias is not None:
        lm_head_bias = model_class.get_lm_head().bias.data.clone().float()
    
    # Create a standalone lm_head module
    lm_head = torch.nn.Linear(
        lm_head_weights.shape[1], 
        lm_head_weights.shape[0], 
        bias=(lm_head_bias is not None),
        device=device
    )
    lm_head.weight.data = lm_head_weights.to(device)
    if lm_head_bias is not None:
        lm_head.bias.data = lm_head_bias.to(device)
    
    tokenizer = model_class.get_tokenizer()
    
    # Delete the full model and clean GPU memory
    del model_class
    cleanup_gpu_memory()
    
    # Create a subset with only the required components
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
            # Skip empty feature files (e.g. abstract concepts with 0 detections)
            try:
                check = torch.load(arg.features_path[0], map_location="cpu")
                if not check or len(check) == 0:
                    logger.warning(
                        f"Skipping empty feature file: {arg.features_path[0]} "
                        f"(concept '{arg.save_filename}' had no detections)"
                    )
                    continue
                del check
            except Exception as e:
                logger.warning(f"Skipping unreadable feature file {arg.features_path[0]}: {e}")
                continue
            concept_decompostion(arg, model_subset, logger, device)

        #args.features_path = os.path.join(feature_source, feature_file)
        #concept_decompostion(args, logger, device)
    else:
        concept_decompostion(args, model_subset, logger, device)

    # Final cleanup before exiting
    del model_subset
    cleanup_gpu_memory()
    logger.info("GPU memory cleaned up before exit.")


if __name__ == "__main__":
    main()