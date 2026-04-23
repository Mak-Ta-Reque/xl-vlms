import logging
import os
import torch.nn as nn
__all__ = ["setup_logger", "log_args"]


def setup_logger(log_file=None, level=logging.INFO):
    # Create a custom logger
    logger = logging.getLogger("train_test_logger")
    logger.setLevel(level)
    logger.propagate = False

    # Prevent duplicate logs when setup_logger is called multiple times.
    if logger.handlers:
        logger.handlers.clear()

    # Create handlers
    console_handler = logging.StreamHandler()  # Log to console
    console_handler.setLevel(level)

    # Optional: Log to file as well
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

    # Create formatters and add them to handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    if log_file:
        file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)

    if log_file:
        logger.addHandler(file_handler)

    return logger


def log_args(args, logger):
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")


logger = logging.getLogger(__name__)

def log_num_transformer_layers(model: nn.Module, model_name: str = "model") -> int:
    """
    Log & return the number of Transformer layers for HF LLMs like Gemma / Qwen.

    Priority:
      1) Use model.config.num_hidden_layers  (works for Gemma, Qwen2, Qwen2.5, etc.)
      2) Fallback: try common HF attributes: model.layers / model.model.layers / model.language_model.layers
      3) Last resort: count submodules whose class name looks like a transformer block.
    """
    num_layers = None

    # 1) Most HF decoder-only LMs (Gemma, Qwen) -> use config.num_hidden_layers
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "num_hidden_layers"):
        num_layers = int(cfg.num_hidden_layers)

    # 2) Try common HF internals if config is weird/missing
    if num_layers is None:
        for top_name in ["model", "language_model", "transformer"]:
            top = getattr(model, top_name, None)
            if top is None:
                continue
            for layers_name in ["layers", "h", "blocks"]:
                layers = getattr(top, layers_name, None)
                if layers is not None:
                    num_layers = len(layers)
                    break
            if num_layers is not None:
                break

    # 3) Very rough fallback: count "block-like" modules
    if num_layers is None:
        num_layers = 0
        for m in model.modules():
            name = m.__class__.__name__.lower()
            if ("decoderlayer" in name) or name.endswith("block"):
                num_layers += 1

    msg = f"[TransformerLayers] {model_name}: total transformer layers = {num_layers}"
    print(msg)
    logger.info(msg)

    return num_layers
