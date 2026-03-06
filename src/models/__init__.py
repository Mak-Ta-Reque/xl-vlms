import argparse
from typing import Callable, Optional, Tuple, Union

import torch

__all__ = ["get_model_class"]


SUPPORTED_MODELS = [
    "llava-1.5",
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "HuggingFaceM4/idefics2-8b",
    "allenai/Molmo-7B-D-0924",
    "StanfordAIMI/CheXagent-8b",
]


def get_model_class(
    model_name_or_path: str = "llava-hf/llava-1.5-7b-hf",
    processor_name: str = "llava-hf/llava-1.5-7b-hf",
    device: Union[torch.device, str, None] = None,
    logger: Callable = None,
    args: argparse.Namespace = None,
    device_config=None,
) -> Tuple[Callable]:
    """Load model class.

    Parameters
    ----------
    device_config : DeviceConfig | None
        Preferred way to specify placement.  When supplied, *device* is ignored
        and the model is loaded with ``device_map`` / ``max_memory`` from the
        config.  When ``None``, legacy ``device`` + ``.to()`` behaviour is kept.
    """
    # Lazy-import to avoid circular deps when device_utils lives in src/
    if device_config is None and device is not None:
        try:
            from device_utils import parse_device_config  # type: ignore
            device_config = parse_device_config(str(device))
        except Exception:
            pass  # fall back to legacy .to(device)

    _dc_kwarg = dict(device_config=device_config) if device_config is not None else {}

    if "llava-1.5" in model_name_or_path:
        from models.llava import LLaVA

        model_class = LLaVA(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
            cache_dir=args.cache_dir,
            **_dc_kwarg,
        )
    elif "Qwen2.5-VL" in model_name_or_path:
        from models.qwen_2_5 import Qwen2_5VL

        model_class = Qwen2_5VL(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
            **_dc_kwarg,
        )
    elif "Qwen2-VL" in model_name_or_path:
        from models.qwen_vl import QwenVL

        model_class = QwenVL(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
            **_dc_kwarg,
        )
    elif "gemma-3" in model_name_or_path:
        from models.gemma3 import Gemma3nVL

        model_class = Gemma3nVL(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
            **_dc_kwarg,
        )
    elif "idefics" in model_name_or_path:
        from models.idefics2 import IDEFICS

        model_class = IDEFICS(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
            **_dc_kwarg,
        )
    elif "Molmo" in model_name_or_path:
        from models.molmo import Molmo

        model_class = Molmo(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
            **_dc_kwarg,
        )
    elif "CheXagent" in model_name_or_path:
        from models.chexagent import CheXagent
        model_class = CheXagent(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
            **_dc_kwarg,
        )

    else:
        raise NotImplementedError(
            f"Got {model_name_or_path}, but only these models are supported {SUPPORTED_MODELS}"
        )

    if logger is not None:
        logger.info(f"Successfully loaded {model_name_or_path}, device config: {device_config or device}")

    # Place model on device: single-GPU device_config or legacy device string
    if getattr(model_class.model_, 'hf_device_map', None) is None:
        if device_config is not None and not device_config.is_multi_gpu:
            model_class.model_.to(device_config.primary_device)
        elif device_config is None and device is not None:
            model_class.model_.to(device)

    return model_class
