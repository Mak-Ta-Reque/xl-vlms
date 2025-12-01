from typing import Any, Callable, Dict, Optional

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

from .image_text_model import ImageTextModel

__all__ = ["LLaVA"]


class LLaVA(ImageTextModel):

    def __init__(
        self,
        model_name_or_path: str = "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        processor_name: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> None:

        args = kwargs.pop("args", None)
        if args is None:
            args_dict: Dict[str, Any] = {}
        elif isinstance(args, dict):
            args_dict = args
        else:
            try:
                args_dict = vars(args)
            except TypeError:
                args_dict = {}

        def pick(*keys: str, default: Any = None):
            for key in keys:
                if key in kwargs and kwargs[key] is not None:
                    return kwargs[key]
                if key in args_dict and args_dict[key] is not None:
                    return args_dict[key]
            return default

        self.cache_dir = pick("cache_dir")
        self.token = pick("token", "hf_token", "use_auth_token")
        self.trust_remote_code = pick("trust_remote_code")
        self.device_map = pick("device_map", default="auto")
        self.torch_dtype = pick("torch_dtype", default=torch.bfloat16)
        resolved_local_files_only = pick("local_files_only", default=local_files_only)

        super().__init__(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name or model_name_or_path,
            local_files_only=resolved_local_files_only,
        )

    def set_model(
        self,
    ) -> None:
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "local_files_only": self.local_files_only,
        }
        if self.cache_dir is not None:
            load_kwargs["cache_dir"] = self.cache_dir
        if self.token is not None:
            load_kwargs["token"] = self.token
        if self.trust_remote_code is not None:
            load_kwargs["trust_remote_code"] = self.trust_remote_code

        self.model_ = LlavaForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            **load_kwargs,
        )

    def get_language_model(
        self,
    ) -> Callable:

        return self.model_.language_model

    def get_lm_head(
        self,
    ) -> Callable:

        return self.model_.language_model.lm_head

    def set_processor(
        self,
    ) -> None:

        proc_kwargs: Dict[str, Any] = {"local_files_only": self.local_files_only}
        if self.cache_dir is not None:
            proc_kwargs["cache_dir"] = self.cache_dir
        if self.token is not None:
            proc_kwargs["token"] = self.token
        if self.trust_remote_code is not None:
            proc_kwargs["trust_remote_code"] = self.trust_remote_code

        self.processor_ = AutoProcessor.from_pretrained(
            self.processor_name,
            **proc_kwargs,
        )
        self.tokenizer_ = self.processor_.tokenizer

    def set_preprocessor(
        self,
    ) -> None:

        self.preprocessor_ = self.preprocess_input

    def preprocess_text(
        self,
        instruction: str = "What are these?",
        response: str = "",
        generation_mode: bool = False,
        **kwargs: Any,
    ) -> str:

        text = instruction or "Describe this image."
        text = text.strip()
        prompt = f"[INST] <image>\n{text} [/INST]"
        return prompt
