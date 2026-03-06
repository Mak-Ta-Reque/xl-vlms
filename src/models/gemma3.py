from typing import Any, Callable, Dict, List

import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import os
from .image_text_model import ImageTextModel

__all__ = ["Gemma3nVL"]


class Gemma3nVL(ImageTextModel):

    def set_model(self) -> None:
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=self.local_files_only,
            token=os.getenv("HF_TOKEN", None),
        )
        _use_device_map = self.device_config is not None and self.device_config.is_multi_gpu
        if _use_device_map:
            if self.device_config.device_map is not None:
                load_kwargs["device_map"] = self.device_config.device_map
            if self.device_config.max_memory is not None:
                load_kwargs["max_memory"] = self.device_config.max_memory
        self.model_ = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_name_or_path, **load_kwargs
        ).eval()
        if not _use_device_map and self.device_config is not None:
            self.model_ = self.model_.to(self.device_config.primary_device)

    def get_language_model(self) -> Callable:
        return self.model_.model if hasattr(self.model_, "model") else self.model_

    def get_lm_head(self) -> Callable:
        return getattr(self.model_, "lm_head", None)

    def set_processor(self) -> None:
        self.processor_ = AutoProcessor.from_pretrained(
            self.processor_name,
            local_files_only=self.local_files_only,
            token=os.getenv("HF_TOKEN", None),
        )
        self.tokenizer_ = self.processor_.tokenizer

    def set_preprocessor(self) -> None:
        self.preprocessor_ = self.preprocess_input

    def get_conversation_round(
        self,
        instruction: str = "Describe this image.",
        response: str = "",
        image_file: str = "",
    ) -> List[Dict[str, Any]]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_file},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        if response:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                }
            )
        return conversation

    def get_conversation_template(
        self,
        instruction: str = "Describe this image.",
        response: str = "",
        image_file: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self.get_conversation_round(
            instruction=instruction,
            response=response,
            image_file=image_file,
        )

    def preprocess_text(
        self,
        conversation,
        generation_mode: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self.processor_.apply_chat_template(
            conversation,
            add_generation_prompt=generation_mode,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    def _preprocess_input(
        self,
        instruction: str = "Describe this image.",
        image_file: Any = None,
        response: str = "",
        generation_mode: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        conversation = self.get_conversation_template(
            instruction=instruction,
            response=response,
            image_file=image_file,
        )

        inputs = self.processor_.apply_chat_template(
            conversation,
            add_generation_prompt=generation_mode,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        return inputs
    
    def preprocess_input(
        self,
        instruction: str = "Describe this image.",
        image_file: Any = None,
        response: str = "",
        generation_mode: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        if image_file:
            # Vision-language input
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_file},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            if response:
                conversation.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response}],
                    }
                )
        else:
            # Text-only input
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": instruction}],
                }
            ]
            if response:
                conversation.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response}],
                    }
                )

        inputs = self.processor_.apply_chat_template(
            conversation,
            add_generation_prompt=generation_mode,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        return inputs

