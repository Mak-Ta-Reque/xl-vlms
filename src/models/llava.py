from typing import Any, Callable, Dict, List

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .image_text_model import ImageTextModel

__all__ = ["LLaVA"]


class LLaVA(ImageTextModel):

    def set_model(
        self,
    ) -> None:
        load_kwargs = dict(
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=self.local_files_only,
        )
        _use_device_map = self.device_config is not None and self.device_config.is_multi_gpu
        if _use_device_map:
            if self.device_config.device_map is not None:
                load_kwargs["device_map"] = self.device_config.device_map
            if self.device_config.max_memory is not None:
                load_kwargs["max_memory"] = self.device_config.max_memory
        self.model_ = LlavaForConditionalGeneration.from_pretrained(
            self.model_name_or_path, **load_kwargs
        )
        if not _use_device_map and self.device_config is not None:
            self.model_ = self.model_.to(self.device_config.primary_device)

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

        self.processor_ = AutoProcessor.from_pretrained(
            self.processor_name, local_files_only=self.local_files_only
        )
        self.tokenizer_ = self.processor_.tokenizer

    def set_preprocessor(
        self,
    ) -> None:

        self.preprocessor_ = self.preprocess_input

    def get_conversation_round(
        self, instruction: str = "What are these?", response: str = ""
    ) -> List[Dict[str, Any]]:

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image"},
                ],
            },
        ]
        if response:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": response},
                    ],
                },
            )

        return conversation

    def preprocess_text(
        self,
        instruction: str = "What are these?",
        response: str = "",
        generation_mode: bool = False,
        **kwargs: Any,
    ) -> str:

        conversation = self.get_conversation_round(
            instruction=instruction, response=response
        )
        prompt = self.processor_.apply_chat_template(
            conversation, add_generation_prompt=generation_mode
        )

        return prompt
