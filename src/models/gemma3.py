from typing import Any, Callable, Dict, List

import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import os
from .image_text_model import ImageTextModel

__all__ = ["Gemma3nVL"]


class Gemma3nVL(ImageTextModel):

    def set_model(self) -> None:
        self.model_ = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,  # Or float16 depending on your hardware
            low_cpu_mem_usage=True,
            local_files_only=self.local_files_only,
            token=os.getenv("HF_TOKEN", None),
        ).eval()

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

