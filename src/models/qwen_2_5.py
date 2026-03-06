from typing import Any, Callable, Dict, List

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from qwen_vl_utils import process_vision_info
from .image_text_model import ImageTextModel

__all__ = ["Qwen2_5VL"]


class Qwen2_5VL(ImageTextModel):
    def set_model(self) -> None:
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
        self.model_ = AutoModelForVision2Seq.from_pretrained(
            self.model_name_or_path, **load_kwargs
        ).eval()
        if not _use_device_map and self.device_config is not None:
            self.model_ = self.model_.to(self.device_config.primary_device)

    def get_language_model(self) -> Callable:
        return self.model_.model

    def get_lm_head(self) -> Callable:
        return self.model_.lm_head

    def set_processor(self) -> None:
        self.processor_ = AutoProcessor.from_pretrained(
            self.processor_name,
            local_files_only=self.local_files_only,
        )
        self.tokenizer_ = self.processor_.tokenizer

    def set_preprocessor(self) -> None:
        self.preprocessor_ = self.preprocess_input

    def get_conversation_round(
        self,
        instruction: str = "What are these?",
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
            },
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
        instruction: str = "What are these?",
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
        conversation: List[Dict[str, Any]],
        generation_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        return self.processor_.apply_chat_template(
            conversation,
            add_generation_prompt=generation_mode,
            tokenize=False,
        )

    def preprocess_images(
        self,
        conversation: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List:
        image_inputs, video_inputs = process_vision_info(conversation)
        return image_inputs, video_inputs

    def preprocess_input(
        self,
        instruction: str = "What are these?",
        image_file: str = None,
        response: str = "",
        generation_mode: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        conversation = self.get_conversation_template(
            instruction=instruction,
            response=response,
            image_file=image_file,
        )

        text = self.preprocess_text(conversation, generation_mode=generation_mode)

        if image_file:
            image_inputs, video_inputs = self.preprocess_images(conversation)
            return self.processor_(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            return self.processor_(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
