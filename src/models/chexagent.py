from typing import Any, Callable, Dict

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from models.image_text_model import ImageTextModel
from helpers.utils import load_image_as_rgb
import logging

__all__ = ["CheXagent"]

class CheXagent(ImageTextModel):

    def set_model(
        self,
    ) -> None:

        self.model_ = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir ='/netscratch/kadir/xl-vlms/cache/hub',
            local_files_only=self.local_files_only,
        )


    def get_language_model(
        self,
    ) -> Callable:

        return self.model_.model

    def get_lm_head(
        self,
    ) -> Callable:
        layer = self.model_
        return self.model_.language_model.lm_head

    def set_processor(
        self,
    ) -> None:

        self.processor_ = AutoProcessor.from_pretrained(
            self.processor_name,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.tokenizer_ = self.processor_.tokenizer

    def set_preprocessor(
        self,
    ) -> None:

        self.preprocessor_ = self.preprocess_input

    def get_conversation_template(
        self,
        instruction: str = "What are these?",
        response: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:

        conversation = instruction
        if response:
            conversation += f" Answer: {response}"
        return conversation

    def preprocess_input(
        self,
        instruction: str = "What are these?",
        image_file: str = None,
        response: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:

        text = self.get_conversation_template(
            instruction=instruction,
            response=response,
            image_file=image_file,
        )

        image = load_image_as_rgb(image_file, out_type="np")
        #image = np.array(image)
        text=f" USER: <s> {text} ASSISTANT: <s>"
        inputs = self.processor_(
            text=text,
            images=[image],
            #padding=True,
            return_tensors="pt",
        ).to(device=self.model_.device, dtype=self.model_.dtype)

        return inputs

    def preprocessor(
        self,
        instruction: str = "What are these?",
        image_file: str = "",
        response: str = "",
        generation_mode: bool = False,
        **kwargs: Any,
    ):
        preprocessor = self.get_preprocessor()
        inputs = preprocessor(
            instruction=instruction,
            image_file=image_file,
            response=response,
            generation_mode=generation_mode,
        )
        return inputs

    def generate(
        self,
        max_new_tokens: int = 200,
        do_sample: bool = False,
        **inputs: Dict[str, Any],
    ):
        inputs = {k: v.unsqueeze(0).to(self.model_.device) for k, v in inputs.items()}
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(
            device_type=device_type, enabled=True, dtype=self.model_.dtype
        ):
            output = self.model_.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    stop_strings="<|endoftext|>",
                    do_sample=do_sample,
                ),
                tokenizer=self.tokenizer_,
            )
        return output



