#!/usr/bin/env python3
"""Vision-Language Model Concept Explainer (Gemma 3n)

- Loads Gemma 3n model & processor
- Hooks a target layer to capture activations
- Compares pooled activations with concept vectors via cosine similarity
- Returns model output + top-N concept matches with grounding

Memory-aware:
- Resizes images to max 512 on the longer side
- Moves captured activations to CPU immediately
- Keeps concept vectors on CPU and computes similarities on CPU
- Supports small batch inference; default batch_size=1
"""

from __future__ import annotations

import os
# Suppress noisy third-party logs early
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide TF INFO/WARNING/ERROR
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import json
import pickle
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Callable
import sys
import argparse
import logging
import warnings

import numpy as np
from PIL import Image
import random
import re
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
from transformers.utils import logging as hf_logging
# Route HF logs through Python logging and keep them quiet by default
hf_logging.set_verbosity_error()
hf_logging.disable_default_handler()
hf_logging.enable_propagation()
# Silence common deprecation/runtime warnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="transformers")

try:
    from transformers import Gemma3nForConditionalGeneration  # type: ignore
    _GEMMA3N_AVAILABLE = True
except Exception:
    _GEMMA3N_AVAILABLE = False


def set_seed_all(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, and HF Transformers for reproducibility.

    Determinism notes:
    - May reduce performance; disables CUDNN benchmarking and TF32.
    - Some CUDA ops remain nondeterministic; small variations can persist across hardware/driver.
    """
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    try:
        from transformers import set_seed as hf_set_seed  # type: ignore
        hf_set_seed(seed)
    except Exception:
        pass


class VLMConceptExplainer:
    REQUIRED_CONCEPT_KEYS = {"concepts"}

    def __init__(
        self,
        model_name: str,
        concept_path: Union[str, Path],
        layer_path: str,
        device: Optional[Union[str, torch.device]] = None,
        trust_remote_code: bool = True,
        hf_token: Optional[str] = None,
        activation_pool: Union[str, Callable[[torch.Tensor], torch.Tensor]] = 'mean',
        default_top_n: int = 5,
        normalize_concepts: bool = True,
        capture_only_last: bool = True,
        exact_match_modules_to_hook: bool = True,
        save_only_generated_tokens: bool = False,
    verbose: bool = False,
    prompt_mode: str = "unsupervised",
    prompt_label: Optional[str] = None,
    prompt_choices: Optional[Union[str, List[str]]] = None,
    ) -> None:
        self.model_name = model_name
        self.layer_path = layer_path
        self.trust_remote_code = trust_remote_code
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
        self.activation_pool = activation_pool
        self.default_top_n = default_top_n
        self.normalize_concepts = normalize_concepts
        self.capture_only_last = capture_only_last
        self.exact_match_modules_to_hook = exact_match_modules_to_hook
        self.verbose = verbose
        self.save_only_generated_tokens = save_only_generated_tokens
        # Prompt configuration
        self.prompt_mode = (prompt_mode or "unsupervised").lower()
        self.prompt_label = prompt_label
        if isinstance(prompt_choices, str):
            # Accept comma-separated string
            self.prompt_choices = [c.strip() for c in prompt_choices.split(',') if c.strip()]
        else:
            self.prompt_choices = prompt_choices

        # Memory behavior
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Load model and processor via repo's loader
        self.model, self.processor = self._load_model(device)
        self.device = getattr(self, '_device_config', None) and self._device_config.primary_device or (
            torch.device(device) if device else next(self.model.parameters()).device
        )

        # Load concept vectors and metadata
        self.concept_data = self._load_concepts(concept_path)
        # Support either 'concepts' or legacy 'activations' key
        vec_key = 'concepts' if 'concepts' in self.concept_data else ('activations' if 'activations' in self.concept_data else None)
        if vec_key is None:
            raise KeyError("Concept file must contain 'concepts' or 'activations'.")
        self.concept_vectors = self._prepare_concept_vectors(self.concept_data[vec_key])  # (K, D), CPU
        self.num_concepts, self.embed_dim = self.concept_vectors.shape
        self.concept_names = self.concept_data.get("concept_names") or self.concept_data.get("names")
        self.text_grounding = self.concept_data.get("text_grounding")
        self.image_grounding_paths = self.concept_data.get("image_grounding_paths")
        self.image_grounding_bboxes = self.concept_data.get("image_grounding_bboxes")

        # Hook setup via project utils
        self._hook_return_fn = None
        self._hook_post_fn = None
        self._utils = None
        self._setup_utils_hooks()

    # ---------- Concept IO ----------
    def _load_concepts(self, path: Union[str, Path]) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Concept file not found: {p}")
        if p.suffix.lower() == '.json':
            return json.loads(p.read_text())
        if p.suffix.lower() in {'.pkl', '.pickle'}:
            with p.open('rb') as f:
                return pickle.load(f)
        if p.suffix.lower() in {'.pt', '.pth'}:
            return torch.load(str(p), map_location='cpu')
        if p.suffix.lower() == '.npz':
            npz = np.load(str(p), allow_pickle=True)
            return {k: (npz[k].tolist() if hasattr(npz[k], 'tolist') else npz[k]) for k in npz.files}
        raise ValueError(f"Unsupported concept file extension: {p.suffix}")

    def _prepare_concept_vectors(self, raw_vectors: Any) -> torch.Tensor:
        if isinstance(raw_vectors, torch.Tensor):
            vecs = raw_vectors.float()
        else:
            vecs = torch.tensor(np.asarray(raw_vectors), dtype=torch.float32)
        if vecs.dim() != 2:
            raise ValueError(f"Concept activation matrix must be 2D, got {tuple(vecs.shape)}")
        if self.normalize_concepts:
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        return vecs.to('cpu')

    # ---------- Model ----------
    def _load_model(self, device_override: Optional[Union[str, torch.device]] = None):
        """Load model via the project's model loader and use its preprocessor.

        Falls back to HF Auto* only if import fails (shouldn't happen in this repo).
        """
        # Ensure we can import from the repo's src/ package
        project_root = Path(__file__).resolve().parents[1]
        src_dir = project_root / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from device_utils import get_device_config  # type: ignore
        device_config = get_device_config(
            str(device_override) if device_override is not None else None
        )
        device = device_config.primary_device
        self._device_config = device_config

        
        from models import get_model_class  # type: ignore

        # Minimal args namespace expected by get_model_class
        args = argparse.Namespace(
            local_files_only=False,
            cache_dir=os.environ.get("HF_HOME"),
        )

        model_class = get_model_class(
            model_name_or_path=self.model_name,
            processor_name=self.model_name,
            device=device,
            logger=None,
            args=args,
            device_config=device_config,
        )
        self._model_class = model_class
        model = model_class.get_model()
        processor = model_class.get_processor()
        return model, processor


    # ---------- Hook ----------
    def _parse_layer_patterns(self) -> List[str]:
        raw = self.layer_path or ""
        parts: List[str] = []
        for sep in (",", "|"):
            if sep in raw:
                parts.extend([p.strip() for p in raw.split(sep)])
                break
        if not parts:
            parts = [raw.strip()]
        # de-dup and drop empties
        uniq: List[str] = []
        for p in parts:
            if p and p not in uniq:
                uniq.append(p)
        return uniq

    def _setup_utils_hooks(self) -> None:
        # Import utils lazily after ensuring src/ in sys.path in _load_model
        try:
            from helpers import utils as _utils  # type: ignore
        except Exception:
            # Fallback: try absolute import
            try:
                import src.helpers.utils as _utils  # type: ignore
            except Exception:
                _utils = None
        self._utils = _utils
        if _utils is None:
            # As a fallback, keep no external hooks (we'll error if activations are needed)
            self._hook_return_fn = None
            self._hook_post_fn = None
            return

        # Clear any previous global state
        try:
            _utils.clear_forward_hooks(self.model)
            _utils.clear_hooks_variables()
        except Exception:
            pass

        modules_to_hook = [self._parse_layer_patterns()]
        hook_names = ["save_hidden_states"]
        # Minimal args namespace controlling exact match
        hook_args = argparse.Namespace(exact_match_modules_to_hook=self.exact_match_modules_to_hook, verbose=self.verbose, save_only_generated_tokens=self.save_only_generated_tokens)
        hook_ret, hook_post = _utils.setup_hooks(
            model=self.model,
            modules_to_hook=modules_to_hook,
            hook_names=hook_names,
            tokenizer=getattr(self._model_class, "get_tokenizer", lambda: None)(),
            logger=None,
            args=hook_args,
        )
        self._hook_return_fn = hook_ret[0] if hook_ret else None
        self._hook_post_fn = hook_post[0] if hook_post else None

    # ---------- Prompts ----------
    def _build_instruction(self, label: Optional[str]) -> str:
        mode = (self.prompt_mode or "unsupervised").lower()
        if mode == "binary":
            lab = self.prompt_label or label
            if lab:
                return (
                    f"Binary classification: Say {lab} if the image contains '{lab}', say 'Something else' if it does not. "
                    f"Respond with only {lab} or 'Something else'."
                )
            # Fallback to unsupervised if no label provided
        if mode == "mcq":
            choices = self.prompt_choices or []
            if choices:
                choice_str = ", ".join(choices)
                return (
                    f"Multiple-choice classification: Choose the single best label from: {choice_str}. "
                    "Respond with exactly one of these options."
                )
            # Fallback to unsupervised if no choices provided
        # Unsupervised/default prompt
        return "Classify and name the main object in each grid, separated by spaces, in a single line."

    def _prepare_inputs_single(self, image: Union[Image.Image, str, Path], label: Optional[str]):
        """Use the repo's model_class.preprocessor to build inputs for a single sample."""
        instruction = self._build_instruction(label)
        # Prefer the repository preprocessor if available
        if getattr(self, "_model_class", None) is not None:
            return self._model_class.preprocessor(
                instruction=instruction,
                image_file=image,
                response="",
                generation_mode=True,
            )
        # Fallback: approximate behavior via processor.apply_chat_template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    # ---------- Utils ----------
    
    @staticmethod
    def _resize_if_large(img: Image.Image, max_side: int = 512) -> Image.Image:
        w, h = img.size
        m = max(w, h)
        if m <= max_side:
            return img
        scale = max_side / float(m)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        return img.resize((nw, nh), Image.Resampling.LANCZOS)

    def _pool_activation(self, act: torch.Tensor) -> torch.Tensor:
        if act is None:
            raise RuntimeError("Activation not captured; check layer_path")
        if callable(self.activation_pool):
            return self.activation_pool(act)
        if self.activation_pool == 'cls':
            # take first token if sequence available
            return act[:, 0] if act.dim() >= 3 else act
        # default 'mean': average across non-batch, non-feature dims; keep last dim as features
        if act.dim() == 1:
            return act
        if act.dim() == 2:
            # (B, D) already pooled
            return act
        # (B, T, D) or (B, H, W, C) -> mean over all dims except last (feature) and batch
        reduce_dims = tuple(range(1, act.dim() - 1))
        if reduce_dims:
            return act.mean(dim=reduce_dims)
        return act

    # ---------- API ----------
    @torch.inference_mode()
    def explain_with_concept(
        self,
        images: List[Union[str, Path, Image.Image]],
        ground_truth_labels: Optional[List[Optional[str]]] = None,
        top_n: Optional[int] = None,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
        batch_size: int = 1,
        save_only_generated_tokens: bool = False,
    ) -> List[Dict[str, Any]]:
        if ground_truth_labels and len(ground_truth_labels) != len(images):
            raise ValueError("ground_truth_labels length must match images length")
        N = top_n if top_n is not None else self.default_top_n
        N = min(N, getattr(self, 'num_concepts', N))
        # Preload + resize PIL if needed, but keep original input (path or PIL) for the model preprocessor
        prepped_inputs: List[Union[Image.Image, str, Path]] = []
        abs_image_paths: List[Optional[str]] = []
        for img_in in images:
            if isinstance(img_in, (str, Path)):
                # Keep path; let model preprocessor load/resize
                prepped_inputs.append(str(img_in))
                try:
                    abs_image_paths.append(os.path.abspath(str(img_in)))
                except Exception:
                    abs_image_paths.append(None)
            elif isinstance(img_in, Image.Image):
                prepped_inputs.append(self._resize_if_large(img_in.convert('RGB')))
                abs_image_paths.append(None)
            else:
                raise TypeError(f"Unsupported image type: {type(img_in)}")

        # Determine labels used for instruction (not ground-truth).
        # Binary mode precedence: explicit prompt_label > provided ground_truth_labels > infer from path > None
        if (self.prompt_mode or "").lower() == "binary":
            if self.prompt_label is not None:
                labels_all = [self.prompt_label] * len(prepped_inputs)
            elif ground_truth_labels:
                labels_all = ground_truth_labels
            else:
                # Try to infer label from image path's parent directory name
                inferred: List[Optional[str]] = []
                for p in abs_image_paths:
                    if p:
                        try:
                            inferred.append(Path(p).parent.name)
                        except Exception:
                            inferred.append(None)
                    else:
                        inferred.append(None)
                labels_all = inferred
        else:
            labels_all = ground_truth_labels if ground_truth_labels else [None] * len(prepped_inputs)

        results: List[Dict[str, Any]] = []

        def chunks(seq, n):
            for i in range(0, len(seq), n):
                yield i, seq[i:i+n]

        for base_idx, img_chunk in chunks(prepped_inputs, batch_size):
            lab_chunk = labels_all[base_idx: base_idx + len(img_chunk)]
            # Reset utils hook buffers
            if self._utils is not None:
                try:
                    self._utils.clear_hooks_variables()
                except Exception:
                    pass
            # Using repo preprocessor which returns device/dtype-aligned inputs.
            # Note: batch_size>1 is not supported; process one by one.
            #assert len(img_chunk) == 1, "Batching not supported; expected single item per chunk"
            inputs = self._prepare_inputs_single(img_chunk[0], lab_chunk[0])
            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

            tokenizer = None
            if getattr(self, "_model_class", None) is not None:
                tokenizer = getattr(self._model_class, "get_tokenizer", lambda: None)()
            if tokenizer is None and hasattr(self.processor, 'tokenizer'):
                tokenizer = self.processor.tokenizer

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                pad_token_id=(tokenizer.eos_token_id if tokenizer is not None else None),
                use_cache=True,
                early_stopping=True,
            )
            # Avoid passing unsupported kwargs to prevent HF warnings; use a cloned generation_config instead
            gen_config = getattr(self.model, 'generation_config', None)
            if gen_config is not None:
                try:
                    gen_config = gen_config.clone()
                except Exception:
                    from copy import deepcopy
                    gen_config = deepcopy(gen_config)
                if temperature > 0:
                    try:
                        gen_config.temperature = float(temperature)
                    except Exception:
                        pass

            if gen_config is not None:
                out_tokens = self.model.generate(**inputs, generation_config=gen_config, **gen_kwargs)
            else:
                out_tokens = self.model.generate(**inputs, **gen_kwargs)

            # Decode full texts and collect token IDs per sample
            input_len = inputs['input_ids'].shape[-1]
            batch_texts: List[str] = []
            gen_token_ids: List[List[int]] = []
            for row in out_tokens:
                new = row[input_len:]
                # Prefer model tokenizer for decoding
                if tokenizer is not None:
                    decoded = tokenizer.decode(new, skip_special_tokens=True)
                else:
                    decoded = self.processor.decode(new, skip_special_tokens=True)
                batch_texts.append(decoded.strip())
                gen_token_ids.append(new.tolist())

            # For debugging show the the decoded generated tokens
            if self.verbose:
                logging.debug("Generated tokens (debug):")
                for text in batch_texts:
                    logging.debug(f" - {text}")
            # Activation sequence across generation steps via utils hook: (B, T, D)
            act_seq = None
            if self._hook_return_fn is not None and self._utils is not None:
                try:
                    # Provide optional kwargs though not needed for base hook
                    hook_out = self._hook_return_fn(
                        tokenizer=tokenizer,
                        image=img_chunk,
                        model_output=None,
                        model_generated_output=out_tokens[0][input_len:].unsqueeze(0),
                    )
                    hidden_dict = hook_out.get("hidden_states", {}) if isinstance(hook_out, dict) else {}
                    if len(hidden_dict) == 0:
                        raise RuntimeError("No hidden_states captured by hooks")
                    # Pick the first hooked module deterministically
                    first_key = sorted(hidden_dict.keys())[0]
                    act_seq = hidden_dict[first_key]  # (B, T, D)
                    # Ensure activations are on CPU for similarity computation
                    #act_seq = act_seq.detach().to('cpu', dtype=torch.float32)
                except Exception as e:
                    raise RuntimeError(f"Failed to retrieve hidden states via hooks: {e}")
            if act_seq is None:
                raise RuntimeError("No activation captured for batch; check layer_path and hooks setup")

            # For each sample, compute per-token top-N concepts
            for j in range(len(act_seq)):
                new_ids = gen_token_ids[j]
                T_new = len(new_ids)
                T_cap = act_seq[j].shape[0]
                # if T_new > T_cap throug an error in hook, else align to shortest
                if T_new < T_cap:
                    raise RuntimeError(f"Captured activation length {T_cap} shorter than generated tokens {T_new}; hook error?")
                t_len = min(T_new, T_cap)
                # Align to last t_len steps of captures (generation time steps)
                acts_j = act_seq[j][-t_len:, :]  # (t_len, D)
                # Use cosine distance per token vs each concept: d = 1 - cos_sim
                  # (t_len, 1, D)
                x = torch.nn.functional.normalize(acts_j, dim=0)
                x = x.unsqueeze(1)
                y = self.concept_vectors.unsqueeze(0)  # (1, K, D)
                sims_tok = torch.nn.functional.cosine_similarity(x, y, dim=2)  # (t_len, K)
                dists_tok = 1.0 - sims_tok

                # decode tokens one-by-one (optional)
                token_texts: List[str] = []
                tok_decoder = tokenizer or getattr(self.processor, 'tokenizer', None)
                for tid in new_ids[-t_len:]:
                    if tok_decoder is not None:
                        token_texts.append(tok_decoder.decode([tid], skip_special_tokens=True))
                    else:
                        token_texts.append(str(tid))

                per_token_concepts: List[Dict[str, Any]] = []
                for t_idx in range(t_len):
                    dists =  dists_tok[t_idx]
                    k = min(N, dists.shape[0])
                    # smallest distances
                    topk = torch.topk(dists, k=k, largest=False)
                    concept_indices = topk.indices.tolist()
                    concept_scores = topk.values.tolist()
                    top_concepts_tok: List[Dict[str, Any]] = []
                    for rank, (ci, dist_val) in enumerate(zip(concept_indices, concept_scores), 1):
                        img_gp = None
                        img_gbbx = None
                        if self.image_grounding_paths and ci < len(self.image_grounding_paths):
                            try:
                                img_gp = str(self.image_grounding_paths[ci])
                                
                            except Exception:
                                img_gp = self.image_grounding_paths[ci]
                        img_gbbx = str(self.image_grounding_bboxes[ci]) if self.image_grounding_bboxes and ci < len(self.image_grounding_bboxes) else None
                        top_concepts_tok.append({
                            'rank': rank,
                            'concept_index': ci,
                            'distance': float(dist_val),
                            'similarity': float(1.0 - dist_val),
                            'concept_name': self.concept_names[ci] if self.concept_names and ci < len(self.concept_names) else None,
                            'text_grounding': self.text_grounding[ci] if self.text_grounding and ci < len(self.text_grounding) else None,
                            'image_grounding_path': img_gp,
                            'image_grounding_bboxes': img_gbbx,
                        })
                    per_token_concepts.append({
                        'token_index': t_idx,
                        'token_id': new_ids[-t_len + t_idx],
                        'token_text': token_texts[t_idx],
                        'top_concepts': top_concepts_tok,
                    })

                # Also compute pooled top concepts over the whole sequence (optional aggregate)
                # Filter tokens before pooling using regex cleanup logic (articles/pronouns and specials)
                def _keep_token_text(s: str) -> bool:
                    if not s or not any(ch.isalnum() for ch in s):
                        return False
                    # Default removal lists
                    remove_words: List[str] = []
                    articles = r'\b(?:a|an|the)\b'
                    pronouns = r'\b(?:i|object|you|he|she|it|we|they|me|him|her|us|them|my|mine|your|yours|his|hers|its|our|ours|that|their|theirs|myself|yourself|himself|herself|itself|ourselves|yourselves|themselves|photo|image|im_end|question)\b'
                    combined_pattern = re.compile(f"{articles}|{pronouns}", flags=re.IGNORECASE)
                    txt = combined_pattern.sub('', s)
                    for word in remove_words:
                        txt = re.sub(rf'\b{re.escape(word)}\b', '', txt, flags=re.IGNORECASE)
                    txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt)
                    txt = re.sub(r'\s+', ' ', txt).strip()
                    return txt != ''

                mask_keep = [
                    _keep_token_text(token_texts[t_idx]) if t_idx < len(token_texts) else False
                    for t_idx in range(t_len)
                ]
                if any(mask_keep):
                    acts_keep = acts_j[torch.tensor(mask_keep, dtype=torch.bool)]  # (m, D)
                else:
                    acts_keep = acts_j  # fallback: keep all if none qualify
                pooled = acts_keep.mean(dim=0, keepdim=True)  # (1, D)
                # Cosine similarity over concepts (higher is better)
                sims_all = F.cosine_similarity(pooled, self.concept_vectors, dim=-1)
                arr = sims_all.detach().cpu().numpy()
                k_all = min(N, arr.shape[0])
                idx_all = np.argsort(arr)[-k_all:][::-1].tolist()
                sim_vals = [float(arr[i]) for i in idx_all]
                top_concepts_all: List[Dict[str, Any]] = []
                for rank, (ci, sim_val) in enumerate(zip(idx_all, sim_vals), 1):
                    img_gp = None
                    img_gbbx = None
                    if self.image_grounding_paths and ci < len(self.image_grounding_paths):
                        try:
                            img_gp = str(self.image_grounding_paths[ci])
                        except Exception:
                            img_gp = self.image_grounding_paths[ci]
                    if self.image_grounding_bboxes and ci < len(self.image_grounding_bboxes):
                        try:
                            img_gbbx = str(self.image_grounding_bboxes[ci])
                        except Exception:
                            img_gbbx = self.image_grounding_bboxes[ci]
                    top_concepts_all.append({
                        'rank': rank,
                        'concept_index': ci,
                        'similarity': float(sim_val),
                        'concept_name': self.concept_names[ci] if self.concept_names and ci < len(self.concept_names) else None,
                        'text_grounding': self.text_grounding[ci] if self.text_grounding and ci < len(self.text_grounding) else None,
                        'image_grounding_path': img_gp,
                        'image_grounding_bboxes': img_gbbx,
                    })

                results.append({
                    'image_path': abs_image_paths[base_idx + j],
                    'ground_truth': lab_chunk[j],
                    'model_output': batch_texts[j],
                    'generated_token_ids': new_ids,
                    'per_token_concepts': per_token_concepts,
                    'top_concepts_over_sequence': top_concepts_all,
                    #'layer_activation_shape': tuple(len(act_seq.shape)),
                    'hook_layer': self.layer_path,
                    'model_name': self.model_name,
                })

        return results

    def close(self):
        # Clear project-level hooks and buffers
        if getattr(self, "_utils", None) is not None:
            try:
                self._utils.clear_forward_hooks(self.model)
                self._utils.clear_hooks_variables()
            except Exception:
                pass


__all__ = ["VLMConceptExplainer"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Quick test for VLMConceptExplainer")
    ap.add_argument('--model_name', default=os.environ.get('VLM_MODEL', 'Qwen/Qwen2.5-VL-3B-Instruct'))
    ap.add_argument('--concept_path', required=True)
    ap.add_argument('--layer_path', required=True)
    ap.add_argument('--image', action='append')  # removed required=True to allow --image_root only
    ap.add_argument('--image_root', default=None, help='Root dir to recursively collect images')
    ap.add_argument('--label', action='append')
    ap.add_argument('--top_n', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=int(os.environ.get('BATCH_SIZE', '10')))
    ap.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    ap.add_argument('--deterministic', action='store_true', help='Enable deterministic kernels (may slow down)')
    ap.add_argument('--verbose', action='store_true', help='Verbose debug logging')
    # Prompt configuration
    ap.add_argument('--prompt_mode', default='unsupervised', choices=['unsupervised', 'binary', 'mcq'], help='Prompt mode to use')
    ap.add_argument('--prompt_label', default=None, help='Label to check for in binary mode')
    ap.add_argument('--choice', action='append', dest='choice_list', help='Add a choice label (can be repeated)')
    ap.add_argument('--choices', default=None, dest='choices_csv', help='Comma-separated choices (alternative to repeated --choice)')
    # New: JSON output and data root controls
    ap.add_argument('--out_json', default=os.path.join(os.environ.get('OUTPUT_DIR', '.'), 'vlm_explanations.json'), help='Path to save JSON results')
    ap.add_argument('--data_root', default=None, help='Root path of dataset. If omitted, inferred from image paths')
    ap.add_argument('--save_only_generated_tokens', default=True, action='store_true', help='Save only the generated tokens in the output')
    args = ap.parse_args()

    # Configure logging for terminal output
    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format='%(levelname)s: %(message)s'
    )

    # Build image list from --image_root if provided
    if getattr(args, 'image_root', None):
        root = os.path.abspath(args.image_root)
        if not os.path.isdir(root):
            raise ValueError(f"--image_root must be a directory: {root}")
        valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
        collected = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in valid_exts:
                    collected.append(os.path.join(dirpath, fn))
        collected.sort()
        if args.image is None:
            args.image = []
        args.image.extend(collected)
        logging.info(f"Discovered {len(collected)} images under {root}")

    if not args.image:
        raise ValueError("No images provided. Use --image one or more times, or provide --image_root.")

    # Set seeds early for reproducibility
    set_seed_all(args.seed, deterministic=args.deterministic)
    # add a  exception is batch size bigger than 1 the code does not work, the  error due the dataloader operation
    if args.batch_size > 1:
        raise ValueError("Batch size greater than 1 is not supported.")
    # Prepare choices list if provided
    prompt_choices = None
    if getattr(args, 'choice_list', None):
        prompt_choices = args.choice_list
    elif getattr(args, 'choices_csv', None):
        prompt_choices = args.choices_csv

    explainer = VLMConceptExplainer(
        args.model_name,
        args.concept_path,
        args.layer_path,
        verbose=args.verbose,
        prompt_mode=args.prompt_mode,
        prompt_label=args.prompt_label,
        prompt_choices=prompt_choices,
        save_only_generated_tokens=args.save_only_generated_tokens,
    )
    res = explainer.explain_with_concept(args.image, ground_truth_labels=args.label, top_n=args.top_n, batch_size=args.batch_size)
    for r in res:
        logging.info(f"Image {r.get('image_path')} -> gt={r['ground_truth']}\nModel: {r['model_output']}")
        top_list = r.get('top_concepts_over_sequence') or r.get('top_concepts') or []
        logging.info("Top concepts (aggregate):")
        for c in top_list:
            logging.info(f"  #{c['rank']} idx={c['concept_index']} sim={c['similarity']:.4f} text={c['text_grounding']}")

        # Optional: brief per-token top-1 preview + top-N concept names
        pt = r.get('per_token_concepts') or []
        if pt:
            logging.info("Per-token top-1 (preview) + top-N names:")
            for tok in pt:
                if tok.get('top_concepts'):
                    c = tok['top_concepts'][0]
                    # Build top-N concept names list
                    n_show = min(args.top_n, len(tok['top_concepts'])) if hasattr(args, 'top_n') else len(tok['top_concepts'])
                    names = []
                    for cc in tok['top_concepts'][:n_show]:
                        nm = cc.get('concept_name') or cc.get('text_grounding') or f"idx{cc.get('concept_index')}"
                        names.append(str(nm))
                    names_str = ", ".join(names)
                    logging.info(
                        f"  t={tok['token_index']} '{tok.get('token_text','')}' -> "
                        f"idx={c['concept_index']} text={c.get('text_grounding')} sim={c['similarity']:.4f} | top{n_show}: {names_str}"
                    )
        logging.info('-'*60)

    # Save consolidated JSON with metadata
    try:
        img_abs_list = [os.path.abspath(str(p)) for p in (args.image or [])]
        if args.data_root is not None:
            data_root = os.path.abspath(args.data_root)
        else:
            data_root = os.path.commonpath(img_abs_list) if img_abs_list else None
    except Exception:
        data_root = None

    out_payload: Dict[str, Any] = {
        'model_card': args.model_name,
        'layer_path': args.layer_path,
        'data_root': data_root,
        'results': res,
    }
    try:
        out_path = args.out_json
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved JSON to {out_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON: {e}")

    explainer.close()
