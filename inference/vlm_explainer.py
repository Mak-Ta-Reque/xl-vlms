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
import json
import pickle
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Callable

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from transformers import Gemma3nForConditionalGeneration  # type: ignore
    _GEMMA3N_AVAILABLE = True
except Exception:
    _GEMMA3N_AVAILABLE = False


class VLMConceptExplainer:
    REQUIRED_CONCEPT_KEYS = {"activations"}

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
    ) -> None:
        self.model_name = model_name
        self.layer_path = layer_path
        self.trust_remote_code = trust_remote_code
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
        self.activation_pool = activation_pool
        self.default_top_n = default_top_n
        self.normalize_concepts = normalize_concepts
        self.capture_only_last = capture_only_last

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        self.model, self.processor = self._load_model()
        self.device = torch.device(device) if device else next(self.model.parameters()).device

        self.concept_data = self._load_concepts(concept_path)
        self.concept_vectors = self._prepare_concept_vectors(self.concept_data["concepts"])  # (K, D), CPU
        self.num_concepts, self.embed_dim = self.concept_vectors.shape
        self.concept_names = self.concept_data.get("text_grounding")
        self.text_grounding = self.concept_data.get("text_grounding")
        self.image_grounding_paths = self.concept_data.get("image_grounding_paths")

        # Activation capture buffers
        self._captured = None
        self._captures = []
        self._hook_handle = self._register_hook()

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
    def _load_model(self):
        cache_dir = os.environ.get("HF_HOME", "/mnt/abka03/huggingface/hub")
        kwargs = dict(cache_dir=cache_dir, trust_remote_code=self.trust_remote_code, torch_dtype=torch.bfloat16, device_map="auto")
        if self.hf_token:
            kwargs['token'] = self.hf_token
        if _GEMMA3N_AVAILABLE and 'gemma-3n' in self.model_name.lower():
            model = Gemma3nForConditionalGeneration.from_pretrained(self.model_name, **kwargs).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs).eval()
        processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=cache_dir, trust_remote_code=self.trust_remote_code, token=self.hf_token)
        return model, processor

    # ---------- Hook ----------
    def _resolve_layer(self) -> torch.nn.Module:
        parts = self.layer_path.split('.')
        module: Any = self.model
        for p in parts:
            if p.isdigit():
                module = module[int(p)]  # type: ignore[index]
            else:
                if not hasattr(module, p):
                    raise AttributeError(f"Layer segment '{p}' not found while resolving '{self.layer_path}'")
                module = getattr(module, p)
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"Resolved object is not nn.Module: {type(module)}")
        return module

    def _register_hook(self):
        target = self._resolve_layer()
        def hook(_m, _inp, out):
            tensor = None
            if isinstance(out, (tuple, list)):
                for o in out:
                    if isinstance(o, torch.Tensor):
                        tensor = o
                        break
            elif isinstance(out, torch.Tensor):
                tensor = out
            if isinstance(tensor, torch.Tensor):
                t = tensor.detach().to('cpu', dtype=torch.float32)
                # Normalize to (B, T, D)
                if t.dim() == 2:
                    t = t.unsqueeze(1)  # (B,1,D)
                # If using KV cache, some layers can emit full sequence. Keep only the last step if requested.
                if self.capture_only_last and t.dim() >= 3 and t.size(1) > 1:
                    t = t[:, -1:, ...]  # keep last token time-step only
                # Keep last slice reference for backward compatibility
                self._captured = t
                self._captures.append(t)
            else:
                self._captured = None
        return target.register_forward_hook(hook)

    # ---------- Prompts ----------
    def _build_messages(self, image: Image.Image, label: Optional[str]) -> List[Dict[str, Any]]:
        if label:
            user_text = f"{label} or something else. Answer with only this two options."
        else:
            user_text = "Classify the image in one or two word"
        system_msg = {"role": "system", "content": [{"type": "text", "text": "You are a helpful vision assistant."}]}
        user_msg = {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]}
        return [system_msg, user_msg]

    def _prepare_inputs_batch(self, images: List[Image.Image], labels: List[Optional[str]]):
        messages_batch = [self._build_messages(img, lab) for img, lab in zip(images, labels)]
        return self.processor.apply_chat_template(
            messages_batch,
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
        max_new_tokens: int = 10,
        temperature: float = 0.0,
        batch_size: int = 1,
    ) -> List[Dict[str, Any]]:
        if ground_truth_labels and len(ground_truth_labels) != len(images):
            raise ValueError("ground_truth_labels length must match images length")
        N = top_n if top_n is not None else self.default_top_n

        # Preload + resize
        loaded_images: List[Image.Image] = []
        for img_in in images:
            if isinstance(img_in, (str, Path)):
                img = Image.open(img_in).convert('RGB')
            elif isinstance(img_in, Image.Image):
                img = img_in.convert('RGB')
            else:
                raise TypeError(f"Unsupported image type: {type(img_in)}")
            loaded_images.append(self._resize_if_large(img))

        labels_all = ground_truth_labels if ground_truth_labels else [None] * len(loaded_images)

        results: List[Dict[str, Any]] = []

        def chunks(seq, n):
            for i in range(0, len(seq), n):
                yield i, seq[i:i+n]

        for base_idx, img_chunk in chunks(loaded_images, batch_size):
            lab_chunk = labels_all[base_idx: base_idx + len(img_chunk)]
            self._captured = None
            self._captures = []
            inputs = self._prepare_inputs_batch(img_chunk, lab_chunk)
            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None,
                use_cache=True,
            )
            if temperature > 0:
                gen_kwargs['temperature'] = temperature
            out_tokens = self.model.generate(**inputs, **gen_kwargs)

            # Decode full texts and collect token IDs per sample
            input_len = inputs['input_ids'].shape[-1]
            batch_texts: List[str] = []
            gen_token_ids: List[List[int]] = []
            for row in out_tokens:
                new = row[input_len:]
                batch_texts.append(self.processor.decode(new, skip_special_tokens=True).strip())
                gen_token_ids.append(new.tolist())

            # Activation sequence across generation steps: (B, T_captures, D)
            act_seq = torch.cat(self._captures, dim=1) if self._captures else None
            if act_seq is None:
                raise RuntimeError("No activation captured for batch; check layer_path")

            # For each sample, compute per-token top-N concepts
            for j in range(act_seq.shape[0]):
                new_ids = gen_token_ids[j]
                T_new = len(new_ids)
                T_cap = act_seq.shape[1]
                t_len = min(T_new, T_cap)
                # Align to last t_len steps of captures (generation time steps)
                acts_j = act_seq[j, -t_len:, :]  # (t_len, D)
                # Use cosine distance per token vs each concept: d = 1 - cos_sim
                x = acts_j.unsqueeze(1)  # (t_len, 1, D)
                y = self.concept_vectors.unsqueeze(0)  # (1, K, D)
                sims_tok = torch.nn.functional.cosine_similarity(x, y, dim=2)  # (t_len, K)
                dists_tok = 1.0 - sims_tok

                # decode tokens one-by-one (optional)
                token_texts: List[str] = []
                tok_decoder = getattr(self.processor, 'tokenizer', None)
                for tid in new_ids[-t_len:]:
                    if tok_decoder is not None:
                        token_texts.append(tok_decoder.decode([tid], skip_special_tokens=True))
                    else:
                        token_texts.append(str(tid))

                per_token_concepts: List[Dict[str, Any]] = []
                for t_idx in range(t_len):
                    dists = dists_tok[t_idx]
                    k = min(N, dists.shape[0])
                    # smallest distances
                    topk = torch.topk(dists, k=k, largest=False)
                    concept_indices = topk.indices.tolist()
                    concept_scores = topk.values.tolist()
                    top_concepts_tok: List[Dict[str, Any]] = []
                    for rank, (ci, dist_val) in enumerate(zip(concept_indices, concept_scores), 1):
                        top_concepts_tok.append({
                            'rank': rank,
                            'concept_index': ci,
                            'distance': float(dist_val),
                            'similarity': float(1.0 - dist_val),
                            'concept_name': self.concept_names[ci] if self.concept_names and ci < len(self.concept_names) else None,
                            'text_grounding': self.text_grounding[ci] if self.text_grounding and ci < len(self.text_grounding) else None,
                            'image_grounding_path': self.image_grounding_paths[ci] if self.image_grounding_paths and ci < len(self.image_grounding_paths) else None,
                        })
                    per_token_concepts.append({
                        'token_index': t_idx,
                        'token_id': new_ids[-t_len + t_idx],
                        'token_text': token_texts[t_idx],
                        'top_concepts': top_concepts_tok,
                    })

                # Also compute pooled top concepts over the whole sequence (optional aggregate)
                # Filter out non-alphanumeric tokens before pooling
                def _is_alnum(s: str) -> bool:
                    return any(ch.isalnum() for ch in s)

                mask_keep = [
                    _is_alnum(token_texts[t_idx]) if t_idx < len(token_texts) else False
                    for t_idx in range(t_len)
                ]
                if any(mask_keep):
                    acts_keep = acts_j[torch.tensor(mask_keep, dtype=torch.bool)]  # (m, D)
                else:
                    acts_keep = acts_j  # fallback: keep all if none qualify
                pooled = acts_keep.mean(dim=0, keepdim=True)  # (1, D)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                # cosine distance over concepts (smaller is closer)
                sims_all = (pooled @ self.concept_vectors.T)[0]
                dists_all = 1.0 - sims_all
                k_all = min(N, dists_all.shape[0])
                topk_all = torch.topk(dists_all, k=k_all, largest=False)
                idx_all = topk_all.indices.tolist()
                dist_all = topk_all.values.tolist()
                top_concepts_all: List[Dict[str, Any]] = []
                for rank, (ci, dist_val) in enumerate(zip(idx_all, dist_all), 1):
                    top_concepts_all.append({
                        'rank': rank,
                        'concept_index': ci,
                        'distance': float(dist_val),
                        'similarity': float(1.0 - dist_val),
                        'concept_name': self.concept_names[ci] if self.concept_names and ci < len(self.concept_names) else None,
                        'text_grounding': self.text_grounding[ci] if self.text_grounding and ci < len(self.text_grounding) else None,
                        'image_grounding_path': self.image_grounding_paths[ci] if self.image_grounding_paths and ci < len(self.image_grounding_paths) else None,
                    })

                results.append({
                    'image_index': base_idx + j,
                    'ground_truth': lab_chunk[j],
                    'model_output': batch_texts[j],
                    'generated_token_ids': new_ids,
                    'per_token_concepts': per_token_concepts,
                    'top_concepts_over_sequence': top_concepts_all,
                    'layer_activation_shape': tuple(act_seq.shape),
                })

        return results

    def close(self):
        if hasattr(self, '_hook_handle') and self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None


__all__ = ["VLMConceptExplainer"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Quick test for VLMConceptExplainer")
    ap.add_argument('--model_name', default='google/gemma-3n-E4B-it')
    ap.add_argument('--concept_path', required=True)
    ap.add_argument('--layer_path', required=True)
    ap.add_argument('--image', action='append', required=True)
    ap.add_argument('--label', action='append')
    ap.add_argument('--top_n', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=2)
    args = ap.parse_args()

    explainer = VLMConceptExplainer(args.model_name, args.concept_path, args.layer_path)
    res = explainer.explain_with_concept(args.image, ground_truth_labels=args.label, top_n=args.top_n, batch_size=args.batch_size)
    for r in res:
        print(f"Image {r['image_index']} -> gt={r['ground_truth']}\nModel: {r['model_output']}")
        top_list = r.get('top_concepts_over_sequence') or r.get('top_concepts') or []
        print("Top concepts (aggregate):")
        for c in top_list:
            print(f"  #{c['rank']} idx={c['concept_index']} sim={c['similarity']:.4f} name={c['concept_name']} text={c['text_grounding']}")

        # Optional: brief per-token top-1 preview + top-N concept names
        pt = r.get('per_token_concepts') or []
        if pt:
            print("Per-token top-1 (preview) + top-N names:")
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
                    print(
                        f"  t={tok['token_index']} '{tok.get('token_text','')}' -> "
                        f"idx={c['concept_index']} name={c.get('concept_name')} sim={c['similarity']:.4f} | top{n_show}: {names_str}"
                    )
        print('-'*60)
    explainer.close()
