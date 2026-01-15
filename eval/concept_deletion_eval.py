#!/usr/bin/env python3
"""
Concept Deletion/Insertion Evaluation

- Loads a VLM and tokenizer using the repo's model loader
- Dissects the model at layer_path concept space and builds a minimal sub-model: [optional final norm] -> lm_head
- Given a saved explanations JSON (from vlm_explainer.py) and a concepts file (.pth/.pt/.json/.npz),
  computes:
  - c-deletion plots by progressively zeroing concept coordinates ordered by gradient importance
  - c-insertion plots by starting from a zero vector and progressively inserting concept coordinates
  w.r.t. selected target token logits.
- Granularities:
  * sequence: use 'top_concepts_over_sequence' per image (one concept per image at a given rank)
  * token: use 'per_token_concepts' (per generated token) and select a concept by rank per token
- Aggregates softmax probabilities across tokens and images to report mean/std curves.

Usage (deletion, sequence):
python -m eval.concept_deletion_eval \
  --results_json /mnt/abka03/Projects/xl-vlms/outputs/vlm_explanations.json \
  --concept_path /path/to/concepts.pth \
  --layer_path "model.layers.17" \
  --model_name google/gemma-3n-E4B-it \
    --mode sequence --rank 1 --num_points 100 --curve_points 64 --out_dir /mnt/abka03/Projects/xl-vlms/outputs

Usage (insertion, token):
python -m eval.concept_deletion_eval \
  --results_json /mnt/abka03/Projects/xl-vlms/outputs/vlm_explanations.json \
  --concept_path /path/to/concepts.pth \
  --layer_path "model.layers.17" \
  --model_name google/gemma-3n-E4B-it \
    --mode token --insertion --rank 1 --num_points 50 --curve_points 64 --out_dir /mnt/abka03/Projects/xl-vlms/outputs
"""
from __future__ import annotations

import os
import re
import json
import math
import argparse
import gc
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


def _set_env_quiet() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    try:
        import random
        random.seed(seed)
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


def _load_concepts(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Concept file not found: {p}")
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    if p.suffix.lower() in {".pkl", ".pickle"}:
        import pickle
        with p.open("rb") as f:
            return pickle.load(f)
    if p.suffix.lower() in {".pt", ".pth"}:
        return torch.load(str(p), map_location="cpu")
    if p.suffix.lower() == ".npz":
        npz = np.load(str(p), allow_pickle=True)
        return {k: (npz[k].tolist() if hasattr(npz[k], "tolist") else npz[k]) for k in npz.files}
    raise ValueError(f"Unsupported concept file extension: {p.suffix}")


def _prepare_concept_matrix(raw_vectors: Any, normalize: bool = True) -> torch.Tensor:
    if isinstance(raw_vectors, torch.Tensor):
        vecs = raw_vectors.float()
    else:
        vecs = torch.tensor(np.asarray(raw_vectors), dtype=torch.float32)
    if vecs.dim() != 2:
        raise ValueError(f"Concept matrix must be 2D, got {tuple(vecs.shape)}")
    if normalize:
        vecs = F.normalize(vecs, p=2, dim=1)
    return vecs


def _get_tokenizer_from(model_class) -> Any:
    try:
        return getattr(model_class, "get_tokenizer", lambda: None)()
    except Exception:
        return None


class LMHeadSubModel(torch.nn.Module):
    """Minimal head: [optional final norm] -> lm_head.

    Accepts hidden vectors from the hooked layer space (same hidden size),
    and projects to vocabulary logits.
    """

    def __init__(self, full_model: torch.nn.Module) -> None:
        super().__init__()
        # Try common locations for final norm and lm_head
        norm = None
        for name in [
            "model.norm",
            "transformer.ln_f",
            "ln_f",
            "final_layernorm",
            "norm",
        ]:
            mod = self._safe_get(full_model, name)
            if isinstance(mod, torch.nn.Module):
                norm = mod
                break
        lm_head = None
        for name in [
            "lm_head",
            "model.lm_head",
        ]:
            mod = self._safe_get(full_model, name)
            if isinstance(mod, torch.nn.Module):
                lm_head = mod
                break
        if lm_head is None:
            # brute-force: search first Linear matching vocab proj
            for m in full_model.modules():
                if isinstance(m, torch.nn.Linear) and m.out_features > m.in_features:
                    lm_head = m
                    break
        if lm_head is None:
            raise RuntimeError("Failed to locate lm_head module on model")
        self.norm = norm
        self.lm_head = lm_head
        # Keep track of head dtype/device for safe casting
        try:
            p = next(self.lm_head.parameters())
            self.head_dtype = p.dtype
            self.head_device = p.device
        except StopIteration:
            self.head_dtype = torch.float32
            self.head_device = torch.device("cpu")
        # Freeze
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _safe_get(root: torch.nn.Module, path: str) -> Optional[torch.nn.Module]:
        cur = root
        for part in path.split('.'):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (..., hidden_size)
        x = h
        # Ensure dtype matches lm_head weights (fp16/bf16 vs fp32)
        x = x.to(dtype=getattr(self, "head_dtype", x.dtype))
        if self.norm is not None:
            x = self.norm(x)
        logits = self.lm_head(x)  # (..., vocab)
        return logits


class ConceptDeletionEvaluator:
    def __init__(
        self,
        model_name: str,
        layer_path: str,
        concept_path: Union[str, Path],
        results_json: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        normalize_concepts: bool = True,
    cache_dir: Optional[str] = None,
    grad_top_zero_frac: float = 0.15,
    concept_mutiply: bool = True,
    ) -> None:
        _set_env_quiet()
        # Ensure repo src on path for model loader
        project_root = Path(__file__).resolve().parents[1]
        src_dir = project_root / "src"
        if str(src_dir) not in os.sys.path:
            os.sys.path.insert(0, str(src_dir))

        # Load model via repo loader
        from models import get_model_class  # type: ignore
        args = argparse.Namespace(
            local_files_only=False,
            cache_dir=cache_dir or os.environ.get("HF_HOME", "/mnt/abka03/huggingface/hub"),
        )
        device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model_class = get_model_class(
            model_name_or_path=model_name,
            processor_name=model_name,
            device=device,
            logger=None,
            args=args,
        )
        self.model = model_class.get_model().eval()
        self.tokenizer = _get_tokenizer_from(model_class)
        self.device = device
        self.layer_path = layer_path
        # Default smoothing: zero out top fraction of gradient magnitudes when ranking coordinates
        self.grad_top_zero_frac = float(grad_top_zero_frac)
        # Whether to multiply gradient with the concept vector prior to insertion/deletion
        self.concept_mutiply = bool(concept_mutiply)

        # Sub-model from layer space to logits
        self.sub_model = LMHeadSubModel(self.model).to(self.device).eval()

        # Load concepts and results
        concept_data = _load_concepts(concept_path)
        vec_key = (
            "concepts" if "concepts" in concept_data else (
                "activations" if "activations" in concept_data else None
            )
        )
        if vec_key is None:
            raise KeyError("Concept file must contain 'concepts' or 'activations'.")
        self.concepts = _prepare_concept_matrix(concept_data[vec_key], normalize=normalize_concepts)
        self.concepts = self.concepts.to(self.device)
        self.concept_names = concept_data.get("concept_names") or concept_data.get("names")
        self.embed_dim = self.concepts.shape[1]

        with open(results_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.results = payload.get("results", payload)
        # Accept layer_path from file if not provided explicitly
        self.file_layer_path = payload.get("layer_path") if isinstance(payload, dict) else None

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available from model class.")

    # ------------- helpers -------------
    @staticmethod
    def _is_alnum_token_text(txt: str) -> bool:
        return bool(re.search(r"[A-Za-z0-9]", txt or ""))

    def _tokenize_with_texts(self, text: str) -> Tuple[List[int], List[str]]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        toks: List[str] = []
        for tid in ids:
            try:
                toks.append(self.tokenizer.decode([tid], skip_special_tokens=True))
            except Exception:
                toks.append(str(tid))
        return ids, toks

    def _grad_and_order(self, vec: torch.Tensor, target_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradient d logit[target_id] / d vec and return (grad, sorted_indices).

        Ranking rule:
        - If self.concept_mutiply is True, sort by descending |grad * vec| (elementwise),
          then optionally skip a top fraction controlled by self.grad_top_zero_frac.
        - Otherwise, sort by descending |grad| with the same optional skip.
        Only the order is affected; probability computations always use the original vec.
        """
        v = vec.detach().clone().to(self.device, dtype=getattr(self.sub_model, "head_dtype", vec.dtype)).requires_grad_(True)
        logits = self.sub_model(v)  # (V)
        if logits.dim() > 1:
            logits = logits.squeeze(0)
        logit_t = logits[target_id]
        self.sub_model.zero_grad(set_to_none=True)
        if v.grad is not None:
            v.grad.zero_()
        logit_t.backward(retain_graph=False)
        g = v.grad.detach()
        # Rank importance by |grad| or |grad * vec| (most -> least)
        if getattr(self, "concept_mutiply", False):
            # Ensure dtype alignment for multiplication (use vec's dtype)
            importance = v.abs() # (g.to(vec.dtype) * vec.abs()) #  vec.abs() works fine
        else:
            importance = g.abs()
        order_all = torch.argsort(importance, dim=-1, descending=True)
        # If a top fraction is specified, skip those coordinates entirely from evaluation
        frac = float(getattr(self, "grad_top_zero_frac", 0.0) or 0.0)
        if frac > 0.0 and order_all.numel() > 0:
            skip_n = int(math.ceil(min(1.0, max(0.0, frac)) * order_all.numel()))
            if skip_n < order_all.numel():
                order = order_all[skip_n:]
            else:
                order = order_all[-1:].clone()
        else:
            order = order_all
        return g, order

    def _prob_curve_by_mask_order(
        self,
        vec: torch.Tensor,
        order: torch.Tensor,
        target_id: int,
        ks: Iterable[int],
    ) -> List[float]:
        base = vec.detach().clone().to(self.device, dtype=getattr(self.sub_model, "head_dtype", vec.dtype))
        probs: List[float] = []
        vocab_dim = None
        for k in ks:
            v_mask = base.clone()
            if k > 0:
                v_mask[order[:k]] = 0.0
            logits = self.sub_model(v_mask)
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            if vocab_dim is None:
                vocab_dim = logits.shape[-1]
            p = F.softmax(logits, dim=-1)[int(target_id)]
            probs.append(float(p.detach().cpu()))
        return probs

    def _prob_curve_by_mask_order_insertion(
        self,
        vec: torch.Tensor,
        order: torch.Tensor,
        target_id: int,
        ks: Iterable[int],
    ) -> List[float]:
        """Start from zero vector and progressively insert coordinates of vec following order."""
        base = vec.detach().clone().to(self.device, dtype=getattr(self.sub_model, "head_dtype", vec.dtype))
        probs: List[float] = []
        for k in ks:
            v_mask = torch.zeros_like(base)
            if k > 0:
                idx = order[:k]
                v_mask[idx] = base[idx]
            logits = self.sub_model(v_mask)
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            p = F.softmax(logits, dim=-1)[int(target_id)]
            probs.append(float(p.detach().cpu()))
        return probs

    @staticmethod
    def _build_ks(embed_dim: int, pct: float, curve_points: int) -> List[int]:
        """Build mask k's as percentages of the effective dimension.

        pct is interpreted as 0-100 (percentage of coordinates to traverse).
        curve_points controls the resolution of the curve (number of samples along that percentage).
        Always includes k=0 and k=ceil(pct% * embed_dim).
        """
        pct = max(0.0, min(100.0, float(pct)))
        max_k = int(math.ceil((pct / 100.0) * float(embed_dim)))
        max_k = max(0, min(embed_dim, max_k))
        if curve_points <= 1 or max_k == 0:
            return [0, max_k]
        xs = np.linspace(0, max_k, int(curve_points))
        ks = sorted(set(int(round(x)) for x in xs))
        if ks[0] != 0:
            ks.insert(0, 0)
        if ks[-1] != max_k:
            ks.append(max_k)
        return ks

    # ------------- evaluation -------------
    def evaluate_sequence(
        self,
    rank: int = 1,
    num_points: float = 100,
        curve_points: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sequence-mode: one concept per image (top_concepts_over_sequence),
        tokenize model_output and evaluate probabilities for all alnum tokens.
        Returns (fractions, mean_probs, std_probs).
        """
        curves: List[List[float]] = []
        # Effective dimension excludes the skipped top-|grad| fraction
        eff_dim = self.embed_dim
        if getattr(self, "grad_top_zero_frac", 0.0):
            eff_dim = max(1, self.embed_dim - int(math.ceil(float(self.grad_top_zero_frac) * self.embed_dim)))
        ks = self._build_ks(eff_dim, num_points, curve_points)
        for item in self.results:
            top_list = item.get("top_concepts_over_sequence") or []
            if not top_list:
                continue
            # pick concept at rank
            chosen = None
            for c in top_list:
                if int(c.get("rank", -1)) == int(rank):
                    chosen = c
                    break
            if chosen is None:
                # default to first if specified rank not present
                chosen = top_list[0]
            ci = int(chosen.get("concept_index", 0))
            if ci < 0 or ci >= self.concepts.shape[0]:
                continue
            vec = self.concepts[ci]
            # targets from model output text
            text = item.get("model_output", "")
            ids, toks = self._tokenize_with_texts(text)
            # gather alnum token ids
            target_ids = [tid for tid, ttxt in zip(ids, toks) if self._is_alnum_token_text(ttxt)]
            if not target_ids:
                continue
            # one gradient ordering per target id (compute separately)
            for tid in target_ids:
                _, order = self._grad_and_order(vec, int(tid))
                probs = self._prob_curve_by_mask_order(vec, order, int(tid), ks)
                curves.append(probs)
        if not curves:
            raise RuntimeError("No curves computed; check inputs and explanation data.")
        arr = np.stack(curves, axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        fracs = np.array([k / float(eff_dim) for k in ks], dtype=np.float32)
        return fracs, mean, std

    def evaluate_token(
        self,
    rank: int = 1,
    num_points: float = 100,
        curve_points: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Token-mode: for each token's explanation, pick the concept at given rank
        and use the token's id as target. Aggregate across all valid tokens and images.
        Returns (fractions, mean_probs, std_probs).
        """
        curves: List[List[float]] = []
        eff_dim = self.embed_dim
        if getattr(self, "grad_top_zero_frac", 0.0):
            eff_dim = max(1, self.embed_dim - int(math.ceil(float(self.grad_top_zero_frac) * self.embed_dim)))
        ks = self._build_ks(eff_dim, num_points, curve_points)
        for item in self.results:
            toks = item.get("per_token_concepts") or []
            for tok in toks:
                token_id = int(tok.get("token_id", -1))
                token_text = str(tok.get("token_text", ""))
                if token_id < 0 or not self._is_alnum_token_text(token_text):
                    continue
                top_list = tok.get("top_concepts") or []
                chosen = None
                for c in top_list:
                    if int(c.get("rank", -1)) == int(rank):
                        chosen = c
                        break
                if chosen is None:
                    if top_list:
                        chosen = top_list[0]
                    else:
                        continue
                ci = int(chosen.get("concept_index", 0))
                if ci < 0 or ci >= self.concepts.shape[0]:
                    continue
                vec = self.concepts[ci]
                _, order = self._grad_and_order(vec, token_id)
                probs = self._prob_curve_by_mask_order(vec, order, token_id, ks)
                curves.append(probs)
        if not curves:
            raise RuntimeError("No curves computed; check inputs and token explanations.")
        arr = np.stack(curves, axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        fracs = np.array([k / float(eff_dim) for k in ks], dtype=np.float32)
        return fracs, mean, std

    def evaluate_sequence_insertion(
        self,
    rank: int = 1,
    num_points: float = 100,
        curve_points: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sequence-mode c-insertion: one concept per image, insert coordinates from high→low grad."""
        curves: List[List[float]] = []
        eff_dim = self.embed_dim
        if getattr(self, "grad_top_zero_frac", 0.0):
            eff_dim = max(1, self.embed_dim - int(math.ceil(float(self.grad_top_zero_frac) * self.embed_dim)))
        ks = self._build_ks(eff_dim, num_points, curve_points)
        for item in self.results:
            top_list = item.get("top_concepts_over_sequence") or []
            if not top_list:
                continue
            chosen = None
            for c in top_list:
                if int(c.get("rank", -1)) == int(rank):
                    chosen = c
                    break
            if chosen is None:
                chosen = top_list[0]
            ci = int(chosen.get("concept_index", 0))
            if ci < 0 or ci >= self.concepts.shape[0]:
                continue
            vec = self.concepts[ci]
            text = item.get("model_output", "")
            ids, toks = self._tokenize_with_texts(text)
            target_ids = [tid for tid, ttxt in zip(ids, toks) if self._is_alnum_token_text(ttxt)]
            if not target_ids:
                continue
            for tid in target_ids:
                _, order = self._grad_and_order(vec, int(tid))
                probs = self._prob_curve_by_mask_order_insertion(vec, order, int(tid), ks)
                curves.append(probs)
        if not curves:
            raise RuntimeError("No curves computed; check inputs and explanation data.")
        arr = np.stack(curves, axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        fracs = np.array([k / float(eff_dim) for k in ks], dtype=np.float32)
        return fracs, mean, std

    def evaluate_token_insertion(
        self,
    rank: int = 1,
    num_points: float = 100,
        curve_points: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Token-mode c-insertion: per-token concept at given rank and token id as target."""
        curves: List[List[float]] = []
        eff_dim = self.embed_dim
        if getattr(self, "grad_top_zero_frac", 0.0):
            eff_dim = max(1, self.embed_dim - int(math.ceil(float(self.grad_top_zero_frac) * self.embed_dim)))
        ks = self._build_ks(eff_dim, num_points, curve_points)
        for item in self.results:
            toks = item.get("per_token_concepts") or []
            for tok in toks:
                token_id = int(tok.get("token_id", -1))
                token_text = str(tok.get("token_text", ""))
                if token_id < 0 or not self._is_alnum_token_text(token_text):
                    continue
                top_list = tok.get("top_concepts") or []
                chosen = None
                for c in top_list:
                    if int(c.get("rank", -1)) == int(rank):
                        chosen = c
                        break
                if chosen is None:
                    if top_list:
                        chosen = top_list[0]
                    else:
                        continue
                ci = int(chosen.get("concept_index", 0))
                if ci < 0 or ci >= self.concepts.shape[0]:
                    continue
                vec = self.concepts[ci]
                _, order = self._grad_and_order(vec, token_id)
                probs = self._prob_curve_by_mask_order_insertion(vec, order, token_id, ks)
                curves.append(probs)
        if not curves:
            raise RuntimeError("No curves computed; check inputs and token explanations.")
        arr = np.stack(curves, axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        fracs = np.array([k / float(eff_dim) for k in ks], dtype=np.float32)
        return fracs, mean, std

    # ------------- plotting -------------
    @staticmethod
    def plot_and_save(
        fracs: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        title: str,
        out_png: Union[str, Path],
        xlabel: Optional[str] = None,
    ) -> None:
        if not _HAS_PLT:
            return
        plt.figure(figsize=(6, 4))
        plt.plot(fracs, mean, label="mean prob", color="C0")
        plt.fill_between(fracs, mean - std, mean + std, color="C0", alpha=0.2, label="±1 std")
        plt.xlabel(xlabel or "fraction of concept coordinates zeroed (most → least important)")
        plt.ylabel("softmax probability of target token")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(str(out_png)), exist_ok=True)
        plt.savefig(str(out_png), dpi=160)
        plt.close()


def run_with_args(args: argparse.Namespace) -> None:
    """Run concept deletion/insertion evaluation using a parsed Namespace."""
    _seed_everything(args.seed, deterministic=args.deterministic)

    # Resolve layer_path from file if not provided
    file_layer_path = None
    try:
        with open(args.results_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
            if isinstance(payload, dict):
                file_layer_path = payload.get("layer_path")
    except Exception:
        pass
    layer_path = args.layer_path or (file_layer_path or "")

    evaluator = ConceptDeletionEvaluator(
        model_name=args.model_name,
        layer_path=layer_path,
        concept_path=args.concept_path,
        results_json=args.results_json,
        device=args.device,
        grad_top_zero_frac=args.grad_top_zero_frac,
        concept_mutiply=getattr(args, "concept_mutiply", True),
    )

    xlabel = None
    if args.mode == "sequence":
        if args.insertion:
            fracs, mean, std = evaluator.evaluate_sequence_insertion(rank=args.rank, num_points=args.num_points, curve_points=args.curve_points)
            title = f"Concept insertion (sequence, rank={args.rank})"
            base = f"c_insertion_sequence_rank{args.rank}.png"
            xlabel = "fraction of concept coordinates inserted (most → least important)"
        else:
            fracs, mean, std = evaluator.evaluate_sequence(rank=args.rank, num_points=args.num_points, curve_points=args.curve_points)
            title = f"Concept deletion (sequence, rank={args.rank})"
            base = f"c_deletion_sequence_rank{args.rank}.png"
            xlabel = "fraction of concept coordinates zeroed (most → least important)"
    else:
        if args.insertion:
            fracs, mean, std = evaluator.evaluate_token_insertion(rank=args.rank, num_points=args.num_points, curve_points=args.curve_points)
            title = f"Concept insertion (token, rank={args.rank})"
            base = f"c_insertion_token_rank{args.rank}.png"
            xlabel = "fraction of concept coordinates inserted (most → least important)"
        else:
            fracs, mean, std = evaluator.evaluate_token(rank=args.rank, num_points=args.num_points, curve_points=args.curve_points)
            title = f"Concept deletion (token, rank={args.rank})"
            base = f"c_deletion_token_rank{args.rank}.png"
            xlabel = "fraction of concept coordinates zeroed (most → least important)"

    # Save plot
    ConceptDeletionEvaluator.plot_and_save(fracs, mean, std, title, Path(args.out_dir) / base, xlabel=xlabel)

    # Also dump CSV and JSON
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = Path(args.out_dir) / base.replace(".png", ".csv")
    out_json = Path(args.out_dir) / base.replace(".png", ".json")
    try:
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            header0 = "fraction_inserted" if args.insertion else "fraction_zeroed"
            w.writerow([header0, "mean_prob", "std_prob"])
            for x, m, s in zip(fracs.tolist(), mean.tolist(), std.tolist()):
                w.writerow([x, m, s])
    except Exception:
        pass
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "fractions": fracs.tolist(),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "mode": args.mode,
                "insertion": args.insertion,
                "rank": args.rank,
                "layer_path": layer_path,
                "model_name": args.model_name,
                "results_json": args.results_json,
                "concept_path": args.concept_path,
                "grad_top_zero_frac": args.grad_top_zero_frac,
                "concept_mutiply": getattr(args, "concept_mutiply", True),
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    # pro-actively release model weights and CUDA caches to avoid accumulation across runs
    evaluator = None
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    

def main() -> None:
    ap = argparse.ArgumentParser(description="Concept deletion/insertion evaluation (c-deletion / c-insertion)")
    ap.add_argument("--results_json", required=True, help="JSON file produced by vlm_explainer.py")
    ap.add_argument("--concept_path", required=True, help="Concept matrix file (.pth/.pt/.json/.npz)")
    ap.add_argument("--model_name", default="google/gemma-3n-E4B-it")
    ap.add_argument("--layer_path", required=False, help="Hooked layer path; if omitted, read from results JSON")
    ap.add_argument("--mode", choices=["sequence", "token"], default="sequence")
    ap.add_argument("--insertion", action="store_true", help="Use c-insertion instead of c-deletion")
    ap.add_argument("--rank", type=int, default=1, help="Concept rank to evaluate (1 = top)")
    # Interpret --num_points as percentage of the gradient vector length to traverse (0-100)
    ap.add_argument("--num_points", type=float, default=100.0, help="Percentage of coordinates to traverse (0-100). 100 = full vector")
    # New: curve resolution along that percentage range
    ap.add_argument("--curve_points", type=int, default=64, help="Number of evaluation samples along the selected percentage range")
    ap.add_argument("--device", default=None, help="cuda, cpu, or cuda:N")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out_dir", default="/mnt/abka03/Projects/xl-vlms/outputs")
    # New: smoothing fraction for zeroing top-|grad| before ranking
    ap.add_argument("--grad_top_zero_frac", type=float, default=0.0, help="Fraction of top-|grad| coordinates to zero before ranking (smoothing)")
    # New: whether to multiply gradient with concept vector prior to op
    try:
        from argparse import BooleanOptionalAction  # py3.9+
        ap.add_argument("--concept_multiply", action=BooleanOptionalAction, default=True,
                        help="If true (default), multiply gradient with the concept vector before deletion/insertion")
    except Exception:
        # Fallback: presence of flag sets True; no negation flag
        ap.add_argument("--concept_multiply", action="store_true", default=True,
                        help="If true, multiply gradient with the concept vector when ordering (default: False)")
    args = ap.parse_args()

    run_with_args(args)


if __name__ == "__main__":
    main()
