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


def _curve_auc_start_relative(fracs: np.ndarray, y: np.ndarray) -> Optional[float]:
    """AUC of the curve normalized by its OWN fraction=0 value (y[0], the
    true unperturbed baseline for that image/rank -- nothing zeroed yet),
    not by its full min/max range. Needed for cross-RANK comparisons (rank 1
    vs 2 vs 3): rank-1's concept is, by construction, the most strongly
    activating one for a token, so its curve starts from a genuinely higher
    baseline than rank-2/3's -- min-max normalization (or raw AUC) lets that
    baseline difference dominate, unfairly rewarding/penalizing a rank for
    how large its own concept's absolute weight happens to be rather than
    how faithfully deleting its top coordinates crashes ITS OWN confidence.
    Dividing by y[0] fixes one real, principled reference point (the actual
    unperturbed probability) instead of squashing between two arbitrary
    curve-specific extrema, so the ending value still carries real
    information about how much the curve actually falls."""
    y0 = float(y[0])
    if y0 == 0.0:
        return None
    y_rel = y / y0
    span = float(np.max(fracs) - np.min(fracs))
    if span <= 0.0:
        return None
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y_rel, fracs)) / span


def _curve_auc_relative(fracs: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Same min-max-rescaled AUC as eval/concept_curve_auc_eval.py::_compute_auc_relative,
    applied to a single (per-image) curve instead of the aggregate mean curve."""
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_range = y_max - y_min
    if y_range <= 0.0:
        return None
    y_rel = (y - y_min) / y_range
    span = float(np.max(fracs) - np.min(fracs))
    if span <= 0.0:
        return None
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y_rel, fracs)) / span


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
        from device_utils import get_device_config  # type: ignore
        args = argparse.Namespace(
            local_files_only=False,
            cache_dir=cache_dir or os.environ.get("HF_HOME"),
        )
        device_config = get_device_config(
            str(device) if device is not None else None
        )
        device = device_config.primary_device
        model_class = get_model_class(
            model_name_or_path=model_name,
            processor_name=model_name,
            device=device,
            logger=None,
            args=args,
            device_config=device_config,
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

        # Real per-token activations from actual grid inference (saved by
        # vlm_explainer_multibatch.py alongside the JSON) -- insertion/
        # deletion should ablate concept coordinates within what the model
        # ACTUALLY computed for that image/token, not within the standalone
        # concept vector in isolation. Falls back to None (callers fall back
        # to the concept vector itself) for older explanation files saved
        # before this existed.
        self.activations: Optional[torch.Tensor] = None
        activations_path = payload.get("activations_path") if isinstance(payload, dict) else None
        if activations_path and Path(activations_path).exists():
            arr = np.load(activations_path)
            self.activations = torch.as_tensor(arr, dtype=torch.float32)

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
            importance = (g.to(vec.dtype) * vec).abs()
        else:
            importance = g.abs()
        order_all = torch.argsort(importance, dim=-1, descending=True)
        order = self._apply_skip_frac(order_all)
        return g, order

    def _apply_skip_frac(self, order_all: torch.Tensor) -> torch.Tensor:
        """Skip the configured top fraction of an ordering (shared by gradient
        and random orderings, so both traverse the same effective dimension)."""
        frac = float(getattr(self, "grad_top_zero_frac", 0.0) or 0.0)
        if frac > 0.0 and order_all.numel() > 0:
            skip_n = int(math.ceil(min(1.0, max(0.0, frac)) * order_all.numel()))
            if skip_n < order_all.numel():
                return order_all[skip_n:]
            return order_all[-1:].clone()
        return order_all

    def _order_for(self, vec: torch.Tensor, target_id: int, order_mode: str) -> torch.Tensor:
        """Dispatch to the requested ordering: 'value' (default), 'random',
        or 'gradient' (kept available, no longer the pipeline default)."""
        if order_mode == "random":
            return self._random_order(vec.shape[-1])
        if order_mode == "gradient":
            _, order = self._grad_and_order(vec, target_id)
            return order
        return self._value_order(vec)

    def _real_vec_for(self, tok: Dict[str, Any], fallback: torch.Tensor) -> torch.Tensor:
        """Real per-token activation for this token (what the model ACTUALLY
        computed for that image), if available; falls back to the concept
        vector itself for explanation files saved before activations were
        persisted."""
        idx = tok.get("activation_index")
        if self.activations is not None and idx is not None and 0 <= int(idx) < self.activations.shape[0]:
            return self.activations[int(idx)].to(fallback.device)
        return fallback

    def _value_order(self, vec: torch.Tensor) -> torch.Tensor:
        """Value-magnitude ordering: sort coordinates by descending |vec|,
        no gradient involved. This is the default ordering -- gradient-based
        ranking (|grad * vec|) weights coordinates by how much a local
        perturbation moves the target logit, which can heavily reweight which
        coordinates look "important" relative to their actual magnitude in
        the concept vector; pure value ordering instead tests the concept
        vector's own largest components directly."""
        order_all = torch.argsort(vec.detach().abs(), dim=-1, descending=True)
        return self._apply_skip_frac(order_all)

    def _random_order(self, dim: int) -> torch.Tensor:
        """Random-baseline ordering: a random permutation of coordinates, with
        the same skip-fraction semantics as gradient ordering, so insertion/
        deletion curves are comparable apples-to-apples against chance. Uses
        the global RNG seeded by _seed_everything for reproducibility."""
        order_all = torch.randperm(dim, device=self.device)
        return self._apply_skip_frac(order_all)

    def _prob_curve_by_mask_order(
        self,
        vec: torch.Tensor,
        order: torch.Tensor,
        target_id: int,
        ks: Iterable[int],
        batch_size: int = 256,
    ) -> List[float]:
        """Batched: build every masked state as one [K, embed_dim] tensor and
        run the (cheap) sub_model (norm + lm_head only, no transformer
        layers) in one or a few forward passes instead of one pass per k.
        The unbatched version issued K sequential forward calls per curve --
        with curve_points~100 and thousands of curves per config, that pure
        Python-call overhead dominated wall time (GPU sat near-idle between
        calls). Batching gives the same math, ~50-100x faster."""
        ks = list(ks)
        base = vec.detach().clone().to(self.device, dtype=getattr(self.sub_model, "head_dtype", vec.dtype))
        v_batch = base.unsqueeze(0).repeat(len(ks), 1)  # [K, embed_dim]
        for i, k in enumerate(ks):
            if k > 0:
                v_batch[i, order[:k]] = 0.0
        probs: List[float] = []
        with torch.no_grad():
            for start in range(0, len(ks), batch_size):
                chunk = v_batch[start:start + batch_size]
                logits = self.sub_model(chunk)  # [B, vocab]
                p = F.softmax(logits, dim=-1)[:, int(target_id)]
                probs.extend(float(x) for x in p.detach().cpu())
        return probs

    def _prob_curve_by_mask_order_insertion(
        self,
        vec: torch.Tensor,
        order: torch.Tensor,
        target_id: int,
        ks: Iterable[int],
        batch_size: int = 256,
    ) -> List[float]:
        """Start from zero vector and progressively insert coordinates of vec
        following order. Batched -- see _prob_curve_by_mask_order docstring."""
        ks = list(ks)
        base = vec.detach().clone().to(self.device, dtype=getattr(self.sub_model, "head_dtype", vec.dtype))
        v_batch = torch.zeros((len(ks), base.shape[-1]), device=base.device, dtype=base.dtype)
        for i, k in enumerate(ks):
            if k > 0:
                idx = order[:k]
                v_batch[i, idx] = base[idx]
        probs: List[float] = []
        with torch.no_grad():
            for start in range(0, len(ks), batch_size):
                chunk = v_batch[start:start + batch_size]
                logits = self.sub_model(chunk)  # [B, vocab]
                p = F.softmax(logits, dim=-1)[:, int(target_id)]
                probs.extend(float(x) for x in p.detach().cpu())
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
        order_mode: str = "gradient",
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
            # one ordering per target id (compute separately)
            for tid in target_ids:
                order = self._order_for(vec, int(tid), order_mode)
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
        order_mode: str = "gradient",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Token-mode: for each token's explanation, pick the concept at given rank
        and use the token's id as target. Aggregate across all valid tokens and images.
        Returns (fractions, mean_probs, std_probs).
        """
        curves: List[List[float]] = []
        # Rebuttal per-image AUC: tracked alongside (not instead of) the flat
        # `curves` aggregate above, keyed by image_path -- see run_with_args()
        # and self._last_per_image, which the smoke test/regression checks
        # confirm doesn't alter the pre-existing (fracs, mean, std) contract.
        per_image_curves: Dict[str, List[List[float]]] = {}
        eff_dim = self.embed_dim
        if getattr(self, "grad_top_zero_frac", 0.0):
            eff_dim = max(1, self.embed_dim - int(math.ceil(float(self.grad_top_zero_frac) * self.embed_dim)))
        ks = self._build_ks(eff_dim, num_points, curve_points)
        for item in self.results:
            image_path = item.get("image_path")
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
                order = self._order_for(vec, token_id, order_mode)
                probs = self._prob_curve_by_mask_order(vec, order, token_id, ks)
                curves.append(probs)
                if image_path is not None:
                    per_image_curves.setdefault(image_path, []).append(probs)
        if not curves:
            raise RuntimeError("No curves computed; check inputs and token explanations.")
        arr = np.stack(curves, axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        fracs = np.array([k / float(eff_dim) for k in ks], dtype=np.float32)
        self._last_per_image = self._build_per_image_auc(per_image_curves, fracs)
        return fracs, mean, std

    def _build_per_image_auc(
        self, per_image_curves: Dict[str, List[List[float]]], fracs: np.ndarray
    ) -> List[Dict[str, Any]]:
        rows = []
        # Also stash the RAW per-image mean curves (not just their AUC) so a
        # caller that wants post-hoc AUC/scaling without re-running the model
        # (e.g. scripts/gen_raw_prob_curves.py) can read them off the
        # evaluator. self._last_per_image (AUC scalars) is unchanged.
        self._last_per_image_curves = {}
        self._last_fracs = np.asarray(fracs, dtype=np.float64)
        for image_path, item_curves in per_image_curves.items():
            item_arr = np.asarray(item_curves, dtype=np.float64)
            item_mean = item_arr.mean(axis=0)
            trapezoid = getattr(np, "trapezoid", np.trapz)
            fracs64 = fracs.astype(np.float64)
            self._last_per_image_curves[image_path] = {
                "curve": item_mean,               # raw predicted-token prob per masking step
                "n_tokens": len(item_curves),
            }
            rows.append({
                "image_path": image_path,
                "auc": float(trapezoid(item_mean, fracs64)),
                "auc_relative": _curve_auc_relative(fracs64, item_mean),
                "auc_start_relative": _curve_auc_start_relative(fracs64, item_mean),
                "y0": float(item_mean[0]),
                "n_tokens": len(item_curves),
            })
        return rows

    def evaluate_whole_concept_token(self, rank: int = 1, random_baseline: bool = True) -> Dict[str, float]:
        """Whole-vector insert/delete: for each token, measure the model's
        confidence in its OWN generated word with the rank-N concept vector
        fully present vs fully absent (zero vector) -- no coordinate sweep,
        no ordering method (value/gradient/random coordinate order doesn't
        apply here, since there's no partial state between "off" and "on").

        This directly answers "does THIS concept, as a whole, move the
        probability of the token the model actually said" -- independently
        per rank, so rank1/rank2/rank3 can be compared on equal footing by
        their own delta (prob_with - prob_without), rather than by the shape
        of a per-coordinate sweep curve.

        If random_baseline is True, also computes the same delta for a
        randomly chosen OTHER concept (not the one assigned to this token) as
        a sanity check: the real assigned concept's delta should exceed a
        random concept's delta if the assignment is meaningful.
        """
        rng = torch.Generator(device="cpu")
        deltas: List[float] = []
        with_vals: List[float] = []
        without_vals: List[float] = []
        random_deltas: List[float] = []
        n_concepts = self.concepts.shape[0]

        def _prob_for(vec: torch.Tensor, target_id: int) -> float:
            v = vec.detach().clone().to(self.device, dtype=getattr(self.sub_model, "head_dtype", vec.dtype))
            logits = self.sub_model(v)
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            return float(F.softmax(logits, dim=-1)[int(target_id)].detach().cpu())

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
                if ci < 0 or ci >= n_concepts:
                    continue
                vec = self.concepts[ci]
                zero_vec = torch.zeros_like(vec)
                prob_with = _prob_for(vec, token_id)
                prob_without = _prob_for(zero_vec, token_id)
                with_vals.append(prob_with)
                without_vals.append(prob_without)
                deltas.append(prob_with - prob_without)

                if random_baseline and n_concepts > 1:
                    rand_ci = int(torch.randint(0, n_concepts - 1, (1,), generator=rng).item())
                    if rand_ci >= ci:
                        rand_ci += 1  # skip the real concept index, stays uniform over the rest
                    rand_vec = self.concepts[rand_ci]
                    prob_with_random = _prob_for(rand_vec, token_id)
                    random_deltas.append(prob_with_random - prob_without)

        if not deltas:
            raise RuntimeError("No whole-concept measurements computed; check inputs and token explanations.")

        result = {
            "rank": rank,
            "prob_with_mean": float(np.mean(with_vals)),
            "prob_with_std": float(np.std(with_vals)),
            "prob_without_mean": float(np.mean(without_vals)),
            "prob_without_std": float(np.std(without_vals)),
            "delta_mean": float(np.mean(deltas)),
            "delta_std": float(np.std(deltas)),
            "n_tokens": len(deltas),
        }
        if random_baseline and random_deltas:
            result["delta_random_mean"] = float(np.mean(random_deltas))
            result["delta_random_std"] = float(np.std(random_deltas))
        return result

    def evaluate_sequence_insertion(
        self,
    rank: int = 1,
    num_points: float = 100,
        curve_points: int = 64,
        order_mode: str = "gradient",
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
                order = self._order_for(vec, int(tid), order_mode)
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
        order_mode: str = "gradient",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Token-mode c-insertion: per-token concept at given rank and token id as target."""
        curves: List[List[float]] = []
        per_image_curves: Dict[str, List[List[float]]] = {}
        eff_dim = self.embed_dim
        if getattr(self, "grad_top_zero_frac", 0.0):
            eff_dim = max(1, self.embed_dim - int(math.ceil(float(self.grad_top_zero_frac) * self.embed_dim)))
        ks = self._build_ks(eff_dim, num_points, curve_points)
        for item in self.results:
            image_path = item.get("image_path")
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
                order = self._order_for(vec, token_id, order_mode)
                probs = self._prob_curve_by_mask_order_insertion(vec, order, token_id, ks)
                curves.append(probs)
                if image_path is not None:
                    per_image_curves.setdefault(image_path, []).append(probs)
        if not curves:
            raise RuntimeError("No curves computed; check inputs and token explanations.")
        arr = np.stack(curves, axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        fracs = np.array([k / float(eff_dim) for k in ks], dtype=np.float32)
        self._last_per_image = self._build_per_image_auc(per_image_curves, fracs)
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

        # Raw target-token probability sits in a tiny absolute band (~1/vocab
        # size, e.g. ~4e-6) with per-token std often comparable to or larger
        # than the curve's own range across the sweep -- plotted raw, the
        # mean line is swallowed by its own +/-1 std shading and different
        # ranks become visually indistinguishable even when their AUC-relative
        # values differ meaningfully. Min-max rescale mean to its own [0, 1]
        # range (same transform as eval/concept_curve_auc_eval.py's
        # _compute_auc_relative) so the plot actually shows the shape being
        # measured; rescale std by the same factor so the shaded band stays
        # proportionally meaningful instead of dominating the line.
        y_min, y_max = float(np.min(mean)), float(np.max(mean))
        y_range = y_max - y_min
        if y_range > 0:
            mean_plot = (mean - y_min) / y_range
            std_plot = std / y_range
            ylabel = "relative softmax probability (min-max rescaled to this curve's own range)"
        else:
            mean_plot, std_plot = mean, std
            ylabel = "softmax probability of target token"

        plt.figure(figsize=(6, 4))
        plt.plot(fracs, mean_plot, label="mean (rescaled)", color="C0")
        plt.fill_between(fracs, mean_plot - std_plot, mean_plot + std_plot, color="C0", alpha=0.2, label="±1 std")
        plt.xlabel(xlabel or "fraction of concept coordinates zeroed (most → least important)")
        plt.ylabel(ylabel)
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

    order_mode = getattr(args, "order_mode", "value")
    suffix = "_random" if order_mode == "random" else ""
    title_tag = " [random baseline]" if order_mode == "random" else ""

    if args.mode == "whole_concept":
        result = evaluator.evaluate_whole_concept_token(rank=args.rank, random_baseline=True)
        out_json = Path(args.out_dir) / f"c_whole_concept_rank{args.rank}.json"
        out_csv = Path(args.out_dir) / f"c_whole_concept_rank{args.rank}.csv"
        os.makedirs(args.out_dir, exist_ok=True)
        payload = {
            **result,
            "layer_path": layer_path,
            "model_name": args.model_name,
            "results_json": args.results_json,
            "concept_path": args.concept_path,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            keys = list(result.keys())
            w.writerow(keys)
            w.writerow([result[k] for k in keys])
        print(f"Saved whole-concept rank={args.rank}: delta_mean={result['delta_mean']:.6g} "
              f"delta_random_mean={result.get('delta_random_mean', float('nan')):.6g} "
              f"(n_tokens={result['n_tokens']})")
        return

    xlabel = None
    if args.mode == "sequence":
        if args.insertion:
            fracs, mean, std = evaluator.evaluate_sequence_insertion(rank=args.rank, num_points=args.num_points, curve_points=args.curve_points, order_mode=order_mode)
            title = f"Concept insertion (sequence, rank={args.rank}){title_tag}"
            base = f"c_insertion_sequence_rank{args.rank}{suffix}.png"
            xlabel = "fraction of concept coordinates inserted (most → least important)"
        else:
            fracs, mean, std = evaluator.evaluate_sequence(rank=args.rank, num_points=args.num_points, curve_points=args.curve_points, order_mode=order_mode)
            title = f"Concept deletion (sequence, rank={args.rank}){title_tag}"
            base = f"c_deletion_sequence_rank{args.rank}{suffix}.png"
            xlabel = "fraction of concept coordinates zeroed (most → least important)"
    else:
        if args.insertion:
            fracs, mean, std = evaluator.evaluate_token_insertion(rank=args.rank, num_points=args.num_points, curve_points=args.curve_points, order_mode=order_mode)
            title = f"Concept insertion (token, rank={args.rank}){title_tag}"
            base = f"c_insertion_token_rank{args.rank}{suffix}.png"
            xlabel = "fraction of concept coordinates inserted (most → least important)"
        else:
            fracs, mean, std = evaluator.evaluate_token(rank=args.rank, num_points=args.num_points, curve_points=args.curve_points, order_mode=order_mode)
            title = f"Concept deletion (token, rank={args.rank}){title_tag}"
            base = f"c_deletion_token_rank{args.rank}{suffix}.png"
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
                "order_mode": order_mode,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Rebuttal per-image AUC (token mode only -- sequence mode has no
    # per-item granularity to preserve beyond what's already in the
    # aggregate). Written as a sibling file; concept_curve_auc_eval.py only
    # reads the un-suffixed base JSON/CSV above, so this is purely additive.
    per_image_rows = getattr(evaluator, "_last_per_image", None) if args.mode == "token" else None
    if per_image_rows:
        per_image_csv = Path(args.out_dir) / base.replace(".png", "_per_image.csv")
        per_image_json = Path(args.out_dir) / base.replace(".png", "_per_image.json")
        try:
            import csv
            with open(per_image_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_path", "auc", "auc_relative", "auc_start_relative", "y0", "n_tokens"])
                for row in per_image_rows:
                    w.writerow([row["image_path"], row["auc"], row["auc_relative"],
                                row.get("auc_start_relative"), row.get("y0"), row["n_tokens"]])
        except Exception:
            pass
        try:
            with open(per_image_json, "w", encoding="utf-8") as f:
                json.dump({
                    "rows": per_image_rows,
                    "mode": args.mode,
                    "insertion": args.insertion,
                    "rank": args.rank,
                    "layer_path": layer_path,
                    "results_json": args.results_json,
                    "order_mode": order_mode,
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
    ap.add_argument("--model_name", default=os.environ.get('VLM_MODEL', 'Qwen/Qwen2.5-VL-3B-Instruct'))
    ap.add_argument("--layer_path", required=False, help="Hooked layer path; if omitted, read from results JSON")
    ap.add_argument("--mode", choices=["sequence", "token", "whole_concept"], default="sequence")
    ap.add_argument("--insertion", action="store_true", help="Use c-insertion instead of c-deletion")
    ap.add_argument("--rank", type=int, default=1, help="Concept rank to evaluate (1 = top)")
    # Interpret --num_points as percentage of the gradient vector length to traverse (0-100)
    ap.add_argument("--num_points", type=float, default=100.0, help="Percentage of coordinates to traverse (0-100). 100 = full vector")
    # New: curve resolution along that percentage range
    ap.add_argument("--curve_points", type=int, default=64, help="Number of evaluation samples along the selected percentage range")
    ap.add_argument("--device", default=None, help="cuda, cpu, or cuda:N")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out_dir", default=os.environ.get('OUTPUT_DIR', '.'))
    # New: smoothing fraction for zeroing top-|grad| before ranking
    ap.add_argument("--grad_top_zero_frac", type=float, default=0.0, help="Fraction of top-|grad| coordinates to zero before ranking (smoothing)")
    ap.add_argument("--order_mode", choices=["value", "gradient", "random"], default="value",
                    help="'value' (default): rank coordinates by |vec| magnitude alone, no gradient. "
                         "'gradient': rank by |grad*vec| importance (available, no longer the default -- "
                         "gradient-based ranking reweights coordinates by local logit sensitivity, which "
                         "can diverge a lot from the concept vector's own largest components). "
                         "'random': random permutation baseline (chance-level faithfulness), "
                         "output files get a '_random' suffix so they sit alongside the real curves.")
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
