#!/usr/bin/env python3
import argparse
import ast
import json
import os
import random
import re
import warnings
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from bert_score import score as bert_score
from packaging import version
from PIL import Image
from transformers import logging as hf_logging

import clip


hf_logging.set_verbosity_error()


_PUNCT_RE = re.compile(r"[\.,;:!?()\[\]{}\"']")
_SPACE_RE = re.compile(r"\s+")

_CLIP_MODEL = None
_CLIP_PREPROCESS = None


def normalize_piece(piece: str) -> str:
    piece = (piece or "").lower()
    for prefix in ("##", "▁", "Ġ"):
        if piece.startswith(prefix):
            piece = piece[len(prefix):]
    return piece


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text


def _split_text(text: str) -> List[str]:
    text = _normalize_text(text)
    return text.split() if text else []


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [value]
        except Exception:
            return [value]
    if isinstance(value, list):
        return value
    return [value]


def build_gt_token_map(gt_words: List[str], gt_tokens: List[str], sim_thresh: float = 0.8) -> List[str]:
    if not gt_words:
        return [""] * len(gt_tokens)

    words = [w.lower() for w in gt_words]
    tokens = [normalize_piece(t) for t in gt_tokens]

    out: List[str] = []
    i = 0
    j = 0
    while j < len(tokens):
        if i >= len(words):
            out.append(gt_words[-1])
            j += 1
            continue

        target = words[i]
        buf = ""
        start_j = j

        while j < len(tokens) and len(buf) <= len(target):
            buf += tokens[j]
            j += 1

            if buf == target:
                out.extend([gt_words[i]] * (j - start_j))
                i += 1
                break

            if len(buf) >= len(target):
                sim = SequenceMatcher(None, buf, target).ratio()
                if sim >= sim_thresh:
                    out.extend([gt_words[i]] * (j - start_j))
                    i += 1
                    break
        else:
            out.append(gt_words[i])
            j = start_j + 1

    return out


def _parse_bbox(bbox: Any) -> Tuple[float, float, float, float] | None:
    if bbox is None:
        return None

    if isinstance(bbox, dict):
        if all(k in bbox for k in ("xmin", "ymin", "xmax", "ymax")):
            return float(bbox["xmin"]), float(bbox["ymin"]), float(bbox["xmax"]), float(bbox["ymax"])
        if all(k in bbox for k in ("x", "y", "w", "h")):
            x, y, w, h = float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])
            return x, y, x + w, y + h

    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = map(float, bbox)
        if x2 <= x1 or y2 <= y1:
            return x1, y1, x1 + x2, y1 + y2
        return x1, y1, x2, y2

    return None


def _safe_open_image(path: str) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _crop_with_bbox(img: Image.Image, bbox: Any) -> Image.Image:
    parsed = _parse_bbox(bbox)
    if parsed is None:
        return img

    x1, y1, x2, y2 = parsed
    width, height = img.size

    x1 = max(0, min(width - 1, x1))
    x2 = max(1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(1, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, images: List[Any], preprocess: Any) -> None:
        self.images = images
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self.images[idx]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
        return {"image": self.preprocess(image)}


def _load_clip_once(model_name: str = "ViT-B/32", device: str = "cuda") -> Tuple[Any, Any]:
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is None or _CLIP_PREPROCESS is None:
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load(model_name, device=device, jit=False)
        _CLIP_MODEL.eval()
    return _CLIP_MODEL, _CLIP_PREPROCESS


def _extract_image_features(images: List[Any], model: Any, device: str, preprocess: Any, batch_size: int = 64) -> np.ndarray:
    loader = torch.utils.data.DataLoader(
        CLIPImageDataset(images, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    image_features = []
    with torch.no_grad():
        for batch in loader:
            image_tensor = batch["image"].to(device)
            if device == "cuda":
                image_tensor = image_tensor.to(torch.float16)
            image_features.append(model.encode_image(image_tensor).cpu().numpy())
    return np.vstack(image_features)


def _extract_text_features(candidates: List[str], model: Any, device: str, batch_size: int = 256) -> np.ndarray:
    text_features = []
    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            chunk = candidates[i : i + batch_size]
            tokenized = clip.tokenize(chunk, truncate=True).to(device)
            feats = model.encode_text(tokenized).cpu().numpy()
            text_features.append(feats)
    return np.vstack(text_features)


def _get_clip_per_pair(model: Any, images: List[Any], candidates: List[str], device: str, preprocess: Any, w: float = 2.5) -> np.ndarray:
    image_features = _extract_image_features(images, model, device, preprocess)
    text_features = _extract_text_features(candidates, model, device)

    if version.parse(np.__version__) >= version.parse("1.21"):
        warnings.warn(
            "New numpy normalization differs slightly from paper results. For exact replication, use numpy<1.21."
        )

    image_features = image_features / np.sqrt(np.sum(image_features ** 2, axis=1, keepdims=True))
    text_features = text_features / np.sqrt(np.sum(text_features ** 2, axis=1, keepdims=True))

    return (w * np.clip(np.sum(image_features * text_features, axis=1), 0, None)).astype(np.float32)


def _extract_topk_predictions(result: Dict[str, Any], ks: Tuple[int, ...]) -> Dict[str, Any]:
    gt_words = _split_text(result.get("model_output", ""))
    per_tokens = result.get("per_token_concepts", []) or []
    token_text_ranked = [t.get("token_text", "") for t in per_tokens]
    gt_token_word_map = build_gt_token_map(gt_words, token_text_ranked)

    top_concepts_per_token = [t.get("top_concepts", []) for t in per_tokens]

    predictions_text = []
    predictions_images = []
    predictions_bboxes = []

    for pred_list in top_concepts_per_token:
        pred_text_k: Dict[int, List[str]] = {}
        pred_img_k: Dict[int, List[str]] = {}
        pred_bbox_k: Dict[int, List[Any]] = {}

        for k in ks:
            pred_k = pred_list[:k]
            text_items: List[str] = []
            image_items: List[str] = []
            bbox_items: List[Any] = []

            for item in pred_k:
                for tg in _to_list(item.get("text_grounding", [])):
                    norm = _normalize_text(str(tg))
                    if norm:
                        text_items.append(norm)

                img_list = _to_list(item.get("image_grounding_path", []))
                box_list = _to_list(item.get("image_grounding_bboxes", []))

                if len(box_list) == len(img_list):
                    for img_path, bbox in zip(img_list, box_list):
                        if img_path:
                            image_items.append(img_path)
                            bbox_items.append(bbox)
                elif len(box_list) == 1 and len(img_list) > 1:
                    for img_path in img_list:
                        if img_path:
                            image_items.append(img_path)
                            bbox_items.append(box_list[0])
                else:
                    for img_path in img_list:
                        if img_path:
                            image_items.append(img_path)
                            bbox_items.append(None)

            pred_text_k[k] = sorted(set(text_items))
            pred_img_k[k] = image_items
            pred_bbox_k[k] = bbox_items

        predictions_text.append(pred_text_k)
        predictions_images.append(pred_img_k)
        predictions_bboxes.append(pred_bbox_k)

    return {
        "gt_token_word_map": gt_token_word_map,
        "predictions_text": predictions_text,
        "prediction_image": predictions_images,
        "prediction_bboxes": predictions_bboxes,
    }


def _build_concept_bank(concept_path: Path) -> List[Dict[str, Any]]:
    concept_data = torch.load(str(concept_path), map_location="cpu")
    text_grounding = concept_data.get("text_grounding", [])
    image_paths = concept_data.get("image_grounding_paths", [])
    image_bboxes = concept_data.get("image_grounding_bboxes", [])

    num_concepts = max(
        len(text_grounding) if isinstance(text_grounding, list) else 0,
        len(image_paths) if isinstance(image_paths, list) else 0,
    )

    bank: List[Dict[str, Any]] = []
    for i in range(num_concepts):
        tg = text_grounding[i] if i < len(text_grounding) else []
        ip = image_paths[i] if i < len(image_paths) else []
        ib = image_bboxes[i] if i < len(image_bboxes) else []
        bank.append(
            {
                "text_grounding": _to_list(tg),
                "image_grounding_paths": _to_list(ip),
                "image_grounding_bboxes": _to_list(ib),
            }
        )
    return bank


def _sample_indices(num_concepts: int, k: int, rng: random.Random) -> List[int]:
    if num_concepts <= 0 or k <= 0:
        return []
    if k <= num_concepts:
        return rng.sample(range(num_concepts), k)
    picked = list(range(num_concepts))
    while len(picked) < k:
        picked.append(rng.randrange(num_concepts))
    return picked


def _build_random_predictions(gt_token_word_map: List[str], concept_bank: List[Dict[str, Any]], ks: Tuple[int, ...], rng: random.Random) -> Dict[str, Any]:
    max_k = max(ks)
    num_concepts = len(concept_bank)

    predictions_text = []
    predictions_images = []
    predictions_bboxes = []

    for _ in gt_token_word_map:
        sampled = _sample_indices(num_concepts=num_concepts, k=max_k, rng=rng)

        pred_text_k: Dict[int, List[str]] = {}
        pred_img_k: Dict[int, List[str]] = {}
        pred_bbox_k: Dict[int, List[Any]] = {}

        for k in ks:
            chosen = sampled[:k]
            text_items: List[str] = []
            image_items: List[str] = []
            bbox_items: List[Any] = []

            for ci in chosen:
                concept_item = concept_bank[ci]
                for tg in concept_item["text_grounding"]:
                    norm = _normalize_text(str(tg))
                    if norm:
                        text_items.append(norm)

                imgs = concept_item["image_grounding_paths"]
                bboxes = concept_item["image_grounding_bboxes"]

                if len(bboxes) == len(imgs):
                    for img_path, bbox in zip(imgs, bboxes):
                        if img_path:
                            image_items.append(img_path)
                            bbox_items.append(bbox)
                elif len(bboxes) == 1 and len(imgs) > 1:
                    for img_path in imgs:
                        if img_path:
                            image_items.append(img_path)
                            bbox_items.append(bboxes[0])
                else:
                    for img_path in imgs:
                        if img_path:
                            image_items.append(img_path)
                            bbox_items.append(None)

            pred_text_k[k] = sorted(set(text_items))
            pred_img_k[k] = image_items
            pred_bbox_k[k] = bbox_items

        predictions_text.append(pred_text_k)
        predictions_images.append(pred_img_k)
        predictions_bboxes.append(pred_bbox_k)

    return {
        "gt_token_word_map": gt_token_word_map,
        "predictions_text": predictions_text,
        "prediction_image": predictions_images,
        "prediction_bboxes": predictions_bboxes,
    }


def _sample_bertscore_per_k(gt_words: List[str], pred_text_list: List[Dict[int, List[str]]], k: int, bert_device: str) -> float:
    if not gt_words:
        return 0.0

    preds_flat: List[str] = []
    gts_flat: List[str] = []
    group_sizes: List[int] = []

    for gt, pred_k in zip(gt_words, pred_text_list):
        preds = list(pred_k.get(k, []))
        group_sizes.append(len(preds))
        if preds:
            preds_flat.extend(preds)
            gts_flat.extend([gt] * len(preds))

    if not preds_flat:
        return 0.0

    _, _, f1 = bert_score(preds_flat, gts_flat, lang="en", verbose=False, device=bert_device)
    f1_vals = np.asarray([float(x) for x in f1], dtype=np.float32)

    idx = 0
    mean_per_gt = []
    for sz in group_sizes:
        if sz == 0:
            mean_per_gt.append(0.0)
        else:
            mean_per_gt.append(float(f1_vals[idx : idx + sz].mean()))
            idx += sz

    return float(np.mean(mean_per_gt)) if mean_per_gt else 0.0


def _sample_clipscore_per_k(gt_words: List[str], pred_img_list: List[Dict[int, List[str]]], pred_bbox_list: List[Dict[int, List[Any]]], k: int, clip_device: str) -> float:
    if not gt_words:
        return 0.0

    model, preprocess = _load_clip_once(device=clip_device)

    texts_flat: List[str] = []
    crops_flat: List[Image.Image] = []
    group_sizes: List[int] = []

    for gt, pred_imgs_k, pred_bboxes_k in zip(gt_words, pred_img_list, pred_bbox_list):
        imgs = list(pred_imgs_k.get(k, []))
        bboxes = list(pred_bboxes_k.get(k, [])) if isinstance(pred_bboxes_k, dict) else []

        crops_for_gt: List[Image.Image] = []
        if imgs:
            if len(bboxes) == len(imgs):
                pairs = zip(imgs, bboxes)
            elif len(bboxes) == 1 and len(imgs) > 1:
                pairs = ((img_path, bboxes[0]) for img_path in imgs)
            else:
                pairs = ((img_path, None) for img_path in imgs)

            for img_path, bbox in pairs:
                image = _safe_open_image(img_path)
                if image is None:
                    continue
                crops_for_gt.append(_crop_with_bbox(image, bbox))

        group_sizes.append(len(crops_for_gt))
        if crops_for_gt:
            crops_flat.extend(crops_for_gt)
            texts_flat.extend([gt] * len(crops_for_gt))

    if not crops_flat:
        return 0.0

    per_pair = _get_clip_per_pair(
        model=model,
        images=crops_flat,
        candidates=texts_flat,
        device=clip_device,
        preprocess=preprocess,
        w=2.5,
    )

    idx = 0
    mean_per_gt = []
    for sz in group_sizes:
        if sz == 0:
            mean_per_gt.append(0.0)
        else:
            mean_per_gt.append(float(per_pair[idx : idx + sz].mean()))
            idx += sz

    return float(np.mean(mean_per_gt)) if mean_per_gt else 0.0


def _compute_metric_summary(records: List[Dict[str, Any]], ks: Tuple[int, ...], bert_device: str, clip_device: str) -> Dict[int, Dict[str, float]]:
    bert_per_sample = {k: [] for k in ks}
    clip_per_sample = {k: [] for k in ks}

    for record in records:
        gt_words = record["gt_token_word_map"]
        pred_text_list = record["predictions_text"]
        pred_img_list = record["prediction_image"]
        pred_bbox_list = record["prediction_bboxes"]

        for k in ks:
            bert_per_sample[k].append(_sample_bertscore_per_k(gt_words, pred_text_list, k, bert_device))
            clip_per_sample[k].append(_sample_clipscore_per_k(gt_words, pred_img_list, pred_bbox_list, k, clip_device))

    summary: Dict[int, Dict[str, float]] = {}
    for k in ks:
        bvals = bert_per_sample[k]
        cvals = clip_per_sample[k]
        summary[k] = {
            "bert_mean": float(np.mean(bvals)) if bvals else 0.0,
            "bert_std": float(np.std(bvals)) if bvals else 0.0,
            "clip_mean": float(np.mean(cvals)) if cvals else 0.0,
            "clip_std": float(np.std(cvals)) if cvals else 0.0,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BERT/CLIP top-k scores and random activation baseline.")
    parser.add_argument("--json_path", required=True, help="Path to vlm_explanations.json")
    parser.add_argument("--concept_path", required=True, help="Path to combined_concept_*_raw.pth")
    parser.add_argument("--max_k", type=int, default=5, help="Evaluate k=1..max_k")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for baseline concept sampling")
    parser.add_argument("--out_dir", required=True, help="Output directory for summary table")
    parser.add_argument("--output_prefix", default="clip_bert_topk", help="Output file prefix")
    parser.add_argument("--bert_device", default=os.environ.get("BERT_DEVICE", os.environ.get("DEVICE", "cuda")))
    parser.add_argument("--clip_device", default=os.environ.get("CLIP_DEVICE", os.environ.get("DEVICE", "cuda")))
    args = parser.parse_args()

    explanations_path = Path(args.json_path)
    concept_path = Path(args.concept_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not explanations_path.exists():
        raise FileNotFoundError(f"Missing explanations json: {explanations_path}")
    if not concept_path.exists():
        raise FileNotFoundError(f"Missing concept file: {concept_path}")

    with explanations_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", payload)
    ks = tuple(range(1, max(1, args.max_k) + 1))

    ranked_records: List[Dict[str, Any]] = []
    for result in results:
        ranked_records.append(_extract_topk_predictions(result, ks))

    ranked_summary = _compute_metric_summary(
        records=ranked_records,
        ks=ks,
        bert_device=args.bert_device,
        clip_device=args.clip_device,
    )

    concept_bank = _build_concept_bank(concept_path)
    rng = random.Random(args.seed)
    random_records: List[Dict[str, Any]] = []
    for record in ranked_records:
        random_records.append(
            _build_random_predictions(
                gt_token_word_map=record["gt_token_word_map"],
                concept_bank=concept_bank,
                ks=ks,
                rng=rng,
            )
        )

    random_summary = _compute_metric_summary(
        records=random_records,
        ks=ks,
        bert_device=args.bert_device,
        clip_device=args.clip_device,
    )

    table_rows = []
    for k in ks:
        row = {
            "rank_k": k,
            "bert_mean": ranked_summary[k]["bert_mean"],
            "bert_std": ranked_summary[k]["bert_std"],
            "clip_mean": ranked_summary[k]["clip_mean"],
            "clip_std": ranked_summary[k]["clip_std"],
            "random_bert_mean": random_summary[k]["bert_mean"],
            "random_bert_std": random_summary[k]["bert_std"],
            "random_clip_mean": random_summary[k]["clip_mean"],
            "random_clip_std": random_summary[k]["clip_std"],
        }
        table_rows.append(row)

    json_path = out_dir / f"{args.output_prefix}_table.json"
    csv_path = out_dir / f"{args.output_prefix}_table.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "json_path": str(explanations_path),
                "concept_path": str(concept_path),
                "max_k": args.max_k,
                "seed": args.seed,
                "rows": table_rows,
            },
            f,
            indent=2,
        )

    headers = [
        "rank_k",
        "bert_mean",
        "bert_std",
        "clip_mean",
        "clip_std",
        "random_bert_mean",
        "random_bert_std",
        "random_clip_mean",
        "random_clip_std",
    ]

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in table_rows:
            f.write(
                ",".join(
                    [str(row["rank_k"])]
                    + [f"{row[h]:.6f}" for h in headers[1:]]
                )
                + "\n"
            )

    print("Top-k table:")
    print("k | bert(mean±std) | clip(mean±std) | random_bert(mean±std) | random_clip(mean±std)")
    for row in table_rows:
        print(
            f"{row['rank_k']} | "
            f"{row['bert_mean']:.4f}±{row['bert_std']:.4f} | "
            f"{row['clip_mean']:.4f}±{row['clip_std']:.4f} | "
            f"{row['random_bert_mean']:.4f}±{row['random_bert_std']:.4f} | "
            f"{row['random_clip_mean']:.4f}±{row['random_clip_std']:.4f}"
        )

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
