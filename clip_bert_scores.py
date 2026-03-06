#!/usr/bin/env python3
import os, json, re, ast
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np
from bert_score import score as bert_score

# ---- NEW: CLIP imports ----
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# 1) Token -> word alignment
# ============================================================
def normalize_piece(p: str) -> str:
    p = (p or "").lower()
    for pref in ("##", "▁", "Ġ"):
        if p.startswith(pref):
            p = p[len(pref):]
    return p


def build_gt_token_mod(gt_word, gt_token, sim_thresh=0.8):
    """
    Align token pieces to source words.
    Returns list same length as gt_token where each entry is the word.
    """
    words = [w.lower() for w in gt_word]
    tokens = [normalize_piece(t) for t in gt_token]

    out, i, j = [], 0, 0
    while j < len(tokens):
        if i >= len(words):
            out.append(gt_word[-1])
            j += 1
            continue

        target = words[i]
        buf = ""
        start_j = j

        while j < len(tokens) and len(buf) <= len(target):
            buf += tokens[j]
            j += 1

            if buf == target:
                out.extend([gt_word[i]] * (j - start_j))
                i += 1
                break

            if len(buf) >= len(target):
                sim = SequenceMatcher(None, buf, target).ratio()
                if sim >= sim_thresh:
                    out.extend([gt_word[i]] * (j - start_j))
                    i += 1
                    break
        else:
            out.append(gt_word[i])
            j = start_j + 1

    return out

# ============================================================
# 2) Text helpers
# ============================================================
_PUNCT_RE = re.compile(r"[\.\,\;\:\!\?\(\)\[\]\{\}\"\']")
_SPACE_RE = re.compile(r"\s+")

def _normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s)
    return s

def split_text(s: str):
    s = _normalize_text(s)
    return s.split() if s else []

def lexical_sim(a: str, b: str) -> float:
    a, b = _normalize_text(a), _normalize_text(b)
    if not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb))

def _to_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return v if isinstance(v, list) else [x]
        except Exception:
            return [x]
    if isinstance(x, list):
        return x
    return [x]

# ============================================================
# 3) Predictions extraction (Top-K)
# ============================================================
def get_top_k_predictions(prediction, k=3):
    prediction_k = prediction[:k]
    top_k_tg, top_k_images, top_k_bboxes = [], [], []
    for ranked_pred in prediction_k:
        tg = _to_list(ranked_pred.get("text_grounding", []))
        img_grounding = _to_list(ranked_pred.get("image_grounding_path", []))
        image_bboxes = ranked_pred.get("image_grounding_bboxes", [])
        top_k_tg.append(tg)
        top_k_images.append(img_grounding)
        top_k_bboxes.append(image_bboxes)
    return top_k_tg, top_k_images, top_k_bboxes


def extract_gt_and_prediction(result, ks=(1,2,3)):
    gt_words = split_text(result.get("model_output", ""))
    gt_image = result.get("image_path", "")

    per_tokens = result.get("per_token_concepts", []) or []
    token_text_ranked = [t.get("token_text", "") for t in per_tokens]
    gt_token_word_map = build_gt_token_mod(gt_words, token_text_ranked)

    top_concepts_per_token = [t.get("top_concepts", []) for t in per_tokens]

    predictions_text_all_tokens = []
    prediction_image_all_tokens = []
    prediction_bboxes_all_tokens = []
    for gt_word, pred_list in zip(gt_token_word_map, top_concepts_per_token):
        pred_text_k = {}
        pred_img_k = {}
        pred_bbox_k = {}
        for k in ks:
            top_k_tg, top_k_imgs, top_k_bboxes = get_top_k_predictions(pred_list, k=k)

            flat_tg = set(
                _normalize_text(item)
                for sub in top_k_tg
                for item in _to_list(sub)
                if _normalize_text(item)
            )

            flat_imgs = [
                item
                for sub in top_k_imgs
                for item in _to_list(sub)
                if item
            ]

            flat_bboxes = [
                item
                for sub in top_k_bboxes
                for item in _to_list(sub)
                if item
            ]

            pred_text_k[k] = flat_tg
            pred_img_k[k] = flat_imgs
            pred_bbox_k[k] = flat_bboxes

        predictions_text_all_tokens.append(pred_text_k)
        prediction_image_all_tokens.append(pred_img_k)
        prediction_bboxes_all_tokens.append(pred_bbox_k)

    return gt_image, gt_token_word_map, predictions_text_all_tokens, prediction_image_all_tokens, prediction_bboxes_all_tokens


def extract_prediction_and_explantion_data(json_path, ks=(1,2,3)):
    with open(json_path, "r") as f:
        data = json.load(f)

    results = data["results"] if isinstance(data, dict) and "results" in data else data
    new_result = []

    for r in results:
        gt_image, gt_map, pred_text, pred_img, pred_bbox = extract_gt_and_prediction(r, ks=ks)
        new_result.append({
            "gt_image": gt_image,
            "gt_token_token_word_map": gt_map,
            "predictions_text": pred_text,
            "prediction_image": pred_img,
            "prediction_bboxes": pred_bbox
        })

    return new_result

# ============================================================
# 4) FAST per-sample BERTScore
# your exact formula:
#   max over preds per GT token, then mean over GT tokens
# ============================================================
def sample_bertscore_per_k(gt_words, pred_text_list, k):
    """
    Compute:
      mean_i  max_j  BERTScore(gt_i, pred_{i,j}^topk)
    """
    n = len(gt_words)
    if n == 0:
        return 0.0

    preds_flat, gts_flat, group_sizes = [], [], []
    for gt, pred_k in zip(gt_words, pred_text_list):
        preds = list(pred_k.get(k, []))
        group_sizes.append(len(preds))
        if preds:
            preds_flat.extend(preds)
            gts_flat.extend([gt] * len(preds))

    if not preds_flat:
        return 0.0

    _, _, F1 = bert_score(
        preds_flat,
        gts_flat,
        lang="en",
        verbose=False,
        device=os.environ.get("BERT_DEVICE", os.environ.get("DEVICE", "auto")),
    )
    f1_vals = np.asarray([float(x) for x in F1], dtype=np.float32)

    # max per GT group
    idx = 0
    max_per_gt = []
    for sz in group_sizes:
        if sz == 0:
            max_per_gt.append(0.0)
        else:
            max_per_gt.append(float(f1_vals[idx:idx+sz].max()))
            idx += sz

    return float(np.mean(max_per_gt)) if max_per_gt else 0.0


def compute_bertscore_per_sample(results, ks=(1,2,3),):
    per_sample_scores = {k: [] for k in ks}

    for r in results:
        gt_words = r.get("gt_token_token_word_map", [])
        pred_text_list = r.get("predictions_text", [])

        for k in ks:
            s_k = sample_bertscore_per_k(gt_words, pred_text_list, k)
            per_sample_scores[k].append(s_k)

    final = {}
    for k in ks:
        vals = per_sample_scores[k]
        final[k] = [
            float(np.mean(vals)) if vals else 0.0,
            float(np.std(vals)) if vals else 0.0
        ]
    return final

# ============================================================
# 4b) CLIPScore utilities (FOLLOWING YOUR TECHNIQUE)
# ============================================================
import warnings
from packaging import version
import tqdm
import clip  # pip install git+https://github.com/openai/CLIP.git

# global clip cache
_CLIP_MODEL = None
_CLIP_PREPROCESS = None

def load_clip_once(model_name="ViT-B/32", device="cuda"):
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is None or _CLIP_PREPROCESS is None:
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load(model_name, device=device, jit=False)
        _CLIP_MODEL.eval()
    return _CLIP_MODEL, _CLIP_PREPROCESS


class CLIPImageDataset(torch.utils.data.Dataset):
    """
    Accepts:
      - list of image file paths (str)
      - or list of PIL.Image
    """
    def __init__(self, images, preprocess):
        self.images = images
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        if isinstance(x, Image.Image):
            img = x.convert("RGB")
        else:
            img = Image.open(x).convert("RGB")
        return {"image": self.preprocess(img)}


def extract_image_features(images, bounding_boxes, model, device, preprocess, batch_size=64):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data, desc="CLIP image feats"):
            b = b["image"].to(device)
            if device == "cuda":
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def extract_text_features(candidates, model, device, batch_size=256):
    """
    candidates: list[str]
    returns NxD numpy array
    """
    all_text_features = []
    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            chunk = candidates[i:i+batch_size]
            tok = clip.tokenize(chunk, truncate=True).to(device)
            feats = model.encode_text(tok).cpu().numpy()
            all_text_features.append(feats)
    return np.vstack(all_text_features)


def get_clip_score(model, images, candidates, device, preprocess, w=2.5):
    """
    Your exact CLIPScore.
    images:
      - list of paths or PIL imgs
      - OR precomputed numpy feats (NxD)
    candidates:
      - list of strings OR precomputed feats (NxD)
    Returns:
      mean_score, per_pair_scores, text_features
    """
    if isinstance(images, list):

        images = extract_image_features(images, model, device, preprocess)

    if isinstance(candidates, list):
        candidates = extract_text_features(candidates, model, device)

    # normalize (your technique)
    if version.parse(np.__version__) < version.parse("1.21"):
        # avoid sklearn dep; numpy normalization ok for old versions
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))
    else:
        warnings.warn(
            "New numpy normalization differs slightly from paper results. "
            "For exact replication, use numpy<1.21."
        )
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return float(np.mean(per)), per.astype(np.float32), candidates


# ============================================================
# 4c) CLIPScore per-sample (same reduction as BERTScore)
# ============================================================
def sample_clipscore_per_k(gt_words, pred_img_list, pred_bbox_list, k, device=None):
    """
    mean_i max_j CLIP(g_i, image_{i,j}^topk)

    pred_img_list: list[dict] aligned with gt_words
      each dict: k -> list of image paths (or PIL crops)

    pred_bbox_list is ignored here because we assume you already used it
    to crop upstream if you want crops.
    """
    n = len(gt_words)
    if n == 0:
        return 0.0

    if device is None:
        device = os.environ.get("CLIP_DEVICE", os.environ.get("DEVICE", "auto"))

    model, preprocess = load_clip_once(device=device)

    texts_flat, images_flat, bounding_box_flat, group_sizes = [], [], []

    for gt, pred_k_imgs, pred_k_bboxes in zip(gt_words, pred_img_list, pred_bbox_list):
        imgs = list(pred_k_imgs.get(k, []))  # list of paths or PIL
        group_sizes.append(len(imgs))
        if imgs:
            images_flat.extend(imgs)
            texts_flat.extend([gt] * len(imgs))
            bounding_box_flat.extend(pred_k_bboxes.get(k, []))   

    if not images_flat:
        return 0.0

    # aligned per-pair CLIPScore (your function)
    _, per_pair, _ = get_clip_score(
        model=model,
        images=images_flat,
        bounding_box_flat=bounding_box_flat,
        candidates=texts_flat,
        device=device,
        preprocess=preprocess,
        w=2.5,
    )

    # max per GT group
    idx = 0
    max_per_gt = []
    for sz in group_sizes:
        if sz == 0:
            max_per_gt.append(0.0)
        else:
            max_per_gt.append(float(per_pair[idx:idx+sz].max()))
            idx += sz

    return float(np.mean(max_per_gt)) if max_per_gt else 0.0


def compute_clipscore_per_sample(results, ks=(1,2,3)):
    per_sample_scores = {k: [] for k in ks}

    for r in results:
        gt_words = r.get("gt_token_token_word_map", [])
        pred_img_list = r.get("prediction_image", [])
        pred_bbox_list = r.get("prediction_bboxes", [])

        for k in ks:
            s_k = sample_clipscore_per_k(gt_words, pred_img_list, pred_bbox_list, k)
            per_sample_scores[k].append(s_k)

    final = {}
    for k in ks:
        vals = per_sample_scores[k]
        final[k] = [
            float(np.mean(vals)) if vals else 0.0,
            float(np.std(vals)) if vals else 0.0
        ]
    return final

#===============================
# 5) Main
# ============================================================
def main():
    ks = (1,2,3)

    ROOT_DIR = Path(os.environ.get("ROOT_DIR", Path.cwd()))
    DEFAULT_OUTPUT = ROOT_DIR / "outputs/qwen2_5_10cls_sam/imnet100"
    OUTPUT_DIR_BASE = Path(os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT))
    DECOMP_METHOD = os.environ.get("DECOMP_METHOD", "snmf")

    EXPLANATIONS_JSON = OUTPUT_DIR_BASE / "explanations" / DECOMP_METHOD / "vlm_explanations.json"
    if not EXPLANATIONS_JSON.exists():
        raise FileNotFoundError(f"Missing explanations at {EXPLANATIONS_JSON}")

    extracted = extract_prediction_and_explantion_data(str(EXPLANATIONS_JSON), ks=ks)

    bert_sum = compute_bertscore_per_sample(extracted, ks=ks)
    clip_sum = compute_clipscore_per_sample(extracted, ks=ks)

    print("BERTScore between GT tokens and predicted concepts (text):")
    for k, (mean, std) in bert_sum.items():
        print(f"  Top-{k}: BERTScore F1 = {mean:.4f} ± {std:.4f}")

    print("\nCLIPScore between GT tokens and predicted concept images:")
    for k, (mean, std) in clip_sum.items():
        print(f"  Top-{k}: CLIP cosine = {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
