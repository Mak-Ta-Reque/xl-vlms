#!/usr/bin/env python3
import os, json, re, ast
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np
from bert_score import score as bert_score






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
           

            # NOTE: keep ALL text_grounding labels (your old code effectively did this via set)
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

    Efficient batching:
      for this sample+K, flatten only within sample,
      score once, then reduce max per GT.
    """
    n = len(gt_words)
    if n == 0:
        return 0.0

    # fallback lexical

    preds_flat, gts_flat, group_sizes = [], [], []
    for gt, pred_k in zip(gt_words, pred_text_list):
        preds = list(pred_k.get(k, []))
        group_sizes.append(len(preds))
        if preds:
            preds_flat.extend(preds)
            gts_flat.extend([gt] * len(preds))

    if not preds_flat:
        return 0.0

    _, _, F1 = bert_score(preds_flat, gts_flat, lang="en", verbose=True, device=os.environ.get("BERT_DEVICE", os.environ.get("DEVICE", "cuda")))
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
    """
    For each sample:
        score_k(sample) = mean over GT tokens of max BERTScore per GT
    Then dataset:
        mean/std over samples.
    """
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
        final[k] = [float(np.mean(vals)) if vals else 0.0,
                    float(np.std(vals)) if vals else 0.0]
    return final

# ============================================================
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

    # CPU extraction
    extracted = extract_prediction_and_explantion_data(str(EXPLANATIONS_JSON), ks=ks)
    bert_sum = compute_bertscore_per_sample(extracted, ks=ks)

    print("BERTScore between GT tokens and predicted concepts (text):")
    for k, (mean, std) in bert_sum.items():
        print(f"  Top-{k}: BERTScore F1 = {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()
