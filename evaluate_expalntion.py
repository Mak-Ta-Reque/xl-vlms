import os, json, ast, re
from pathlib import Path
import numpy as np
from PIL import Image
from difflib import SequenceMatcher
import ast

# token -> text 

def normalize_piece(p):
    # strip common tokenizer markers
    p = p.lower()
    for pref in ("##", "▁", "Ġ"):
        if p.startswith(pref):
            p = p[len(pref):]
    return p

def build_gt_token_mod(gt_word, gt_token, sim_thresh=0.8):
    words = [w.lower() for w in gt_word]
    tokens = [normalize_piece(t) for t in gt_token]

    out = []
    i = 0  # word pointer
    j = 0  # token pointer

    while j < len(tokens):
        if i >= len(words):
            # if extra tokens remain, just map them to last word
            out.append(gt_word[-1])
            j += 1
            continue

        target = words[i]
        buf = ""
        start_j = j

        # keep adding tokens until we match current word
        while j < len(tokens) and len(buf) <= len(target):
            buf += tokens[j]
            j += 1

            if buf == target:
                # exact match: map this token group to the word
                out.extend([gt_word[i]] * (j - start_j))
                i += 1
                break

            # if buffer is long enough, allow fuzzy stop
            if len(buf) >= len(target):
                sim = SequenceMatcher(None, buf, target).ratio()
                if sim >= sim_thresh:
                    out.extend([gt_word[i]] * (j - start_j))
                    i += 1
                    break
        else:
            # fallback if we exit the inner loop weirdly
            out.append(gt_word[i])
            j = start_j + 1

    return out


# ---------------------------
# Text helpers
# ---------------------------
def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\.\,\;\:\!\?\(\)\[\]\{\}\"\']", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def split_text(s: str):
    s = _normalize_text(s)
    if not s:
        return []
    return s.split()


def extract_ranked_concepts(token_dict, max_concepts=None):
    """
    token_dict: one entry from per_token_concepts
    returns list[str] ranked by your pipeline similarity
    """
    ranked = token_dict.get("top_concepts", [])
    if max_concepts is not None:
        ranked = ranked[:max_concepts]

    out = []
    for c in ranked:
        tg = c.get("text_grounding", [])
        if isinstance(tg, str):
            # sometimes stored as string repr of list
            try:
                tg = ast.literal_eval(tg)
            except Exception:
                tg = [tg]
        if not tg:
            out.append("")
        else:
            # take first label as representative
            out.append(_normalize_text(tg[0]))
    return out

def lexical_sim(a: str, b: str) -> float:
    """
    light similarity: token overlap ratio.
    Replace with embedding cosine if you want.
    """
    a, b = _normalize_text(a), _normalize_text(b)
    if not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb))

def get_top_k_predictions(prediction, k=3):
    prediction_k = prediction[:k]
    top_k_tg = []
    top_k_images = []
    for ranked_prediction in prediction_k:
        tg = ranked_prediction.get("text_grounding", [])
        img_grounding = ranked_prediction.get("image_grounding_path", [])
        top_k_tg.append(tg)
        top_k_images.append(ast.literal_eval(img_grounding) if isinstance(img_grounding, str) else img_grounding)
    return top_k_tg, top_k_images

# ---------------------------
# Top-K concept recall
# ---------------------------
def extract_gt_and_prediction(result, ks=(1,2,3), sim_fn=None):
    """
    For each GT token, find best matching concept across ALL tokens,
    then check its rank within that token's ranked list.
    Returns recall@K for each K.
    """
    if sim_fn is None:
        sim_fn = lexical_sim

    gt_tokens = split_text(result.get("model_output", ""))
    gt_image = result.get("image_path", "")
    per_tokens = result.get("per_token_concepts", [])
    toke_text_ranked =  [t["token_text"] for t in per_tokens]
    gt_token_token_word_map = build_gt_token_mod(gt_tokens, toke_text_ranked)
    top_concepts_prediction = [t["top_concepts"] for t in per_tokens]
    predictions_text = {}
    prediction_image = {}
    predictions_text_all_tokens = []
    prediction_image_all_tokens = []
    for  gt_token_token_word, prediction in zip(gt_token_token_word_map, top_concepts_prediction):
        for k in ks:
            top_k_tg, top_k_images = get_top_k_predictions(prediction, k=k)
            flat_kg_tg = set([item for sublist in top_k_tg for item in sublist])
            flat_kg_images = set([item for sublist in top_k_images for item in sublist])
            predictions_text[k] = flat_kg_tg
            prediction_image[k] = flat_kg_images
        
        predictions_text_all_tokens.append(predictions_text.copy())
        prediction_image_all_tokens.append(prediction_image.copy())
    return  gt_image, gt_token_token_word_map, predictions_text_all_tokens, prediction_image_all_tokens


# ---------------------------
# BERTScore: top-1 concept text vs prediction tokens
# ---------------------------

from bert_score import score

def clean_bert_similarity_f1(gt, preds, lang="en", model_type=None, agg="max"):
    """
    Compute BERTScore F1 similarity between gt (string) and preds (iterable of strings).

    Args:
        gt (str): ground-truth text.
        preds (iterable): predicted strings (list/set/tuple/dict keys etc.).
        lang (str): language code for BERTScore.
        model_type (str|None): e.g. "roberta-large". If None, uses default for lang.
        agg (str): "max" (default), "mean", or None to return per-pred scores.

    Returns:
        float or dict:
            - if agg is "max" or "mean": returns a single float
            - if agg is None: returns dict {pred: f1}
    """
    # normalize preds into a list of strings
    if preds is None:
        preds = []
    if isinstance(preds, dict):
        preds = list(preds)  # dict keys
    else:
        preds = list(preds)

    # handle empty preds
    if len(preds) == 0:
        return 0.0 if agg in ("max", "mean") else {}

    # compute BERTScore
    kwargs = dict(lang=lang, verbose=False)
    if model_type is not None:
        kwargs["model_type"] = model_type

    P, R, F1 = score(preds, [gt] * len(preds), **kwargs)
    f1_list = [float(x) for x in F1]

    per_pred = dict(zip(preds, f1_list))

    if agg == "max":
        return max(f1_list)
    if agg == "mean":
        return sum(f1_list) / len(f1_list)
    if agg is None:
        return per_pred

    raise ValueError("agg must be 'max', 'mean', or None")


def get_word_wise_bert_score(gt_tokens, pred_tokens, model_type="microsoft/deberta-xlarge-mnli", ks=(1,2,3)):
    k_sim={}
    for k in ks:
        gt_tokens = gt_tokens
        pred_tokens_k = pred_tokens[k]
        k_sim[k]  = clean_bert_similarity_f1(gt_tokens, pred_tokens_k, model_type=model_type, agg="max")
    return k_sim
        



def get_sample_bert_score(r, model_type="microsoft/deberta-xlarge-mnli", ks=(1,2,3)):
    ground_truth_words =r.get("gt_token_token_word_map", [])
    predction_token = r.get("predictions_text", [])
    k_sim = {k:0.0 for k in ks}

    for gt, pred in zip(ground_truth_words, predction_token):
        word_wise_bert_scor = get_word_wise_bert_score(gt, pred, model_type=model_type, ks=ks)
                #accumate scor for all k
        for k in ks:
            k_sim[k] += word_wise_bert_scor[k]
        for k in ks:
            k_sim[k] /= len(ground_truth_words)
    return k_sim

def compute_bertscore_ground_truth_vs_prediction(result, ks=(1,2,3), model_type="microsoft/deberta-xlarge-mnli"):
    """
    Align by token index:
      pred_tokens[i]  vs  top1_concept_text[i]
    Uses bert_score if installed, else falls back to lexical avg.
    """
    summary = {k:[] for k in ks}
    for r in result:
        sample_bert_score =  get_sample_bert_score (r, model_type=model_type, ks=ks)
        for k in ks:
            summary[k].append(sample_bert_score[k])
    final_summary = {}
    for k in ks:
        final_summary[k] = [float(np.mean(summary[k])) if summary[k] else 0.0, float(np.std(summary[k])) if summary[k] else 0.0]
    return final_summary
    

# ---------------------------
# CLIPScore: top-1 concept image vs full model output
# ---------------------------
def _get_first_concept_image_path(concept):
    paths = concept.get("image_grounding_path", [])
    if isinstance(paths, str):
        try:
            paths = ast.literal_eval(paths)
        except Exception:
            paths = [paths]
    if not paths:
        return None

    item = paths[0]
    # stored like "idx@/abs/path.png"
    if isinstance(item, str) and "@" in item:
        _, p = item.split("@", 1)
        return p
    return item if isinstance(item, str) else None


# ---------------------------
# Run evaluation on a JSON file
# ---------------------------
def extract_prediction_and_explantion_data(json_path, ks=(1,2,3)):
    with open(json_path, "r") as f:
        data = json.load(f)

    results = data["results"] if isinstance(data, dict) and "results" in data else data
    new_result = []
    for r in results:
        extract_gt_and_prediction(r, ks=ks)
        gt_image,gt_token_token_word_map, predictions_text, prediction_image = extract_gt_and_prediction(r, ks=ks)
        new_result.append({
            "gt_image": gt_image,
            "gt_token_token_word_map": gt_token_token_word_map,
            "predictions_text": predictions_text,
            "prediction_image": prediction_image
        })

    return new_result



    # Configure paths and summarize explanations
ROOT_DIR = Path(os.environ.get("ROOT_DIR", Path.cwd()))
DEFAULT_OUTPUT = ROOT_DIR / "outputs/qwen2_5_10cls_sam/imnet600"
OUTPUT_DIR_BASE = Path(os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT))
DECOMP_METHOD = os.environ.get("DECOMP_METHOD", "snmf")
EXPLANATIONS_JSON = OUTPUT_DIR_BASE / "explanations" / DECOMP_METHOD / "vlm_explanations.json"

if not EXPLANATIONS_JSON.exists():
    raise FileNotFoundError(f"Missing explanations at {EXPLANATIONS_JSON}")

extracted_ground_truth_prediction = extract_prediction_and_explantion_data(str(EXPLANATIONS_JSON),  ks=(1,2,3))
bert_scor = compute_bertscore_ground_truth_vs_prediction(extracted_ground_truth_prediction,ks=(1,2,3))
print("BERTScore between ground-truth tokens and predicted concepts (text):")
for k in bert_scor:
    mean, std = bert_scor[k]
    print(f"  Top-{k} concepts: BERTScore F1 = {mean:.4f} ± {std:.4f}") 
    recalls = {}

