#!/usr/bin/env python3
import os, json, re, ast, warnings
from pathlib import Path
from difflib import SequenceMatcher
import argparse

import numpy as np
from bert_score import score as bert_score

# ---------------- CLIP deps (inference only) ----------------
import torch
from PIL import Image
import clip  # pip install git+https://github.com/openai/CLIP.git
import tqdm
from packaging import version


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
#   mean_i max_j BERTScore(gt_i, pred_{i,j}^topk)
# ============================================================
def sample_bertscore_per_k(gt_words, pred_text_list, k):
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
        preds_flat, gts_flat, lang="en",
        verbose=False,
        device=os.environ.get("BERT_DEVICE", os.environ.get("DEVICE", "cuda"))
    )
    f1_vals = np.asarray([float(x) for x in F1], dtype=np.float32)

    idx = 0
    max_per_gt = []
    for sz in group_sizes:
        if sz == 0:
            max_per_gt.append(0.0)
        else:
            max_per_gt.append(float(f1_vals[idx:idx+sz].mean()))
            idx += sz

    return float(np.mean(max_per_gt)) if max_per_gt else 0.0


def compute_bertscore_per_sample(results, ks=(1,2,3)):
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
# 4b) CLIPScore (YOUR TECHNIQUE) + BBOX CROPPING
#   mean_i max_j CLIP(gt_i, crop(image_{i,j}))
# ============================================================
# global CLIP cache
_CLIP_MODEL = None
_CLIP_PREPROCESS = None

def load_clip_once(model_name="ViT-B/32", device="cuda"):
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is None or _CLIP_PREPROCESS is None:
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load(model_name, device=device, jit=False)
        _CLIP_MODEL.eval()
    return _CLIP_MODEL, _CLIP_PREPROCESS


def _safe_open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _parse_bbox(b):
    """
    Supports:
      - [x1, y1, x2, y2]
      - [x, y, w, h]
      - {"xmin","ymin","xmax","ymax"}
      - {"x","y","w","h"}
    Returns (x1,y1,x2,y2) or None.
    """
    if b is None:
        return None

    if isinstance(b, dict):
        if all(k in b for k in ("xmin","ymin","xmax","ymax")):
            return float(b["xmin"]), float(b["ymin"]), float(b["xmax"]), float(b["ymax"])
        if all(k in b for k in ("x","y","w","h")):
            x, y, w, h = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
            return x, y, x+w, y+h

    if isinstance(b, (list, tuple)) and len(b) == 4:
        x1, y1, x2, y2 = map(float, b)
        if x2 <= x1 or y2 <= y1:
            # treat as (x,y,w,h)
            return x1, y1, x1+x2, y1+y2
        return x1, y1, x2, y2

    return None


def _crop_with_bbox(img: Image.Image, bbox):
    b = _parse_bbox(bbox)
    if b is None:
        return img
    x1, y1, x2, y2 = b
    w, h = img.size

    x1 = max(0, min(w-1, x1))
    x2 = max(1, min(w,   x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(1, min(h,   y2))

    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


class CLIPImageDataset(torch.utils.data.Dataset):
    """
    images can be list of:
      - PIL.Image
      - file paths (str)
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


def extract_image_features(images, model, device, preprocess, batch_size=64):
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
    return np.vstack(all_image_features)


def extract_text_features(candidates, model, device, batch_size=256):
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
    Your exact CLIPScore:
      per = w * clip(cos, 0, inf)
    """
    if isinstance(images, list):
        images = extract_image_features(images, model, device, preprocess)

    if isinstance(candidates, list):
        candidates = extract_text_features(candidates, model, device)

    # normalize (your technique)
    if version.parse(np.__version__) < version.parse("1.21"):
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


def sample_clipscore_per_k(gt_words, pred_img_list, pred_bbox_list, k, device=None):
    """
    Compute:
      mean_i max_j CLIP(gt_i, crop(image_{i,j}, bbox_{i,j})^topk)

    If bboxes don't align with imgs:
      - len(bxs)==len(imgs): zip 1:1
      - len(bxs)==1: reuse same bbox for all imgs
      - else: ignore bboxes for that token
    """
    n = len(gt_words)
    if n == 0:
        return 0.0

    if device is None:
        device = os.environ.get("CLIP_DEVICE", os.environ.get("DEVICE", "cuda"))

    model, preprocess = load_clip_once(device=device)

    texts_flat, crops_flat, group_sizes = [], [], []

    for gt, pred_k_imgs, pred_k_bxs in zip(gt_words, pred_img_list, pred_bbox_list):
        imgs = list(pred_k_imgs.get(k, []))
        bxs  = list(pred_k_bxs.get(k, [])) if isinstance(pred_k_bxs, dict) else []

        crops_for_gt = []
        if imgs:
            if len(bxs) == len(imgs):
                pairs = zip(imgs, bxs)
            elif len(bxs) == 1:
                pairs = [(p, bxs[0]) for p in imgs]
            else:
                pairs = [(p, None) for p in imgs]

            for pth, bx in pairs:
                im = _safe_open_image(pth)
                if im is None:
                    continue
                crops_for_gt.append(_crop_with_bbox(im, bx))

        group_sizes.append(len(crops_for_gt))
        if crops_for_gt:
            crops_flat.extend(crops_for_gt)
            texts_flat.extend([gt] * len(crops_for_gt))

    if not crops_flat:
        return 0.0

    _, per_pair, _ = get_clip_score(
        model=model,
        images=crops_flat,      # PIL crops
        candidates=texts_flat,
        device=device,
        preprocess=preprocess,
        w=2.5
    )

    idx = 0
    max_per_gt = []
    for sz in group_sizes:
        if sz == 0:
            max_per_gt.append(0.0)
        else:
            max_per_gt.append(float(per_pair[idx:idx+sz].mean()))
            idx += sz

    return float(np.mean(max_per_gt)) if max_per_gt else 0.0


def compute_clipscore_per_sample(results, ks=(1,2,3)):
    per_sample_scores = {k: [] for k in ks}

    for r in results:
        gt_words = r.get("gt_token_token_word_map", [])
        pred_img_list  = r.get("prediction_image", [])
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


# ============================================================
# 5) Main
# ============================================================

def main():
    # evaluate only alpha_0 and alpha_10 (default), same Top-K as before
    ks = (1, 2, 3)

    # allow evaluating a single JSON directly via CLI or a root directory with many alpha subfolders
    parser = argparse.ArgumentParser(
        description="Evaluate BERTScore and CLIPScore for explanations JSON(s)."
    )
    parser.add_argument("--json_path", type=str, default=None,
                        help="Path to a single explanations JSON file to evaluate directly")
    parser.add_argument("--root_dir", type=str, default=None,
                        help="Root directory containing multiple alpha subfolders to evaluate (overrides env ALPHA_ROOT)")
    parser.add_argument("--alphas", type=str, default=None,
                        help="Comma-separated alphas (if omitted, subfolders under --root_dir will be autodetected)")
    parser.add_argument("--decomp_method", type=str, default=os.environ.get("DECOMP_METHOD", "snmf"),
                        help="Decomposition method subfolder under explanations (default: snmf)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save the CSV summary (default: <root_dir>/n_eval_<decomp_method>.csv)")
    args = parser.parse_args()

    if args.json_path:
        EXPLANATIONS_JSON = Path(args.json_path)
        if not EXPLANATIONS_JSON.exists():
            raise FileNotFoundError(f"Missing explanations at {EXPLANATIONS_JSON}")

        extracted = extract_prediction_and_explantion_data(str(EXPLANATIONS_JSON), ks=ks)

        bert_sum = compute_bertscore_per_sample(extracted, ks=ks)
        clip_sum = compute_clipscore_per_sample(extracted, ks=ks)

        print("BERTScore between GT tokens and predicted concepts (text):")
        for k, (mean, std) in bert_sum.items():
            print(f"  Top-{k}: BERTScore F1 = {mean:.4f} ± {std:.4f}")

        print("\nCLIPScore between GT tokens and predicted concept crops (image):")
        for k, (mean, std) in clip_sum.items():
            print(f"  Top-{k}: CLIPScore = {mean:.4f} ± {std:.4f}")

        return

    # root that contains alpha folders
    ALPHA_ROOT = Path(args.root_dir) if args.root_dir else Path(os.environ.get(
        "ALPHA_ROOT",
        "/mnt/abka03/Projects/xl-vlms/outputs/ablation/dictionary_size"
    ))

    if not ALPHA_ROOT.exists():
        raise FileNotFoundError(f"Missing ALPHA_ROOT at {ALPHA_ROOT}")

    # determine which alphas to evaluate
    if args.alphas:
        alphas = [int(x.strip()) for x in args.alphas.split(",") if x.strip()]
    else:
        # autodetect numeric subfolders (supports 'n_<num>' or '<num>')
        alphas = []
        for p in sorted(ALPHA_ROOT.iterdir()):
            if not p.is_dir():
                continue
            m = re.search(r"(\d+)", p.name)
            if m:
                alphas.append(int(m.group(1)))
        alphas = sorted(set(alphas))

    DECOMP_METHOD = args.decomp_method

    rows = []
    for a in alphas:
        candidate1 = ALPHA_ROOT / f"layer_{a}"
        candidate2 = ALPHA_ROOT / str(a)
        out_dir = candidate1 if candidate1.exists() else candidate2
        if not out_dir.exists():
            print(f"[WARN] missing folder for alpha {a}: tried {candidate1} and {candidate2} -> skip")
            continue

        explanations_json = out_dir / "explanations" / DECOMP_METHOD / "vlm_explanations.json"

        if not explanations_json.exists():
            print(f"[WARN] missing: {explanations_json}  -> skip n_{a}")
            continue

        extracted = extract_prediction_and_explantion_data(str(explanations_json), ks=ks)

        bert_sum = compute_bertscore_per_sample(extracted, ks=ks)
        clip_sum = compute_clipscore_per_sample(extracted, ks=ks)

        row = {"n": a}
        for k in ks:
            b_mean, b_std = bert_sum[k]
            c_mean, c_std = clip_sum[k]
            row[f"bert@{k}_mean"] = b_mean
            row[f"bert@{k}_std"]  = b_std
            row[f"clip@{k}_mean"] = c_mean
            row[f"clip@{k}_std"]  = c_std
        rows.append(row)

    if not rows:
        print("No alpha folders evaluated. Check ALPHA_ROOT / DECOMP_METHOD.")
        return

    # -------- print nice table --------
    headers = ["n"] + [f"BERT@{k}" for k in ks] + [f"CLIP@{k}" for k in ks]
    print("\n" + " | ".join(headers))
    print("-" * (len(" | ".join(headers))))

    for r in rows:
        bert_cells = [
            f"{r[f'bert@{k}_mean']:.4f}±{r[f'bert@{k}_std']:.4f}" for k in ks
        ]
        clip_cells = [
            f"{r[f'clip@{k}_mean']:.4f}±{r[f'clip@{k}_std']:.4f}" for k in ks
        ]
        print(" | ".join([str(r["n"])] + bert_cells + clip_cells))

    # -------- save CSV --------
    csv_path = Path(args.output_csv) if args.output_csv else ALPHA_ROOT / f"n_eval_{DECOMP_METHOD}.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(["n"] +
                         [f"bert@{k}_mean" for k in ks] +
                         [f"bert@{k}_std" for k in ks] +
                         [f"clip@{k}_mean" for k in ks] +
                         [f"clip@{k}_std" for k in ks]) + "\n")
        for r in rows:
            line = [str(r["n"])]
            for k in ks:
                line.append(f"{r[f'bert@{k}_mean']:.6f}")
            for k in ks:
                line.append(f"{r[f'bert@{k}_std']:.6f}")
            for k in ks:
                line.append(f"{r[f'clip@{k}_mean']:.6f}")
            for k in ks:
                line.append(f"{r[f'clip@{k}_std']:.6f}")
            f.write(",".join(line) + "\n")

    print(f"\nSaved table to: {csv_path}")

if __name__ == "__main__":
    main()
