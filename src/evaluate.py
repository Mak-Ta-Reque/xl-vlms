#!/usr/bin/env python3
"""
CLIPScore + BERTScore evaluation.

- Uses OpenCLIP if available, else Transformers CLIP if available.
- If neither CLIP backend is installed, CLIPScore is skipped gracefully.
- Expects a JSON list of items with:
    {
      "prediction": "...",          # model text
      "reference": "...",           # gt text
      "image_path": "/abs/or/rel/path.jpg"  # optional for CLIPScore
    }

You can adapt field names via CLI args.
"""

import argparse, json, ast, os
from typing import List, Dict, Optional

import numpy as np
from PIL import Image

# ---------------------------
# Optional deps: CLIP backends
# ---------------------------
_HAS_OPEN_CLIP = False
_HAS_TF_CLIP = False

try:
    import torch
except ImportError as e:
    raise RuntimeError("This script needs torch. Please `pip install torch`.") from e

try:
    import open_clip
    _HAS_OPEN_CLIP = True
except Exception:
    _HAS_OPEN_CLIP = False

try:
    from transformers import CLIPModel, CLIPProcessor
    _HAS_TF_CLIP = True
except Exception:
    _HAS_TF_CLIP = False


class ClipScorer:
    def __init__(self, device="cuda", openclip_model="ViT-B-32", openclip_pretrained="laion2b_s34b_b79k",
                 hf_model="openai/clip-vit-base-patch32"):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.available = False
        self.backend = None

        if _HAS_OPEN_CLIP:
            self.backend = "open_clip"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                openclip_model, pretrained=openclip_pretrained
            )
            self.tokenizer = open_clip.get_tokenizer(openclip_model)
            self.model.to(self.device).eval()
            self.available = True

        elif _HAS_TF_CLIP:
            self.backend = "transformers"
            self.model = CLIPModel.from_pretrained(hf_model).to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained(hf_model)
            self.available = True

        else:
            print("[WARN] Neither open_clip nor transformers CLIP is installed. "
                  "CLIPScore will be skipped. Install with `pip install open-clip-torch` "
                  "or `pip install transformers`.")

    @torch.no_grad()
    def score(self, image_path: str, text: str) -> float:
        if not self.available:
            return float("nan")

        img = Image.open(image_path).convert("RGB")

        if self.backend == "open_clip":
            image = self.preprocess(img).unsqueeze(0).to(self.device)
            text_tok = self.tokenizer([text]).to(self.device)

            img_feat = self.model.encode_image(image)
            txt_feat = self.model.encode_text(text_tok)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            return float((img_feat * txt_feat).sum().item())

        # transformers backend
        inputs = self.processor(text=[text], images=img, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        img_feat = outputs.image_embeds
        txt_feat = outputs.text_embeds
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        return float((img_feat * txt_feat).sum().item())


def safe_get(d: Dict, key: str, default=None):
    return d[key] if key in d else default


def parse_list_string(x):
    """If x is like "['a','b']", return list; else return x."""
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                return ast.literal_eval(s)
            except Exception:
                return x
    return x


def load_items(pred_json: str) -> List[Dict]:
    with open(pred_json, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        raise ValueError("JSON must be a list (or dict with key 'items').")
    return data


def compute_bertscore(preds: List[str], refs: List[str], lang="en"):
    try:
        from bert_score import score as bert_score
    except ImportError:
        print("[WARN] bert_score not installed. Skipping BERTScore. Install via `pip install bert-score`.")
        return None

    P, R, F1 = bert_score(preds, refs, lang=lang, verbose=False)
    return {
        "bertscore_precision": float(P.mean().item()),
        "bertscore_recall": float(R.mean().item()),
        "bertscore_f1": float(F1.mean().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True, help="Path to JSON predictions.")
    ap.add_argument("--pred_key", default="prediction", help="Key for predicted text.")
    ap.add_argument("--ref_key", default="reference", help="Key for reference/GT text.")
    ap.add_argument("--img_key", default="image_path", help="Key for image path (optional).")
    ap.add_argument("--img_root", default=None,
                    help="If provided, join this root with relative image paths.")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "auto"), help="Device config string.")
    ap.add_argument("--bertscore_lang", default="en")
    ap.add_argument("--no_clip", action="store_true", help="Force skip CLIPScore even if backend exists.")
    args = ap.parse_args()

    items = load_items(args.pred_json)

    preds, refs = [], []
    clip_scores = []

    clip_scorer = ClipScorer(device=args.device)
    if args.no_clip:
        clip_scorer.available = False

    for it in items:
        pred = str(safe_get(it, args.pred_key, "")).strip()
        ref  = str(safe_get(it, args.ref_key, "")).strip()
        preds.append(pred)
        refs.append(ref)

        img_path = safe_get(it, args.img_key, None)
        img_path = parse_list_string(img_path)

        # if image paths are a list, take first (you can customize)
        if isinstance(img_path, list) and len(img_path) > 0:
            img_path = img_path[0]

        if isinstance(img_path, str) and img_path:
            if args.img_root and not os.path.isabs(img_path):
                img_path = os.path.join(args.img_root, img_path)

            if os.path.exists(img_path):
                clip_scores.append(clip_scorer.score(img_path, pred))
            else:
                clip_scores.append(float("nan"))
        else:
            clip_scores.append(float("nan"))

    results = {}

    # BERTScore
    bs = compute_bertscore(preds, refs, lang=args.bertscore_lang)
    if bs:
        results.update(bs)

    # CLIPScore
    clip_arr = np.array(clip_scores, dtype=np.float32)
    valid = np.isfinite(clip_arr)
    if valid.any():
        results["clipscore_mean"] = float(clip_arr[valid].mean())
    else:
        results["clipscore_mean"] = float("nan")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
