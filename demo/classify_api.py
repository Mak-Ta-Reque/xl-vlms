#!/usr/bin/env python3
"""
FastAPI backend for the VLM Classify demo.

Replaces the Streamlit vlm_classify_demo.py with a REST API that exposes:
  POST /api/classify   — upload image → classify → return nouns
  POST /api/ground     — image_id + nouns → bounding boxes
  POST /api/explain    — image_id + label → binary concept scoring + prototypes
  GET  /api/samples    — list of sample image filenames
  GET  /api/health     — health check

Usage:
  conda run -n xlvlm-v1 python -m uvicorn demo.classify_api:app \
    --host 0.0.0.0 --port 8501
"""

from __future__ import annotations

import os
import re
import sys
import json
import uuid
import threading
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent

from dotenv import load_dotenv

load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

_device_str = os.getenv("DEVICE", "auto")
# Legacy support: if DEVICE_ID is set but DEVICE is not, convert
if "DEVICE" not in os.environ and "DEVICE_ID" in os.environ:
    _device_str = f"cuda:{os.environ['DEVICE_ID']}"

sys.path.insert(0, str(PROJECT_ROOT / "inference"))
sys.path.insert(0, str(_THIS_DIR))

import torch
from vis_utils import (
    MASK_COLORS,
    MASK_COLORS_HEX,
    get_num_prototypes,
    pil_to_base64,
    render_prototype,
)

# ── config ────────────────────────────────────────────────────────────────


def _resolve_concept_path() -> Path:
    cp = os.getenv("CONCEPT_PATH")
    if cp:
        return Path(cp)
    output_dir = os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "outputs/noidle"))
    return Path(output_dir) / "concept" / "snmf" / "combined_concept_snmf_raw.pth"


CONCEPT_PTH = _resolve_concept_path()
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
LAYER_PATH = os.getenv("LAYER_PATH", "model.language_model.norm")
IMAGE_SIZE_WIDTH = int(os.getenv("IMAGE_SIZE_WIDTH", "512"))

CLASSIFY_PROMPT = os.getenv(
    "CLASSIFY_PROMPT",
    "Name only the main objects in each part of the image. "
    "Answer with a short comma-separated list of single words, no descriptions.",
)

BBOX_COLORS_HEX: list[str] = [
    "#e67e22",
    "#27ae60",
    "#c0392b",
    "#2980b9",
    "#8e44ad",
]

TEMP_DIR = PROJECT_ROOT / "api" / "temp_images"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_DIR = _THIS_DIR / "samples"

# ── FastAPI app ───────────────────────────────────────────────────────────

app = FastAPI(title="VLM Classify API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve sample images as static files
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")

# ── singleton explainer ──────────────────────────────────────────────────

_explainer = None
_explainer_lock = threading.Lock()


def _get_explainer():
    global _explainer
    if _explainer is None:
        with _explainer_lock:
            if _explainer is None:
                from vlm_explainer_multibatch import VLMConceptExplainer

                _explainer = VLMConceptExplainer(
                    model_name=VLM_MODEL,
                    concept_path=str(CONCEPT_PTH),
                    layer_path=LAYER_PATH,
                    device=_device_str,
                    prompt_mode="unsupervised",
                    normalize_concepts=False,
                    verbose=False,
                    save_only_generated_tokens=True,
                )
    return _explainer


# ── concept predictions cache ────────────────────────────────────────────

_concept_predictions: Optional[List] = None
_concept_predictions_lock = threading.Lock()


def _load_concept_predictions() -> List:
    global _concept_predictions
    if _concept_predictions is None:
        with _concept_predictions_lock:
            if _concept_predictions is None:
                try:
                    data = torch.load(str(CONCEPT_PTH), map_location="cpu")
                    _concept_predictions = data.get(
                        "image_grounding_predictions", []
                    )
                except Exception:
                    _concept_predictions = []
    return _concept_predictions


# ── spaCy singleton ──────────────────────────────────────────────────────

_nlp = None
_nlp_lock = threading.Lock()


def _get_spacy():
    global _nlp
    if _nlp is None:
        with _nlp_lock:
            if _nlp is None:
                import spacy

                try:
                    _nlp = spacy.load("en_core_web_sm")
                except OSError:
                    spacy.cli.download("en_core_web_sm")
                    _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ── helper functions (ported from vlm_classify_demo.py) ──────────────────


def extract_nouns(text: str) -> List[str]:
    """Extract unique noun lemmas from model output."""
    nlp = _get_spacy()
    doc = nlp(text)
    nouns: list[str] = []
    seen: set[str] = set()
    for chunk in doc.noun_chunks:
        lemma = chunk.root.lemma_.lower().strip()
        if lemma and lemma not in seen and chunk.root.pos_ in ("NOUN", "PROPN"):
            nouns.append(lemma)
            seen.add(lemma)
    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN"):
            lemma = tok.lemma_.lower().strip()
            if lemma and lemma not in seen:
                nouns.append(lemma)
                seen.add(lemma)
    return nouns


def _bbox_prompt_for_nouns(nouns: List[str]) -> str:
    noun_list = ", ".join(nouns)
    n = len(nouns)
    return (
        f"For each of these {n} main objects draw a tight bounding box: {noun_list}. "
        f"The image may contain one photo or a grid of photos. "
        f"Return exactly {n} bounding boxes, one per object. "
        f"Output ONLY a JSON array, each element: "
        f'{{"name": "<object>", "bbox": [x_min, y_min, x_max, y_max]}}. '
        f"Coordinates in pixels. No other text."
    )


def _bbox_prompt_single(noun: str) -> str:
    return (
        f"Draw a tight bounding box around the main '{noun}' in the image. "
        f"The image may be a single photo or a grid. "
        f'Return ONLY: [{{"name": "{noun}", "bbox": [x_min, y_min, x_max, y_max]}}] '
        f"with pixel coordinates. No other text."
    )


def _parse_bbox_json(text: str) -> List[Dict[str, Any]]:
    """Parse JSON bboxes from model output, with fallback regex."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_\-]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    blob = None
    if text.startswith("[") and "]" in text:
        blob = text[: text.rfind("]") + 1]
    elif text.startswith("{") and "}" in text:
        blob = text[: text.rfind("}") + 1]
    else:
        m = re.search(r"[\[\{][\s\S]*[\]\}]", text)
        if m:
            blob = m.group(0)

    if blob is None and ("[" in text or "{" in text):
        start = min((text.index(c) for c in "[{" if c in text))
        blob = text[start:]
        open_sq = blob.count("[") - blob.count("]")
        open_br = blob.count("{") - blob.count("}")
        blob += "]" * max(0, open_sq) + "}" * max(0, open_br)

    objects: List[Dict[str, Any]] = []
    if blob:
        try:
            js = json.loads(blob)
            items: list = []
            if isinstance(js, dict):
                items = js.get("objects", [])
            elif isinstance(js, list):
                items = js
            for o in items:
                if not isinstance(o, dict):
                    continue
                name = o.get("name", "")
                bbox = o.get("bbox") or o.get("box")
                if name and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        bb = [float(x) for x in bbox]
                        objects.append({"name": name.lower().strip(), "bbox": bb})
                    except (ValueError, TypeError):
                        pass
        except json.JSONDecodeError:
            pass
    if not objects:
        for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
            name = m.group(1)
            rest = text[m.end() : m.end() + 200]
            bm = re.search(r'"bbox"\s*:\s*\[([^\]]+)\]', rest)
            if bm:
                nums = re.findall(r"-?\d*\.?\d+", bm.group(1))
                if len(nums) >= 4:
                    bb = [float(nums[i]) for i in range(4)]
                    objects.append({"name": name.lower().strip(), "bbox": bb})
    return objects


def _run_grounding(image_path: str, nouns: List[str]) -> List[Dict[str, Any]]:
    """Ask the VLM to return bounding boxes, with per-noun fallback."""
    exp = _get_explainer()
    old_prompt = exp.prompt

    exp.prompt = _bbox_prompt_for_nouns(nouns)
    try:
        results = exp.explain_with_concept(
            images=[image_path],
            top_n=1,
            max_new_tokens=512,
            temperature=0.0,
            batch_size=1,
        )
    finally:
        exp.prompt = old_prompt
    raw_text = results[0].get("model_output", "")
    objects = _parse_bbox_json(raw_text)

    found_nouns = set()
    for obj in objects:
        for noun in nouns:
            if noun in obj["name"] or obj["name"] in noun:
                found_nouns.add(noun)
                break

    missing = [n for n in nouns if n not in found_nouns]

    for noun in missing:
        exp.prompt = _bbox_prompt_single(noun)
        try:
            res = exp.explain_with_concept(
                images=[image_path],
                top_n=1,
                max_new_tokens=300,
                temperature=0.0,
                batch_size=1,
            )
        finally:
            exp.prompt = old_prompt
        extra = _parse_bbox_json(res[0].get("model_output", ""))
        if extra:
            for o in extra:
                o["name"] = noun
            objects.extend(extra[:1])

    return objects


def _classify_image(image_path: str, custom_prompt: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """Classify an image, returning (result_dict, prompt_used)."""
    exp = _get_explainer()
    old_prompt = exp.prompt
    prompt_used = custom_prompt if custom_prompt else CLASSIFY_PROMPT
    exp.prompt = prompt_used
    try:
        results = exp.explain_with_concept(
            images=[image_path],
            top_n=5,
            max_new_tokens=60,
            temperature=0.0,
            batch_size=1,
        )
    finally:
        exp.prompt = old_prompt
    return results[0], prompt_used


def _run_binary_for_class(image_path: str, label: str) -> Dict[str, Any]:
    exp = _get_explainer()
    old_prompt, old_mode, old_label = exp.prompt, exp.prompt_mode, exp.prompt_label
    exp.prompt = None
    exp.prompt_mode = "binary"
    exp.prompt_label = label
    try:
        results = exp.explain_with_concept(
            images=[image_path],
            ground_truth_labels=[label],
            top_n=5,
            max_new_tokens=80,
            temperature=0.0,
            batch_size=1,
        )
    finally:
        exp.prompt, exp.prompt_mode, exp.prompt_label = (
            old_prompt,
            old_mode,
            old_label,
        )
    return results[0]


# ── Pydantic models ──────────────────────────────────────────────────────


class GroundRequest(BaseModel):
    image_id: str
    nouns: List[str]


class ExplainRequest(BaseModel):
    image_id: str
    label: str


class GroundedObject(BaseModel):
    name: str
    bbox: List[float]


class PrototypeInfo(BaseModel):
    image_b64: str


class ConceptInfo(BaseModel):
    rank: int
    similarity: float
    concept_index: int
    concept_names: List[str]
    predictions: List[str]
    prototypes: List[PrototypeInfo]


# ── routes ────────────────────────────────────────────────────────────────


def _resolve_image_path(image_id: str) -> Path:
    """Resolve an image_id to its temp file path (prevent directory traversal)."""
    safe_name = Path(image_id).name
    p = TEMP_DIR / safe_name
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")
    return p


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/samples")
async def list_samples():
    """Return list of available sample image filenames."""
    if not SAMPLES_DIR.exists():
        return {"samples": []}
    files = sorted(
        f.name for f in SAMPLES_DIR.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    return {"samples": files}


@app.post("/api/classify")
async def classify(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    """Upload an image, classify it, return model output + extracted nouns.

    Accepts an optional `prompt` form field to override the default classify prompt.
    """
    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save to temp
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    image_id = f"{uuid.uuid4().hex}{suffix}"
    save_path = TEMP_DIR / image_id
    content = await file.read()
    save_path.write_bytes(content)

    # Classify (with optional custom prompt)
    custom_prompt = prompt.strip() if prompt and prompt.strip() else None
    try:
        result, prompt_used = _classify_image(str(save_path), custom_prompt=custom_prompt)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

    model_output = result.get("model_output", "").strip()
    nouns = extract_nouns(model_output)

    return {
        "image_id": image_id,
        "model_output": model_output,
        "nouns": nouns,
        "prompt": prompt_used,
    }


@app.post("/api/ground")
async def ground(req: GroundRequest):
    """Run grounding to detect bounding boxes for the specified nouns."""
    img_path = _resolve_image_path(req.image_id)

    if not req.nouns:
        return {"objects": []}

    grounding_prompt = _bbox_prompt_for_nouns(req.nouns) if len(req.nouns) > 1 else (
        _bbox_prompt_single(req.nouns[0]) if len(req.nouns) == 1 else ""
    )

    try:
        objects = _run_grounding(str(img_path), req.nouns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grounding failed: {e}")

    return {
        "objects": [GroundedObject(name=o["name"], bbox=o["bbox"]) for o in objects],
        "prompt": grounding_prompt,
    }


@app.post("/api/explain")
async def explain(req: ExplainRequest):
    """Run binary concept scoring for a selected class label.

    Returns top 3 concepts with prototypes rendered as base64.
    """
    img_path = _resolve_image_path(req.image_id)

    try:
        binary_result = _run_binary_for_class(str(img_path), req.label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explain failed: {e}")

    model_output = binary_result.get("model_output", "")
    top_concepts_raw = binary_result.get("top_concepts_over_sequence") or []
    all_preds = _load_concept_predictions()
    concept_dir = CONCEPT_PTH.parent
    PROTO_W, PROTO_H = 140, 100

    concepts: List[dict] = []
    n_show = min(3, len(top_concepts_raw))

    for rank, concept_info in enumerate(top_concepts_raw[:n_show], 1):
        sim = concept_info.get("similarity", 0.0)
        ci = concept_info.get("concept_index")

        # Concept names (dedup)
        raw_cname = concept_info.get("concept_name")
        if isinstance(raw_cname, (list, tuple)) and raw_cname:
            cnames = sorted(set(str(v) for v in raw_cname if v))
        elif isinstance(raw_cname, str) and raw_cname:
            cnames = [raw_cname]
        else:
            cnames = []

        # Predictions from .pth file
        preds_list: List[str] = []
        if ci is not None and ci < len(all_preds):
            preds_raw = all_preds[ci]
            if isinstance(preds_raw, (list, tuple)):
                preds_list = [str(p) for p in preds_raw]

        # Render prototypes
        color = MASK_COLORS[(rank - 1) % len(MASK_COLORS)]
        n_avail = get_num_prototypes(concept_info)
        n_protos = min(3, n_avail)
        protos: List[dict] = []
        for pi in range(n_protos):
            # Render WITH mask
            rendered_masked = render_prototype(
                concept_info,
                concept_dir=concept_dir,
                target_width=max(PROTO_W, 300),
                mask_color=color,
                proto_index=pi,
                draw_mask=True,
            )
            # Render WITHOUT mask (clean)
            rendered_clean = render_prototype(
                concept_info,
                concept_dir=concept_dir,
                target_width=max(PROTO_W, 300),
                mask_color=color,
                proto_index=pi,
                draw_mask=False,
            )
            if rendered_masked is not None:
                # Crop helper
                def _crop_proto(img: Image.Image) -> Image.Image:
                    w, h = img.size
                    scale = max(PROTO_W / w, PROTO_H / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    left = (new_w - PROTO_W) // 2
                    top_ = (new_h - PROTO_H) // 2
                    return img.crop((left, top_, left + PROTO_W, top_ + PROTO_H))

                masked_b64 = pil_to_base64(_crop_proto(rendered_masked), fmt="JPEG")
                clean_b64 = pil_to_base64(_crop_proto(rendered_clean), fmt="JPEG") if rendered_clean else masked_b64
                protos.append({
                    "image_b64": masked_b64,
                    "image_b64_clean": clean_b64,
                })

        concepts.append(
            {
                "rank": rank,
                "similarity": sim,
                "concept_index": ci if ci is not None else -1,
                "concept_names": cnames,
                "predictions": preds_list,
                "prototypes": protos,
            }
        )

    return {
        "model_output": model_output,
        "top_concepts": concepts,
        "mask_colors_hex": MASK_COLORS_HEX[: n_show],
        "bbox_colors_hex": BBOX_COLORS_HEX,
    }


@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    """Serve a temp image as base64 data URI."""
    img_path = _resolve_image_path(image_id)
    img = Image.open(img_path).convert("RGB")
    return {"image_b64": pil_to_base64(img, fmt="JPEG")}


# ── lifecycle ─────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event():
    print(f"VLM Classify API starting...")
    print(f"  Model:   {VLM_MODEL}")
    print(f"  Concept: {CONCEPT_PTH}")
    print(f"  Device:  {_device_str}")


@app.on_event("shutdown")
async def shutdown_event():
    global _explainer
    if _explainer is not None:
        try:
            _explainer.close()
        except Exception:
            pass
        _explainer = None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "demo.classify_api:app",
        host="0.0.0.0",
        port=int(os.getenv("CLASSIFY_API_PORT", "8501")),
        reload=False,
    )
