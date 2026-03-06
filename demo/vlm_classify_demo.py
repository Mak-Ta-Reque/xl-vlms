#!/usr/bin/env python3
"""
Compact VLM Classification Demo — clean interface for paper figures.

Upload an image → model classifies it → top concepts shown compactly.

Usage:  conda run -n xlvlm-v1 streamlit run demo/vlm_classify_demo.py
"""

from __future__ import annotations

import os
import sys
import re
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent

from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

_device_str = os.getenv("DEVICE", "auto")
if "DEVICE" not in os.environ and "DEVICE_ID" in os.environ:
    _device_str = f"cuda:{os.environ['DEVICE_ID']}"

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

sys.path.insert(0, str(PROJECT_ROOT / "inference"))
sys.path.insert(0, str(_THIS_DIR))

import torch

from vis_utils import (
    MASK_COLORS,
    MASK_COLORS_HEX,
    get_num_prototypes,
    render_prototype,
    resize_to_width,
)

# ──────────────────────── config ──────────────────────────────────────────

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
DISPLAY_W = int(os.getenv("DISPLAY_W", "500"))

# Distinct bbox palette (warm/earthy) — avoids clashing with concept colours
BBOX_COLORS: list[tuple[int, int, int]] = [
    (230, 126, 34),   # orange
    (39, 174, 96),    # green
    (192, 57, 43),    # red
    (41, 128, 185),   # blue
    (142, 68, 173),   # purple
]
BBOX_COLORS_HEX: list[str] = [
    "#e67e22", "#27ae60", "#c0392b", "#2980b9", "#8e44ad",
]

CLASSIFY_PROMPT = os.getenv(
    "CLASSIFY_PROMPT",
    "Name only the main objects in each part of the image. "
    "Answer with a short comma-separated list of single words, no descriptions.",
)

# ──────────────────────── CSS ─────────────────────────────────────────────

COMPACT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* lock to viewport — no scrolling */
html, body { overflow: hidden !important; height: 100vh !important; }
.main .block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0 !important;
    max-width: 960px !important;
    overflow: hidden !important;
}
/* hide Streamlit footer & hamburger */
footer, #MainMenu, header[data-testid="stHeader"] { display: none !important; }

/* header */
.compact-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    margin-bottom: 0.4rem;
    border: 1px solid rgba(56,189,248,0.15);
    display: flex; align-items: center; gap: 0.5rem;
}
.compact-header h2 { color: #f1f5f9; margin: 0; font-size: 1rem; font-weight: 700; }
.compact-header span { color: #64748b; font-size: 0.75rem; }

/* prediction badge */
.pred-badge { text-align: center; margin: 0.2rem 0; }
.pred-badge .label { color: #94a3b8; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; }
.pred-badge .value {
    font-size: 1.3rem; font-weight: 700; color: #38bdf8;
    background: #0f172a; border-radius: 6px; padding: 0.2rem 0.8rem;
    display: inline-block; margin-top: 0.1rem;
    border: 1px solid rgba(56,189,248,0.25);
}

/* concept row */
.concept-row {
    display: flex; align-items: center; gap: 0.3rem;
    background: #1e293b; border-radius: 6px;
    padding: 0.2rem 0.5rem; margin-bottom: 0.15rem;
    border-left: 3px solid;
    white-space: nowrap; overflow: hidden;
}
.concept-row .rank { color: #64748b; font-size: 0.9rem; font-weight: 700; min-width: 1.2rem; }
.concept-row .name { color: #e2e8f0; font-weight: 600; font-size: 0.95rem; }
.concept-row .sim { font-family: 'Fira Code', monospace; font-size: 0.9rem; font-weight: 600; margin-left: auto; }

/* section label */
.section-label {
    color: #64748b; font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.1em; font-weight: 600; margin: 0.3rem 0 0.15rem 0;
}

/* image — centred, natural size, clickable */
.fixed-img { text-align: center; }
.fixed-img img { max-height: none !important; object-fit: contain; cursor: pointer; }

/* selected class row below image */
.selected-row {
    display: flex; align-items: center; gap: 0.5rem;
    margin: 0.3rem 0 0.15rem 0;
}

/* prototype row */
.proto-row img { max-height: 120px !important; object-fit: contain; }

/* selected-class pill shown after bbox click */
.selected-pill {
    display: inline-block; font-weight: 700; font-size: 1.2rem;
    padding: 0.15rem 0.7rem; border-radius: 6px;
    text-transform: capitalize; margin-top: 0.15rem;
}

/* shrink uploader */
[data-testid="stFileUploader"] { margin-bottom: 0.3rem; }
[data-testid="stFileUploader"] section { padding: 0.3rem !important; }
[data-testid="stFileUploader"] small { font-size: 0.7rem; }
</style>
"""

# ──────────────────────── concept file loader (debug) ─────────────────────

@st.cache_resource
def _load_concept_file_predictions() -> List:
    """Load image_grounding_predictions from the concept .pth file."""
    try:
        data = torch.load(str(CONCEPT_PTH), map_location="cpu")
        return data.get("image_grounding_predictions", [])
    except Exception:
        return []

# ──────────────────────── spaCy noun extraction ──────────────────────────

@st.cache_resource
def _load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def extract_nouns(text: str) -> List[str]:
    """Extract unique noun lemmas from model output."""
    nlp = _load_spacy()
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


# ──────────────────────── model ───────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_explainer():
    from vlm_explainer_multibatch import VLMConceptExplainer
    return VLMConceptExplainer(
        model_name=VLM_MODEL,
        concept_path=str(CONCEPT_PTH),
        layer_path=LAYER_PATH,
        device=_device_str,
        prompt_mode="unsupervised",
        normalize_concepts=False,
        verbose=False,
        save_only_generated_tokens=True,
    )


def classify_image(image_path: str) -> Dict[str, Any]:
    exp = _load_explainer()
    old_prompt = exp.prompt
    exp.prompt = CLASSIFY_PROMPT
    try:
        results = exp.explain_with_concept(
            images=[image_path], top_n=5,
            max_new_tokens=60, temperature=0.0, batch_size=1,
        )
    finally:
        exp.prompt = old_prompt
    return results[0]


def run_binary_for_class(image_path: str, label: str) -> Dict[str, Any]:
    """Run binary-mode concept scoring for a selected class."""
    exp = _load_explainer()
    old_prompt, old_mode, old_label = exp.prompt, exp.prompt_mode, exp.prompt_label
    exp.prompt = None
    exp.prompt_mode = "binary"
    exp.prompt_label = label
    try:
        results = exp.explain_with_concept(
            images=[image_path], ground_truth_labels=[label],
            top_n=5, max_new_tokens=80, temperature=0.0, batch_size=1,
        )
    finally:
        exp.prompt, exp.prompt_mode, exp.prompt_label = old_prompt, old_mode, old_label
    return results[0]


# ──────────────────────── grounding / bbox ────────────────────────────────


def _bbox_prompt_for_nouns(nouns: List[str]) -> str:
    """Build a grounding prompt that mentions the specific nouns."""
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
    """Prompt for a single object's bounding box."""
    return (
        f"Draw a tight bounding box around the main '{noun}' in the image. "
        f"The image may be a single photo or a grid. "
        f'Return ONLY: [{{"name": "{noun}", "bbox": [x_min, y_min, x_max, y_max]}}] '
        f"with pixel coordinates. No other text."
    )


def _run_grounding(image_path: str, nouns: List[str]) -> List[Dict[str, Any]]:
    """Ask the VLM to return bounding boxes, with per-noun fallback."""
    exp = _load_explainer()
    old_prompt = exp.prompt

    # -- Attempt 1: bulk prompt with known nouns --
    exp.prompt = _bbox_prompt_for_nouns(nouns)
    try:
        results = exp.explain_with_concept(
            images=[image_path], top_n=1,
            max_new_tokens=512, temperature=0.0, batch_size=1,
        )
    finally:
        exp.prompt = old_prompt
    raw_text = results[0].get("model_output", "")
    objects = _parse_bbox_json(raw_text)

    # -- Check which nouns were found --
    found_nouns = set()
    for obj in objects:
        for noun in nouns:
            if noun in obj["name"] or obj["name"] in noun:
                found_nouns.add(noun)
                break

    missing = [n for n in nouns if n not in found_nouns]

    # -- Attempt 2: per-noun fallback for missing nouns --
    for noun in missing:
        exp.prompt = _bbox_prompt_single(noun)
        try:
            res = exp.explain_with_concept(
                images=[image_path], top_n=1,
                max_new_tokens=300, temperature=0.0, batch_size=1,
            )
        finally:
            exp.prompt = old_prompt
        extra = _parse_bbox_json(res[0].get("model_output", ""))
        if extra:
            for o in extra:
                o["name"] = noun
            objects.extend(extra[:1])

    return objects


def _parse_bbox_json(text: str) -> List[Dict[str, Any]]:
    """Parse JSON bboxes from model output, with fallback regex."""
    text = text.strip()
    # Strip code fences (```json ... ```)
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_\-]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Try to find a JSON blob — could be {"objects":[...]} or bare [...]
    blob = None
    if text.startswith("[") and "]" in text:
        # Bare array response
        blob = text[:text.rfind("]") + 1]
    elif text.startswith("{") and "}" in text:
        blob = text[:text.rfind("}") + 1]
    else:
        # Search for either array or object
        m = re.search(r"[\[\{][\s\S]*[\]\}]", text)
        if m:
            blob = m.group(0)

    # Repair truncated JSON
    if blob is None and ("[" in text or "{" in text):
        start = min(
            (text.index(c) for c in "[{" if c in text),
        )
        blob = text[start:]
        open_sq = blob.count("[") - blob.count("]")
        open_br = blob.count("{") - blob.count("}")
        blob += "]" * max(0, open_sq) + "}" * max(0, open_br)

    objects: List[Dict[str, Any]] = []
    if blob:
        try:
            js = json.loads(blob)
            # Handle both {"objects": [...]} and bare [...]
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
    # Fallback regex
    if not objects:
        for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
            name = m.group(1)
            rest = text[m.end():m.end() + 200]
            bm = re.search(r'"bbox"\s*:\s*\[([^\]]+)\]', rest)
            if bm:
                nums = re.findall(r'-?\d*\.?\d+', bm.group(1))
                if len(nums) >= 4:
                    bb = [float(nums[i]) for i in range(4)]
                    objects.append({"name": name.lower().strip(), "bbox": bb})
    return objects


def _draw_bboxes_on_image(
    img: Image.Image,
    objects: List[Dict[str, Any]],
    nouns: List[str],
) -> Tuple[Image.Image, List[Tuple[float, float, float, float, str]]]:
    """Draw color-coded bounding boxes + labels on image.

    Returns (annotated_image, click_regions) where each click region is
    (x1, y1, x2, y2, noun) in *original* pixel coordinates.
    """
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    w, h = img_draw.size

    # Build noun→color map (use BBOX palette, distinct from concept colours)
    noun_colors: Dict[str, Tuple[int, int, int]] = {}
    for i, noun in enumerate(nouns):
        noun_colors[noun] = BBOX_COLORS[i % len(BBOX_COLORS)]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    click_regions: List[Tuple[float, float, float, float, str]] = []

    for obj in objects:
        name = obj["name"]
        bbox = obj["bbox"]
        # Match to closest noun
        color = (200, 200, 200)  # default grey
        matched_name = name
        for noun in nouns:
            if noun in name or name in noun:
                color = noun_colors[noun]
                matched_name = noun
                break

        # Convert coords to pixel coords on actual image
        x1, y1, x2, y2 = bbox
        # Detect coordinate space:
        #   - All in [0,1] → normalized, scale to image dims
        #   - Otherwise → pixel coords already in image space (use directly)
        if all(0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
        # (pixel coords >1 are used as-is — model outputs in actual image resolution)
        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        # Ensure x1 < x2 and y1 < y2 (model may return swapped coords)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        # Skip degenerate boxes
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        # Label background
        label = matched_name
        tw = draw.textlength(label, font=font)
        th = 27
        label_y = max(0, y1 - th - 2)
        draw.rectangle([x1, label_y, x1 + tw + 6, label_y + th + 2], fill=color)
        draw.text((x1 + 3, label_y + 1), label, fill="black", font=font)

        # Store click region (in original pixel coords)
        click_regions.append((x1, y1, x2, y2, matched_name))

    return img_draw, click_regions


# ──────────────────────── app ─────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="XL-VLM Classifier",
        page_icon="🔬",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(COMPACT_CSS, unsafe_allow_html=True)

    # Debug toggle in sidebar
    with st.sidebar:
        debug_mode = st.checkbox("Debug mode", value=False, key="_debug_mode")

    # Header
    st.markdown(
        '<div class="compact-header">'
        '<h2>🔬 CVT LVLM Explainer [Qwen2.5-VL-3B] </h2>'
        '<span>Upload → Detect → Explain</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Upload
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.caption("Drop an image above to classify it.")
        # Sample image download links
        _samples_dir = _THIS_DIR / "samples"
        _sample_files = [
            "n02325366_6190.JPEG",
            "n02123045_10633.JPEG",
            "grid_4_65__rabbit__brown_bear__tiger__dog.jpg",
        ]
        st.markdown(
            '<div class="section-label" style="margin-top:0.5rem;">Sample images</div>',
            unsafe_allow_html=True,
        )
        _dl_cols = st.columns(len(_sample_files))
        for _ci, _fname in enumerate(_sample_files):
            _fpath = _samples_dir / _fname
            if _fpath.exists():
                with _dl_cols[_ci]:
                    with open(_fpath, "rb") as _f:
                        st.download_button(
                            label=_fname.split(".")[0][:18],
                            data=_f,
                            file_name=_fname,
                            mime="image/jpeg",
                            key=f"dl_{_ci}",
                        )
        return

    # Save to temp
    suffix = Path(uploaded.name).suffix or ".jpg"
    temp_dir = PROJECT_ROOT / "api" / "temp_images"
    temp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(temp_dir))
    tmp.write(uploaded.getvalue())
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    input_img = Image.open(tmp_path).convert("RGB")

    # Run classification
    need_rerun = (
        "cls_result" not in st.session_state
        or st.session_state.get("_last_img") != uploaded.name
    )
    if need_rerun:
        with st.spinner("Classifying…"):
            result = classify_image(tmp_path)
            st.session_state["cls_result"] = result
            st.session_state["_last_img"] = uploaded.name
            st.session_state["_tmp_path"] = tmp_path
            st.session_state.pop("selected_class", None)
            # Clear old binary caches
            for k in list(st.session_state.keys()):
                if k.startswith("_binary_"):
                    del st.session_state[k]
            st.session_state.pop("_grounding", None)

    result = st.session_state["cls_result"]
    prediction = result.get("model_output", "").strip()

    # Extract class nouns from model output
    nouns = extract_nouns(prediction)

    # Run grounding for bounding boxes
    if "_grounding" not in st.session_state and nouns:
        with st.spinner("Detecting bounding boxes…"):
            grounding_objs = _run_grounding(
                st.session_state.get("_tmp_path", tmp_path), nouns,
            )
            st.session_state["_grounding"] = grounding_objs
    grounding_objs = st.session_state.get("_grounding", [])

    # Draw bboxes on image
    click_regions: List[Tuple[float, float, float, float, str]] = []
    if grounding_objs and nouns:
        display_img, click_regions = _draw_bboxes_on_image(input_img, grounding_objs, nouns)
    else:
        display_img = input_img

    # Prepare image for display — resize to DISPLAY_W for compact layout
    display_resized = resize_to_width(display_img, DISPLAY_W)
    orig_w, orig_h = input_img.size
    disp_w, disp_h = display_resized.size
    scale_x = orig_w / disp_w
    scale_y = orig_h / disp_h

    # ── Layout: centred image, selected class below ──
    st.markdown('<div class="fixed-img">', unsafe_allow_html=True)
    coords = streamlit_image_coordinates(
        display_resized,
        key="img_click",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Check if click falls inside a bounding box
    if coords is not None and click_regions:
        cx, cy = coords["x"], coords["y"]
        ox, oy = cx * scale_x, cy * scale_y
        for (bx1, by1, bx2, by2, noun) in click_regions:
            if bx1 <= ox <= bx2 and by1 <= oy <= by2:
                if st.session_state.get("selected_class") != noun:
                    st.session_state["selected_class"] = noun
                    st.rerun()
                break

    # Selected class indicator (inline below image)
    _sel = st.session_state.get("selected_class")
    if _sel and _sel in nouns:
        _idx = nouns.index(_sel)
        _hex = BBOX_COLORS_HEX[_idx % len(BBOX_COLORS_HEX)]
        st.markdown(
            f'<div class="selected-row">'
            f'<span class="section-label" style="margin:0;">Selected class</span>'
            f'<span class="selected-pill" style="color:{_hex}; border:1.5px solid {_hex};">{_sel}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif nouns:
        st.markdown('<div class="section-label">Click a bounding box to explain</div>', unsafe_allow_html=True)

    # ── Explanation for selected class ──
    selected = st.session_state.get("selected_class")
    if selected is None:
        return

    cache_key = f"_binary_{selected}"
    if cache_key not in st.session_state:
        with st.spinner(f"Explaining {selected}…"):
            binary_result = run_binary_for_class(
                st.session_state.get("_tmp_path", tmp_path), selected,
            )
            st.session_state[cache_key] = binary_result

    binary_result = st.session_state[cache_key]
    top_concepts = binary_result.get("top_concepts_over_sequence") or []

    # Show model output in debug mode
    if debug_mode:
        _model_out = binary_result.get("model_output", "")
        with st.expander(f"Model output for \u201c{selected}\u201d", expanded=False):
            st.code(_model_out, language=None)

    # ── Concepts + prototypes for selected class (inline rows) ──
    st.markdown(
        f'<div class="section-label">Concepts for &ldquo;{selected}&rdquo;</div>',
        unsafe_allow_html=True,
    )
    n_show = min(3, len(top_concepts))
    concept_dir = CONCEPT_PTH.parent
    PROTO_W, PROTO_H = 140, 100

    for rank, concept_info in enumerate(top_concepts[:n_show], 1):
        sim = concept_info.get("similarity", 0.0)
        tg = f"Concept-{rank}"
        color = MASK_COLORS[(rank - 1) % len(MASK_COLORS)]
        color_hex = MASK_COLORS_HEX[(rank - 1) % len(MASK_COLORS_HEX)]
        pct = f"{sim * 100:.1f}%"

        # Extract concept name: concept_name is a list of identical values → use set to get single name
        raw_cname = concept_info.get("concept_name")
        if isinstance(raw_cname, (list, tuple)) and raw_cname:
            concept_label = ", ".join(sorted(set(str(v) for v in raw_cname if v)))
        elif isinstance(raw_cname, str) and raw_cname:
            concept_label = raw_cname
        else:
            concept_label = ""

        # Get predictions for this concept from the concept file
        ci = concept_info.get("concept_index")
        all_preds = _load_concept_file_predictions()
        preds_list = []
        if ci is not None and ci < len(all_preds):
            preds_raw = all_preds[ci]
            if isinstance(preds_raw, (list, tuple)):
                preds_list = [str(p) for p in preds_raw]

        n_avail = get_num_prototypes(concept_info)
        n_protos = min(3, n_avail)

        # Layout: [label+sim | proto1 | proto2 | proto3]
        col_sizes = [1] + [1] * max(1, n_protos)
        cols = st.columns(col_sizes, gap="small")

        with cols[0]:
            # Build concept name HTML (shown below concept tag + similarity)
            cname_html = ""
            if concept_label:
                cname_html = (
                    f'<div style="color:{color_hex}; font-size:0.8rem; font-weight:500; '
                    f'margin-top:0.1rem; padding-left:0.2rem; text-transform:capitalize;">'
                    f'{concept_label}</div>'
                )
            # Build predictions HTML (shown below concept name)
            preds_html = ""
            if preds_list:
                pills = " ".join(
                    f'<span style="display:inline-block; padding:0.05rem 0.35rem; '
                    f'border-radius:4px; font-size:0.7rem; font-weight:600; margin:0.05rem; '
                    f'background:{"#166534" if p.lower()=="yes" else "#7f1d1d"}; '
                    f'color:{"#bbf7d0" if p.lower()=="yes" else "#fecaca"};">'
                    f'{p}</span>'
                    for p in preds_list
                )
                preds_html = (
                    f'<div style="margin-top:0.1rem; padding-left:0.2rem; line-height:1.6;">'
                    f'{pills}</div>'
                )
            st.markdown(
                f'<div class="concept-row" style="border-left-color:{color_hex};">'
                f'<span class="name">{tg}</span>'
                f'<span class="sim" style="color:{color_hex};">{pct}</span>'
                f'</div>'
                f'{cname_html}'
                f'{preds_html}',
                unsafe_allow_html=True,
            )

        # Debug: show image_grounding_predictions for this concept
        if debug_mode:
            if ci is not None and ci < len(all_preds):
                with st.expander(f"Predictions for {tg} (idx={ci})", expanded=False):
                    preds = all_preds[ci]
                    if isinstance(preds, (list, tuple)):
                        for pi_pred, p in enumerate(preds):
                            st.text(f"{pi_pred}: {p}")
                    else:
                        st.text(str(preds))

        for pi in range(n_protos):
            rendered = render_prototype(
                concept_info,
                concept_dir=concept_dir,
                target_width=max(PROTO_W, 300),
                mask_color=color,
                proto_index=pi,
                draw_mask=True,
            )
            if rendered is not None:
                w, h = rendered.size
                scale = max(PROTO_W / w, PROTO_H / h)
                new_w, new_h = int(w * scale), int(h * scale)
                rendered = rendered.resize((new_w, new_h), Image.Resampling.LANCZOS)
                left = (new_w - PROTO_W) // 2
                top_ = (new_h - PROTO_H) // 2
                rendered = rendered.crop((left, top_, left + PROTO_W, top_ + PROTO_H))
                cols[1 + pi].image(rendered, width="stretch")


if __name__ == "__main__":
    main()
