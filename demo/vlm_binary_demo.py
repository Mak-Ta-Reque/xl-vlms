#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   Interactive VLM Concept Explorer — Streamlit Demo             ║
║                                                                  ║
║   Upload image → ask question → click nouns → see concepts      ║
╚══════════════════════════════════════════════════════════════════╝

Usage:  conda run -n xlvlm-v1 streamlit run demo/vlm_binary_demo.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# ──────────────────────── project paths ───────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent

# Load .env BEFORE any torch / transformers imports so HF_HOME, DEVICE_ID etc.
# are visible to the model loaders (same order as the notebook).
from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

# Read device config from .env DEVICE variable
_device_str = os.getenv("DEVICE", "auto")
if "DEVICE" not in os.environ and "DEVICE_ID" in os.environ:
    _device_str = f"cuda:{os.environ['DEVICE_ID']}"

import streamlit as st
from PIL import Image

sys.path.insert(0, str(PROJECT_ROOT / "inference"))
sys.path.insert(0, str(_THIS_DIR))

from vis_utils import (
    MASK_COLORS,
    MASK_COLORS_HEX,
    get_num_prototypes,
    render_prototype,
    resize_to_width,
)

# ──────────────────────── env / config ────────────────────────────────────


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

# Fixed instruction prepended to every user prompt.
# Set FIXED_INSTRUCTION="" in .env to disable.
FIXED_INSTRUCTION = os.getenv("FIXED_INSTRUCTION", "Describe the image briefly. ")


# ──────────────────────── Custom CSS ─────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header gradient ── */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(56, 189, 248, 0.2);
}
.hero-header h1 {
    color: #f1f5f9;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
}
.hero-header p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}

/* ── Step badges ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, #1e293b, #334155);
    color: #f1f5f9;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    border: 1px solid rgba(56, 189, 248, 0.3);
    margin-bottom: 0.8rem;
}

/* ── Noun pills ── */
div[data-testid="stHorizontalBlock"] button {
    border-radius: 20px !important;
    font-weight: 600 !important;
    text-transform: capitalize !important;
    transition: all 0.2s ease !important;
    border: 2px solid #38bdf8 !important;
    background: transparent !important;
    color: #38bdf8 !important;
}
div[data-testid="stHorizontalBlock"] button:hover {
    background: #38bdf8 !important;
    color: #0f172a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(56, 189, 248, 0.3) !important;
}

/* ── Concept cards ── */
.concept-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 12px;
    padding: 1.2rem;
    border-left: 4px solid;
    margin-bottom: 1rem;
}
.concept-card h4 { color: #f1f5f9; margin: 0 0 0.3rem 0; }
.concept-card .sim-value { font-size: 1.6rem; font-weight: 700; }
.concept-card .label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Model output box ── */
.model-output-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #e2e8f0;
    font-size: 1.05rem;
    line-height: 1.6;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] .stMarkdown { color: #cbd5e1; }

/* ── Info/warning tweaks ── */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* ── Token table ── */
.token-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1e293b;
}
.token-chip {
    background: #334155;
    color: #f1f5f9;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-family: 'Fira Code', monospace;
    font-size: 0.85rem;
    font-weight: 600;
}
</style>
"""


# ──────────────────────── spaCy noun extraction ──────────────────────────

@st.cache_resource
def _load_spacy():
    """Load the spaCy model once (cached across reruns)."""
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("⏳ Downloading spaCy model `en_core_web_sm` …")
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def extract_nouns(text: str) -> List[str]:
    """Extract unique noun lemmas via spaCy noun chunks + POS tagging."""
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


# ──────────────────────── VLM explainer (cached) ─────────────────────────

@st.cache_resource(show_spinner=False)
def _load_explainer():
    """Load explainer once — cached across all Streamlit reruns.

    Configuration mirrors explain_binary.ipynb cell 4 exactly:
    - device = "cuda"
    - normalize_concepts = False (cosine similarity is scale-invariant)
    - save_only_generated_tokens = True (align activations to generated tokens)
    """
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


def run_unsupervised(image_path: str, prompt: str) -> Dict[str, Any]:
    """Run the VLM with a custom free-form prompt."""
    exp = _load_explainer()
    old_prompt = exp.prompt
    exp.prompt = FIXED_INSTRUCTION + prompt
    try:
        results = exp.explain_with_concept(
            images=[image_path], top_n=5,
            max_new_tokens=120, temperature=0.0, batch_size=1,
        )
    finally:
        exp.prompt = old_prompt
    return results[0]


def run_binary_for_noun(image_path: str, noun: str) -> Dict[str, Any]:
    """Run binary-mode concept scoring for a single noun."""
    exp = _load_explainer()
    old_prompt, old_mode, old_label = exp.prompt, exp.prompt_mode, exp.prompt_label
    exp.prompt = None
    exp.prompt_mode = "binary"
    exp.prompt_label = noun
    try:
        results = exp.explain_with_concept(
            images=[image_path], ground_truth_labels=[noun],
            top_n=5, max_new_tokens=80, temperature=0.0, batch_size=1,
        )
    finally:
        exp.prompt, exp.prompt_mode, exp.prompt_label = old_prompt, old_mode, old_label
    return results[0]


# ──────────────────────── UI Components ──────────────────────────────────

def _step_badge(number: int, text: str) -> None:
    st.markdown(
        f'<div class="step-badge">Step {number} &mdash; {text}</div>',
        unsafe_allow_html=True,
    )


def _concept_card(rank: int, concept_info: dict, color_hex: str) -> None:
    """Render a concept info card."""
    sim = concept_info.get("similarity", 0.0)
    text_grounding = concept_info.get("text_grounding", "—")
    concept_idx = concept_info.get("concept_index", "?")
    pct = f"{sim * 100:.1f}%"

    st.markdown(
        f"""<div class="concept-card" style="border-left-color: {color_hex};">
            <div class="label">Concept #{rank} &bull; Index {concept_idx}</div>
            <h4 style="color: {color_hex};">{text_grounding}</h4>
            <div><span class="sim-value" style="color: {color_hex};">{pct}</span>
            <span class="label" style="margin-left: 0.4rem;">cosine similarity ({sim:.4f})</span></div>
        </div>""",
        unsafe_allow_html=True,
    )


# ──────────────────────── Main App ────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="VLM Concept Explorer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Hero header ──
    st.markdown(
        """<div class="hero-header">
            <h1>🔬 VLM Concept Explorer</h1>
            <p>Upload an image · ask a question · click nouns in the answer
            to reveal which learned visual concepts the model activates.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown(f"**Model** `{VLM_MODEL.split('/')[-1]}`")
        st.markdown(f"**Concept bank** `{CONCEPT_PTH.name}`")
        st.markdown(f"**Hook layer** `{LAYER_PATH}`")
        st.divider()
        vis_width = st.slider("Image width (px)", 256, 1024, IMAGE_SIZE_WIDTH, step=64)
        n_protos = st.slider("Prototypes per concept", 1, 5, 3)
        show_masks = st.toggle("Show segmentation masks", value=True)
        st.divider()
        st.caption("Built with Streamlit · VLMConceptExplainer")

    # ── Image upload + question ──
    col_upload, col_q = st.columns([1, 1], gap="large")

    with col_upload:
        uploaded = st.file_uploader(
            "📷 Upload an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Drag & drop or click to browse",
        )

    with col_q:
        user_question = st.text_area(
            "💬 Ask a question about the image",
            value="What objects are in this image?",
            height=100,
        )

    if uploaded is None:
        st.markdown("---")
        cc1, cc2, cc3 = st.columns(3)
        cc1.info("**1 ·** Upload any image")
        cc2.info("**2 ·** Ask a question")
        cc3.info("**3 ·** Click nouns to explore concepts")
        return

    # Save upload to temp file
    suffix = Path(uploaded.name).suffix or ".jpg"
    temp_dir = PROJECT_ROOT / "api" / "temp_images"
    temp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=str(temp_dir)
    )
    tmp_file.write(uploaded.getvalue())
    tmp_file.flush()
    tmp_path = tmp_file.name
    tmp_file.close()

    # Show uploaded image
    input_img = Image.open(tmp_path).convert("RGB")
    input_img_resized = resize_to_width(input_img, vis_width)
    col_img, col_info = st.columns([2, 1], gap="large")
    with col_img:
        st.image(input_img_resized, width="stretch")
    with col_info:
        st.markdown(f"**Filename:** `{uploaded.name}`")
        st.markdown(f"**Size:** {input_img.size[0]}×{input_img.size[1]}")
        st.markdown(f"**Instruction:** {FIXED_INSTRUCTION}")
        st.markdown(f"**Question:** {user_question}")

    st.markdown("---")

    # ── Step 1: VLM inference ──
    _step_badge(1, "VLM generates an answer")

    # Check if we need to rerun
    need_rerun = (
        "vlm_result" not in st.session_state
        or st.session_state.get("_last_q") != user_question
        or st.session_state.get("_last_img") != uploaded.name
    )

    if need_rerun:
        with st.status("🧠 Running VLM inference …", expanded=True) as status:
            st.write("Loading model (cached after first run) …")
            result = run_unsupervised(tmp_path, user_question)
            st.session_state["vlm_result"] = result
            st.session_state["_last_q"] = user_question
            st.session_state["_last_img"] = uploaded.name
            st.session_state["_tmp_path"] = tmp_path
            st.session_state.pop("selected_noun", None)
            # Clear old binary caches
            for k in list(st.session_state.keys()):
                if k.startswith("_binary_"):
                    del st.session_state[k]
            status.update(label="✅ VLM inference complete", state="complete")

    result = st.session_state["vlm_result"]
    model_text: str = result.get("model_output", "").strip()
    st.markdown(
        f'<div class="model-output-box">💡 <strong>Model says:</strong> {model_text}</div>',
        unsafe_allow_html=True,
    )

    # ── Step 2: Noun extraction ──
    st.markdown("")
    _step_badge(2, "Select a noun to explain")

    nouns = extract_nouns(model_text)
    if not nouns:
        st.warning("No nouns detected. Try a different question for richer output.")
        return

    # Render noun pills as buttons
    n_cols = min(len(nouns), 6)
    cols = st.columns(n_cols)
    for i, noun in enumerate(nouns):
        col = cols[i % n_cols]
        if col.button(f"🔍 {noun}", key=f"noun_{i}"):
            st.session_state["selected_noun"] = noun

    selected = st.session_state.get("selected_noun")
    if selected is None:
        st.info("👆 Click a noun above to see which visual concepts activate for it.")
        return

    st.markdown("---")

    # ── Step 3: Binary concept scoring ──
    _step_badge(3, f'Concept explanation for "{selected}"')

    cache_key = f"_binary_{selected}"
    if cache_key not in st.session_state:
        with st.status(f"🔬 Scoring concepts for **{selected}** …", expanded=True) as status:
            binary_result = run_binary_for_noun(
                st.session_state.get("_tmp_path", tmp_path), selected,
            )
            st.session_state[cache_key] = binary_result
            status.update(label=f"✅ Concepts scored for \"{selected}\"", state="complete")

    binary_result = st.session_state[cache_key]
    binary_text = binary_result.get("model_output", "").strip()

    st.markdown(
        f'<div class="model-output-box">🏷️ <strong>Binary check:</strong> {binary_text}</div>',
        unsafe_allow_html=True,
    )

    top_concepts = binary_result.get("top_concepts_over_sequence") or []
    if not top_concepts:
        st.warning("No concepts scored — the concept bank may not cover this noun.")
        return

    concept_dir = CONCEPT_PTH.parent
    n_concepts_show = min(3, len(top_concepts))

    # ── Concept cards + prototype grids ──
    for rank, concept_info in enumerate(top_concepts[:n_concepts_show], start=1):
        color = MASK_COLORS[(rank - 1) % len(MASK_COLORS)]
        color_hex = MASK_COLORS_HEX[(rank - 1) % len(MASK_COLORS_HEX)]

        col_card, col_protos = st.columns([1, 3], gap="large")

        with col_card:
            _concept_card(rank, concept_info, color_hex)

        with col_protos:
            n_avail = get_num_prototypes(concept_info)
            n_show = min(n_protos, n_avail)
            if n_show == 0:
                st.caption("No prototype images available")
                continue

            proto_cols = st.columns(n_show)
            for pi in range(n_show):
                rendered = render_prototype(
                    concept_info,
                    concept_dir=concept_dir,
                    target_width=vis_width,
                    mask_color=color,
                    proto_index=pi,
                    draw_mask=show_masks,
                )
                if rendered is not None:
                    proto_cols[pi].image(
                        rendered,
                        caption=f"Prototype {pi + 1}",
                        width="stretch",
                    )
                else:
                    proto_cols[pi].markdown("_—_")

        if rank < n_concepts_show:
            st.markdown("")

    # ── Per-token breakdown (collapsible) ──
    with st.expander("🧬 Per-token concept breakdown", expanded=False):
        per_tok = binary_result.get("per_token_concepts") or []
        if not per_tok:
            st.caption("No per-token data available.")
        else:
            for tok_info in per_tok:
                token_text = tok_info.get("token_text", "").strip()
                if not token_text:
                    continue
                top_c = (tok_info.get("top_concepts") or [None])[0]
                if top_c:
                    sim = top_c.get("similarity", 0)
                    tg = top_c.get("text_grounding", "—")
                    st.markdown(
                        f'<div class="token-row">'
                        f'<span class="token-chip">{token_text}</span>'
                        f'→ <strong>{tg}</strong> '
                        f'<span style="color:#94a3b8">(sim {sim:.4f})</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
