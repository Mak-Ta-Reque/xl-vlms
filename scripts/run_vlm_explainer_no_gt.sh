#!/usr/bin/env bash
set -euo pipefail

# Run VLM Explainer (without GT). Prefer orchestrator-provided env.
# Recognized env: WORKSPACE_DIR, PYTHON, HF_HOME, VLM_MODEL, CONCEPT_PATH, LAYER_PATH,
# IMAGE_ROOT, TOP_N, EXPLAIN_DIR, DECOMP_DIR

WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/abka03/Projects/xl-vlms}"
# Resolve Python interpreter
if command -v "${PYTHON:-${PYTHON_BIN:-python}}" >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON:-${PYTHON_BIN:-python}}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

export HF_HOME="${HF_HOME:-/mnt/abka03/huggingface/hub}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-2}"

MODEL_NAME="${VLM_MODEL:-${MODEL_NAME:-google/gemma-3n-E4B-it}}"
LAYER_PATH="${LAYER_PATH:-model.language_model.norm}"

# Default concept path from decomposition outputs if not provided
CONCEPT_PATH="${CONCEPT_PATH:-${DECOMP_DIR:-${WORKSPACE_DIR}/outputs/cdgl/train/concept}/combined_concept_snmf_raw.pth}"

# Default image root to validation data
IMAGE_ROOT="${IMAGE_ROOT:-${WORKSPACE_DIR}/data/val}"
TOP_N="${TOP_N:-5}"

OUT_JSON_DIR="${EXPLAIN_DIR:-${WORKSPACE_DIR}/outputs}"
mkdir -p "$OUT_JSON_DIR"
OUT_JSON="${OUT_JSON:-${OUT_JSON_DIR}/vlm_explanations.json}"

cd "$WORKSPACE_DIR"

"$PYTHON_BIN" "$WORKSPACE_DIR/inference/vlm_explainer.py" \
  --model_name "$MODEL_NAME" \
  --concept_path "$CONCEPT_PATH" \
  --layer_path "$LAYER_PATH" \
  --image_root "$IMAGE_ROOT" \
  --top_n "$TOP_N" \
  --out_json "$OUT_JSON"
