#!/usr/bin/env bash
set -euo pipefail

# Reuse env from orchestrator when available; provide soft defaults otherwise
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export HF_HOME="${HF_HOME:-/mnt/abka03/huggingface/hub}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR}"

# Resolve Python interpreter
if command -v "${PYTHON:-${PYTHON_BIN:-python}}" >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON:-${PYTHON_BIN:-python}}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

# Inputs/outputs
DATASET_PATH="${INPUT_DIR:-$ROOT_DIR/data/train}"
MODEL_NAME="${VLM_MODEL:-google/gemma-3n-E4B-it}"
INFER_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs}/inference"
mkdir -p "$INFER_DIR"
OUTPUT_CSV="${OUTPUT_CSV:-$INFER_DIR/objects.csv}"
CONCEPT_MAP_JSON="${CONCEPT_MAP_JSON:-$INFER_DIR/concepts_to_images.json}"

# Optional knobs
BATCH_SIZE_ARG=()
if [[ -n "${BATCH_SIZE:-}" ]]; then BATCH_SIZE_ARG=(--batch_size "${BATCH_SIZE}"); fi
IMAGE_SIZE_ARG=()
if [[ -n "${IMAGE_SIZE:-}" ]]; then IMAGE_SIZE_ARG=(--image_size ${IMAGE_SIZE}); fi # e.g., export IMAGE_SIZE="1048 1048"
IMAGE_BUDGET_ARG=()
if [[ -n "${IMAGE_BUDGET:-}" ]]; then IMAGE_BUDGET_ARG=(--image_budget "${IMAGE_BUDGET}"); fi
PROMPT_TEXT="${PROMPT:-Identify all visible objects, items, textures, colors, materials, and notable visual patterns in the given image. Output only a single-word, comma-separated list. Do not include explanations, sentences, or any extra text—just the detected elements.}"

PROGRAM="$ROOT_DIR/inference/dataset_inference.py"
CONCEPT_MAPPING_SCRIPT="$ROOT_DIR/concept_image_mapping.py"

"$PYTHON_BIN" "$PROGRAM" \
    --dataset_path "$DATASET_PATH" \
    --model_name "$MODEL_NAME" \
    --output_csv "$OUTPUT_CSV" \
    --prompt "$PROMPT_TEXT" \
    --trust_remote_code \
    "${BATCH_SIZE_ARG[@]}" \
    "${IMAGE_SIZE_ARG[@]}" \
    "${IMAGE_BUDGET_ARG[@]}"

"$PYTHON_BIN" "$CONCEPT_MAPPING_SCRIPT" --input "$OUTPUT_CSV" --output "$CONCEPT_MAP_JSON"