#!/bin/bash

# Generate concept features (DL variant)
# Usage:
#   run_feature_gen_dl.sh [MODEL_NAME] [FEATURES_DIR] [HF_HOME]
# If not provided, will use sensible defaults matching previous behavior.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source .env as single source of truth
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi

# ========================
# Inputs and defaults
# ========================
MODEL_NAME_INPUT="${1:-}"
FEATURES_DIR_INPUT="${2:-}"
HF_HOME_INPUT="${3:-}"

# Fall back to env or previous hardcoded defaults
MODEL_NAME="${MODEL_NAME_INPUT:-${VLM_MODEL:-google/gemma-3n-E4B-it}}"
HF_HOME="${HF_HOME_INPUT:-${HF_HOME:-/mnt/abka03/huggingface/hub}}"

# The feature directory root; when invoked by pipeline, this should be OUTPUT_DIR
FEATURES_DIR_ROOT="${FEATURES_DIR_INPUT:-${FEATURES_DIR:-/mnt/abka03/Projects/xl-vlms/outputs}}"

# ========================
# Static/legacy configuration (kept for compatibility)
# Allow env overrides to align with VS Code launch config
# ========================
DATASET_NAME="${DATASET_NAME:-image}"
HOOK_NAME="${HOOK_NAME:-save_hidden_states_for_token_of_interest}"
MODULES_TO_HOOK="${MODULES_TO_HOOK:-model.language_model.norm}"
PYTHON_EXEC="${PYTHON_EXEC:-python}"
SCRIPT_PATH="src/save_features.py"
PROMT_TEMPLATE="${PROMT_TEMPLATE:-dl}"
SPLIT="${SPLIT:-train}"

# Base data dir is expected to contain concept folders with images
# Pick up from pipeline via env BASE_DATA_DIR and SPLIT; default to repo data/train
BASE_DATA_DIR="${BASE_DATA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/${SPLIT}}"

# Controls
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"

# Output dir root for features. Pipeline exports SAVE_DIR -> "$OUTPUT_DIR/SNMF/coco10/${SPLIT}"
# Python code appends 'features' under save_dir, so pass the ROOT and pre-create the subfolder.
SAVE_DIR="${SAVE_DIR:-${FEATURES_DIR_ROOT}}"
mkdir -p "$SAVE_DIR" "$SAVE_DIR/features"

echo "[feature_gen_dl] MODEL_NAME=$MODEL_NAME FEATURES_DIR_ROOT=$FEATURES_DIR_ROOT SPLIT=$SPLIT BASE_DATA_DIR=$BASE_DATA_DIR batch_size=${BATCH_SIZE:-<unset>} PIPELINE_DATASET_SIZE=${DEFAULT_DATASET_SIZE:-<unset>}" >&2

# ========================
# Start timer
# ========================
start_time=$(date +%s)

# ========================
# Feature Extraction Loop
# ========================
COUNT=0

# Support single-directory mode via DATA_DIR env; otherwise iterate subfolders
if [ -n "${DATA_DIR:-}" ]; then
  DIRS=("$DATA_DIR")
else
  DIRS=("$BASE_DATA_DIR"/*/)
fi

for dir_path in "${DIRS[@]}"; do
  if (( COUNT >= MAX_ITERATIONS )); then
    break
  fi

  # Use dataset size provided by pipeline (DEFAULT_DATASET_SIZE env). If unset, fall back to counting files.
  if [ -n "${DEFAULT_DATASET_SIZE:-}" ]; then
    DATASET_SIZE="$DEFAULT_DATASET_SIZE"
  else
    DATASET_SIZE=$(find "$dir_path" -type f -name "*.png" | wc -l)
  fi

  dir_name=$(basename "$dir_path")
  concept="${dir_name//_/ }"

  # Allow override of token of interest via env
  if [ -n "${TOKEN_OF_INTEREST:-}" ]; then
    TOKEN_OF_INTEREST="$TOKEN_OF_INTEREST"
  else
    if [[ "$dir_name" == *"_"* ]]; then
      TOKEN_OF_INTEREST="${dir_name##*_}"
    else
      TOKEN_OF_INTEREST="$dir_name"
    fi
  fi

  echo "Processing: $dir_name | Token: $TOKEN_OF_INTEREST | Split: $SPLIT"

  # Allow override of save filename via env; else derive default
  if [ -n "${SAVE_FILENAME:-}" ]; then
    SAVE_FILENAME="$SAVE_FILENAME"
  else
    SAVE_FILENAME="qwen2_patched_image_${dir_name}_token_of_interest_concept_generation_split_${SPLIT}"
  fi

  HF_HOME="$HF_HOME" "$PYTHON_EXEC" "$SCRIPT_PATH" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --dataset_size "$DATASET_SIZE" \
    --batch_size "${BATCH_SIZE:-2}" \
    --device "${DEVICE:-cuda:0}" \
    --data_dir "$dir_path" \
    --hook_name "$HOOK_NAME" \
    --token_of_interest "$TOKEN_OF_INTEREST" \
    --modules_to_hook "$MODULES_TO_HOOK" \
    --save_dir "$SAVE_DIR" \
    --save_filename "$SAVE_FILENAME" \
    --generation_mode \
    --prompt_template "$PROMT_TEMPLATE" \
    --save_only_generated_tokens \
    --slice_prediction \
    --exact_match_modules_to_hook

  COUNT=$((COUNT + 1))
done

# ========================
# End Timer
# ========================
end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

echo "Feature generation completed in ${hours}h ${minutes}m ${seconds}s"