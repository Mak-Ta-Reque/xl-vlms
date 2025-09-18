#!/bin/bash

# Generate concept features (DL variant)
# Usage:
#   run_feature_gen_dl.sh [MODEL_NAME] [FEATURES_DIR] [HF_HOME]
# If not provided, will use sensible defaults matching previous behavior.

set -Eeuo pipefail

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
# ========================
DATASET_NAME="image"
HOOK_NAME="save_hidden_states_for_token_of_interest"
MODULES_TO_HOOK="model.language_model.norm"
PYTHON_EXEC="${PYTHON_EXEC:-python}"
SCRIPT_PATH="src/save_features.py"
PROMT_TEMPLATE="dl"
SPLIT="${SPLIT:-train}"

# Base data dir is expected to contain concept folders with images
# Pick up from pipeline via env BASE_DATA_DIR and SPLIT; default to repo data/train
BASE_DATA_DIR="${BASE_DATA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/${SPLIT}}"

# Controls
MAX_ITERATIONS="${MAX_ITERATIONS:-10}"
DEFAULT_DATASET_SIZE="${DEFAULT_DATASET_SIZE:-50}"
OVERRIDE_DATASET_SIZE="${OVERRIDE_DATASET_SIZE:-50}"

# Output dir root for features. Pipeline exports SAVE_DIR -> "$OUTPUT_DIR/SNMF/coco10/${SPLIT}"
# Python code appends 'features' under save_dir, so pass the ROOT and pre-create the subfolder.
SAVE_DIR="${SAVE_DIR:-${FEATURES_DIR_ROOT}}"
mkdir -p "$SAVE_DIR" "$SAVE_DIR/features"

# ========================
# Start timer
# ========================
start_time=$(date +%s)

# ========================
# Feature Extraction Loop
# ========================
COUNT=0
for dir_path in "$BASE_DATA_DIR"/*/; do
  if (( COUNT >= MAX_ITERATIONS )); then
    break
  fi

  DATASET_SIZE=$(find "$dir_path" -type f -name "*.png" | wc -l)
  if [ "$DATASET_SIZE" -gt "$DEFAULT_DATASET_SIZE" ]; then
    DATASET_SIZE=$OVERRIDE_DATASET_SIZE
  fi

  dir_name=$(basename "$dir_path")
  concept="${dir_name//_/ }"

  if [[ "$dir_name" == *"_"* ]]; then
    TOKEN_OF_INTEREST="${dir_name##*_}"
  else
    TOKEN_OF_INTEREST="$dir_name"
  fi

  echo "Processing: $dir_name | Token: $TOKEN_OF_INTEREST | Split: $SPLIT"

  SAVE_FILENAME="qwen2_patched_image_${dir_name}_token_of_interest_concept_generation_split_${SPLIT}"

  HF_HOME="$HF_HOME" "$PYTHON_EXEC" "$SCRIPT_PATH" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --dataset_size "$DATASET_SIZE" \
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