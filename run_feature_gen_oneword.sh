#!/bin/bash
set -euo pipefail  # Exit on errors and unset variables

##########################################
# Configuration
##########################################
MODEL_NAME="google/gemma-3n-E4B-it"
DATASET_NAME="image"
DEFAULT_DATASET_SIZE=800
OVERRIDE_DATASET_SIZE=800
HOOK_NAME="save_hidden_states_sentence"
MODULES_TO_HOOK="model.language_model.norm"
PYTHON_EXEC="/mnt/abka03/.conda/envs/xl_vlm/bin/python"
SCRIPT_PATH="src/save_features.py"
HF_HOME="/mnt/abka03/huggingface/hub"
PROMPT_TEMPLATE="oneword"

DATA_DIR="/mnt/abka03/xlvlm_data/oneclass/oneimage_crops"
SAVE_DIR="/mnt/abka03/concept_extraction_result/oneclass/gemma3n/MCoX/SNMF/rabbit/train"

##########################################
# Start total timer
##########################################
START_TIME=$(date +%s)

# Compute dataset size
FILE_COUNT=$(find "$DATA_DIR" -type f -name "*.png" | wc -l)
DATASET_SIZE="$FILE_COUNT"
if (( DATASET_SIZE > DEFAULT_DATASET_SIZE )); then
  DATASET_SIZE=$OVERRIDE_DATASET_SIZE
fi

# Determine concept: use first argument if provided, else derive from folder
DIR_NAME=$(basename "$DATA_DIR")
CONCEPT="${1:-${DIR_NAME//_/ }}"  # default to folder name if no argument provided

if [ -z "$CONCEPT" ]; then
  echo "Error: No concept provided and folder name is empty."
  exit 1
fi

SAVE_FILENAME="qwen2_patched_image_${DIR_NAME}_token_of_interest_concept_generation"

echo "Processing: $DIR_NAME (Concept: \"$CONCEPT\")"

##########################################
# Run Python script
##########################################
HF_HOME="$HF_HOME" "$PYTHON_EXEC" "$SCRIPT_PATH" \
  --model_name "$MODEL_NAME" \
  --dataset_name "$DATASET_NAME" \
  --dataset_size "$DATASET_SIZE" \
  --data_dir "$DATA_DIR" \
  --hook_name "$HOOK_NAME" \
  --modules_to_hook "$MODULES_TO_HOOK" \
  --token_of_interest "$CONCEPT" \
  --prompt_template "$PROMPT_TEMPLATE" \
  --save_dir "$SAVE_DIR" \
  --save_filename "$SAVE_FILENAME" \
  --generation_mode \
  --save_only_generated_tokens \
  --slice_prediction \
  --concept "$CONCEPT" \
  --exact_match_modules_to_hook

##########################################
# End timer and report
##########################################
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo -e "\n✅ Script completed in ${TOTAL_TIME}s (~$((TOTAL_TIME / 60)) minutes)."
