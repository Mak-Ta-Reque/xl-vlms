#!/bin/bash
set -euo pipefail  # Safer script: exit on errors and unset vars

##########################################
# Parse arguments
##########################################
MODEL_NAME=${1:-"google/gemma-3n-E4B-it"}
BASE_DATA_DIR=${2:-"/mnt/abka03/xlvlm_data/imagenet_1000_auto_crops/train"}
SAVE_DIR=${3:-"/mnt/abka03/concept_extraction_result/gemma3n/MCoX/SNMF/imagenet1000/train"}
HF_HOME=${4:-"/mnt/abka03/huggingface/hub"}
##########################################
# Configuration
##########################################
DATASET_NAME="image"
DEFAULT_DATASET_SIZE=800
OVERRIDE_DATASET_SIZE=800
HOOK_NAME="save_hidden_states_sentence"
MODULES_TO_HOOK="model.language_model.norm"
SCRIPT_PATH="src/save_features.py"
PROMPT_TEMPLATE="cgdl"
SPLIT="train"

MAX_ITERATIONS=2000000
COUNT=0

##########################################
# Start total timer
##########################################
START_TIME=$(date +%s)

for dir_path in "${BASE_DATA_DIR}"/*/; do
  if (( COUNT >= MAX_ITERATIONS )); then
    echo "Reached maximum number of iterations ($MAX_ITERATIONS), stopping."
    break
  fi

  FILE_COUNT=$(find "$dir_path" -type f -name "*.png" | wc -l)
  DATASET_SIZE="$FILE_COUNT"
  if (( DATASET_SIZE > DEFAULT_DATASET_SIZE )); then
    DATASET_SIZE=$OVERRIDE_DATASET_SIZE
  fi

  dir_name=$(basename "$dir_path")

  if [[ "$dir_name" == *"_"* ]]; then
    TOKEN_OF_INTEREST="${dir_name##*_}"
  else
    TOKEN_OF_INTEREST="$dir_name"
  fi

  concept="${dir_name//_/ }"
  SAVE_FILENAME="qwen2_patched_image_${dir_name}_token_of_interest_concept_generation_split_${SPLIT}"

  echo "[${COUNT}/${MAX_ITERATIONS}] Processing: $dir_name (Token: $TOKEN_OF_INTEREST, Concept: \"$concept\")"

  HF_HOME="$HF_HOME" python "$SCRIPT_PATH" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --dataset_size "$DATASET_SIZE" \
    --data_dir "$dir_path" \
    --hook_name "$HOOK_NAME" \
    --modules_to_hook "$MODULES_TO_HOOK" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --save_dir "$SAVE_DIR" \
    --save_filename "$SAVE_FILENAME" \
    --generation_mode \
    --save_only_generated_tokens \
    --slice_prediction \
    --concept "$concept" \
    --exact_match_modules_to_hook

  COUNT=$((COUNT + 1))
done

##########################################
# End total timer and report
##########################################
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo -e "\n✅ Script completed in ${TOTAL_TIME}s (~$((TOTAL_TIME / 60)) minutes)"
