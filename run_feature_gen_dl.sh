#!/bin/bash

# ========================
# Start timer
# ========================
start_time=$(date +%s)

# ========================
# Global Configuration
# ========================
MODEL_NAME="google/gemma-3n-E4B-it"
DATASET_NAME="image"
HOOK_NAME="save_hidden_states_for_token_of_interest"
MODULES_TO_HOOK="model.language_model.norm"
PYTHON_EXEC="/mnt/abka03/.conda/envs/xl_vlm/bin/python"
HF_HOME="/mnt/abka03/huggingface/hub"
SCRIPT_PATH="src/save_features.py"
PROMT_TEMPLATE="dl"
SPLIT="val"
SAVE_DIR="/mnt/abka03/concept_extraction_result/publish/gemma3n/DL/SNMF/coco10/${SPLIT}"
MAX_ITERATIONS=10
DEFAULT_DATASET_SIZE=50
OVERRIDE_DATASET_SIZE=50
BASE_DATA_DIR="/mnt/abka03/xlvlm_data/coco_10_concepts/${SPLIT}"
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
