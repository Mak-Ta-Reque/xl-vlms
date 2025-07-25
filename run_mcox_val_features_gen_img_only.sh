#!/bin/bash

# Configuration
MODEL_NAME="google/gemma-3n-E4B-it"
DATASET_NAME="image"
DATASET_SIZE="20"
HOOK_NAME="save_hidden_states_sentence"
MODULES_TO_HOOK="model.language_model.norm"
PYTHON_EXEC="/mnt/abka03/.conda/envs/xl_vlm/bin/python"
SCRIPT_PATH="src/save_features.py"
HF_HOME="/mnt/abka03/huggingface/hub"

# Split-specific setup
SPLIT="val"
BASE_DATA_DIR="/mnt/abka03/xlvlm_data/dtd_processed/${SPLIT}"
SAVE_DIR="/mnt/abka03/concept_extraction_result/gemma3n/MCoX/SNMF/dtd/image_only/${SPLIT}"

mapfile -t shuffled_dirs < <(find "$BASE_DATA_DIR" -mindepth 1 -maxdepth 1 -type d | shuf)

# Convert shuffled dir names to an array of concepts
concepts=()
for path in "${shuffled_dirs[@]}"; do
  name=$(basename "$path")
  concepts+=("${name//_/ }")
done

# Reset loop index
i=0
max_iterations=47
count=0

# Loop through directories
for dir_path in "$BASE_DATA_DIR"/*/; do
  if (( count >= max_iterations )); then
    break
  fi
  dir_name=$(basename "$dir_path")

  # Extract token of interest
  if [[ "$dir_name" == *"_"* ]]; then
    TOKEN_OF_INTEREST="${dir_name##*_}"
  else
    TOKEN_OF_INTEREST="$dir_name"
  fi

  # Use a random concept (non-repeating)
  concept="${concepts[$i]}"
  ((i++))
  if [[ "$concept" == "${dir_name//_/ }" ]]; then
    continue
  fi

  echo "Processing '$concept' with token '$TOKEN_OF_INTEREST'"
  SAVE_FILENAME="qwen2_patched_image_${dir_name}_token_of_interest_concept_generation_split_${SPLIT}"
##--token_of_interest " $TOKEN_OF_INTEREST" \
  HF_HOME="$HF_HOME" "$PYTHON_EXEC" "$SCRIPT_PATH" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --dataset_size "$DATASET_SIZE" \
    --data_dir "$dir_path" \
    --hook_name "$HOOK_NAME" \
    --modules_to_hook "$MODULES_TO_HOOK" \
    --save_dir "$SAVE_DIR" \
    --save_filename "$SAVE_FILENAME" \
    --generation_mode \
    --save_only_generated_tokens \
    --slice_prediction \
    --concept "$concept" \
    --exact_match_modules_to_hook
  count=$((count + 1))
done

# Combine feature files
COMBINE_SCRIPT="src/combine_features.py"
echo "Combining all .pth files in $SAVE_DIR/features ..."
"$PYTHON_EXEC" "$COMBINE_SCRIPT" "$SAVE_DIR/features"
