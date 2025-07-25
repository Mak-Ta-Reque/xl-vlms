#!/bin/bash

# Configuration
MODEL_NAME="google/gemma-3n-E4B-it"
DATASET_NAME="text"
HOOK_NAME="save_hidden_states_sentence"
MODULES_TO_HOOK="model.language_model.norm"
PYTHON_EXEC="/mnt/abka03/.conda/envs/xl_vlm/bin/python"
SCRIPT_PATH="src/save_features.py"
HF_HOME="/mnt/abka03/huggingface/hub"

# Split-specific setup
SPLIT="val"
BASE_DATA_DIR="/mnt/abka03/xlvlm_data/dtd_processed/${SPLIT}"
SAVE_DIR="/mnt/abka03/concept_extraction_result/gemma3n/MCoX/SNMF/dtd/text_val/${SPLIT}"

# Loop through directories


max_iterations=47
count=0
for dir_path in "$BASE_DATA_DIR"/*/; do
  if (( count >= max_iterations )); then
    break
  fi
  dir_name=$(basename "$dir_path")
  echo "$dir_name"

  # Extract token of interest
  if [[ "$dir_name" == *"_"* ]]; then
    TOKEN_OF_INTEREST="${dir_name##*_}"
  else
    TOKEN_OF_INTEREST="$dir_name"
  fi

  # Run feature extraction
  concept="${dir_name//_/ }" 
  echo "Processing: $dir_name with token: $TOKEN_OF_INTEREST in split: $concept"
  SAVE_FILENAME="textonly_${dir_name}_token_of_interest_concept_generation_split_${SPLIT}"
##--token_of_interest " $TOKEN_OF_INTEREST" \
  HF_HOME="$HF_HOME" "$PYTHON_EXEC" "$SCRIPT_PATH" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --data_dir "$TOKEN_OF_INTEREST" \
    --hook_name "$HOOK_NAME" \
    --modules_to_hook "$MODULES_TO_HOOK" \
    --save_dir "$SAVE_DIR" \
    --save_filename "$SAVE_FILENAME" \
    --generation_mode \
    --save_only_generated_tokens \
    --slice_prediction \
    --exact_match_modules_to_hook

  count=$((count + 1))
done

# Combine feature files
COMBINE_SCRIPT="src/combine_features.py"
echo "Combining all .pth files in $SAVE_DIR/features ..."
"$PYTHON_EXEC" "$COMBINE_SCRIPT" "$SAVE_DIR/features"
