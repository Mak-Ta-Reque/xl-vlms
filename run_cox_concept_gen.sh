#!/bin/bash

# Configuration
MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
DATASET_NAME="image"
DATASET_SIZE="50" #Must be changed based ondataset
HOOK_NAME="save_hidden_states_for_token_of_interest"
MODULES_TO_HOOK="model.norm"
PYTHON_EXEC="/mnt/abka03/.conda/envs/xl_vlm/bin/python"
SCRIPT_PATH="src/save_features.py"
HF_HOME="/mnt/abka03/huggingface/hub"

# Split-specific setup
SPLIT="train"
BASE_DATA_DIR="/mnt/abka03/xlvlm_data/noise10concept/${SPLIT}" #Must be changed based on dataset
SAVE_DIR="/mnt/abka03/concept_extraction_result/CoX/SNMF/noisy10/${SPLIT}" #Must be changed based ondataset

# Loop through directories
for dir_path in "$BASE_DATA_DIR"/*/; do
  dir_name=$(basename "$dir_path")
  concept="${dir_name//_/ }" # Convert underscores to spaces
  # Extract token of interest
  if [[ "$dir_name" == *"_"* ]]; then
    TOKEN_OF_INTEREST="${dir_name##*_}"
  else
    TOKEN_OF_INTEREST="$dir_name"
  fi

  # Run feature extraction
  echo "Processing: $dir_name with token: $TOKEN_OF_INTEREST in split: $SPLIT"
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
    --save_only_generated_tokens \
    --slice_prediction \
    --concept "$concept" \
    --exact_match_modules_to_hook
done

# Combine feature files
COMBINE_SCRIPT="src/combine_features.py"
echo "Combining all .pth files in $SAVE_DIR/features ..."
"$PYTHON_EXEC" "$COMBINE_SCRIPT" "$SAVE_DIR/features"

# Run decomposition analysis
ANALYSIS_SCRIPT="src/analyse_features.py"
FEATURES_PATH="$SAVE_DIR/features/combined_features.pth"
ANALYSIS_NAME="decompose_activations_text_grounding_image_grounding"
MODULE_TO_DECOMPOSE="model.norm"
NUM_CONCEPTS=10 # Must be changed based on dataset
DECOMPOSITION_METHOD="snmf"
SAVE_FILENAME="qwen2_results_patch_all_patch_snmf"
CONCEPT_SAVE_DIR="$SAVE_DIR/concept"

mkdir -p "$CONCEPT_SAVE_DIR"

echo "Running decomposition analysis for training split..."
"$PYTHON_EXEC" "$ANALYSIS_SCRIPT" \
  --model_name "$MODEL_NAME" \
  --analysis_name "$ANALYSIS_NAME" \
  --features_path "$FEATURES_PATH" \
  --module_to_decompose "$MODULE_TO_DECOMPOSE" \
  --num_concepts "$NUM_CONCEPTS" \
  --decomposition_method "$DECOMPOSITION_METHOD" \
  --save_filename "$SAVE_FILENAME" \
  --save_dir "$CONCEPT_SAVE_DIR" \
  --load_matched_features
