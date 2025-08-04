#!/bin/bash

# ========================
# Start Timer
# ========================
start_time=$(date +%s)

# ========================
# Global Configuration
# ========================
MODEL_NAME="google/gemma-3n-E4B-it"
PYTHON_EXEC="/mnt/abka03/.conda/envs/xl_vlm/bin/python"
COMBINE_SCRIPT="src/combine_features.py"
ANALYSIS_SCRIPT="src/analyse_features.py"

# ========================
# Decomposition Configuration
# ========================
SPLIT="train"
BASE_SAVE_DIR="/mnt/abka03/concept_extraction_result/publish/gemma3n/DL/SNMF/coco10/${SPLIT}"
FEATURES_PATH="$BASE_SAVE_DIR/features/combined_features.pth"
MODULE_TO_DECOMPOSE="model.language_model.norm"
NUM_CONCEPTS=10

# ========================
# Methods List
# ========================
llsit=("snmf" "sae2" "pca" "simple")

# ========================
# Combine Features
# ========================
echo "Combining all .pth files in $BASE_SAVE_DIR/features ..."
"$PYTHON_EXEC" "$COMBINE_SCRIPT" "$BASE_SAVE_DIR/features"

# ========================
# Loop over methods
# ========================
for METHOD in "${llsit[@]}"; do
  echo "=============================================="
  echo "Running decomposition method: $METHOD"
  echo "=============================================="

  CONCEPT_SAVE_DIR="$BASE_SAVE_DIR/concept/"
  mkdir -p "$CONCEPT_SAVE_DIR"

  ANALYSIS_NAME_FULL="decompose_activations_text_grounding_image_grounding_"
  SAVE_FILENAME_ANALYSIS="gemma3n_results_patch_all_patch_${METHOD}"

  "$PYTHON_EXEC" "$ANALYSIS_SCRIPT" \
    --model_name "$MODEL_NAME" \
    --analysis_name "$ANALYSIS_NAME_FULL" \
    --features_path "$FEATURES_PATH" \
    --module_to_decompose "$MODULE_TO_DECOMPOSE" \
    --num_concepts "$NUM_CONCEPTS" \
    --decomposition_method "$METHOD" \
    --save_filename "$SAVE_FILENAME_ANALYSIS" \
    --save_dir "$CONCEPT_SAVE_DIR" \
    --load_matched_features
done

# ========================
# Clean Up Combined Feature File
# ========================
echo "Deleting combined feature file..."
rm -f "$FEATURES_PATH"

# ========================
# End Timer
# ========================
end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

echo "All decompositions completed in ${hours}h ${minutes}m ${seconds}s"
