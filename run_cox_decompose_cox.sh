#!/bin/bash

# ========================
# Start timer
# ========================
start_time=$(date +%s)

# ========================
# Global Configuration
# ========================
MODEL_NAME="google/gemma-3n-E4B-it"
PYTHON_EXEC="/mnt/abka03/.conda/envs/xl_vlm/bin/python"
HF_HOME="/mnt/abka03/huggingface/hub"
COMBINE_SCRIPT="src/combine_features.py"
ANALYSIS_SCRIPT="src/analyse_features.py"

# ========================
# Dataset Configuration
# ========================
SPLIT="train"
SAVE_DIR="/mnt/abka03/concept_extraction_result/publish/gemma3n/CGDL/SNMF/imagenet1000/${SPLIT}"

# ========================
# Decomposition Configuration
# ========================
FEATURES_PATH="$SAVE_DIR/features/combined_features.pth"
ANALYSIS_NAME="decompose_activations_text_grounding_image_grounding"
MODULE_TO_DECOMPOSE="model.language_model.norm"
NUM_CONCEPTS=10
DECOMPOSITION_METHOD="sae2"
SAVE_FILENAME_ANALYSIS="gemma3n_results_patch_all_patch_sae"
CONCEPT_SAVE_DIR="$SAVE_DIR/concept"

# ========================
# Combine Features
# ========================
#echo "Combining all .pth files in $SAVE_DIR/features ..."
#"$PYTHON_EXEC" "$COMBINE_SCRIPT" "$SAVE_DIR/features"

# ========================
# Decomposition Analysis
# ========================
mkdir -p "$CONCEPT_SAVE_DIR"

echo "Running decomposition analysis..."
"$PYTHON_EXEC" "$ANALYSIS_SCRIPT" \
  --model_name "$MODEL_NAME" \
  --analysis_name "$ANALYSIS_NAME" \
  --features_path "$FEATURES_PATH" \
  --module_to_decompose "$MODULE_TO_DECOMPOSE" \
  --num_concepts "$NUM_CONCEPTS" \
  --decomposition_method "$DECOMPOSITION_METHOD" \
  --save_filename "$SAVE_FILENAME_ANALYSIS" \
  --save_dir "$CONCEPT_SAVE_DIR" \
  --load_matched_features

# ========================
# End Timer
# ========================
end_time=$(date +%s)
runtime=$((end_time - start_time))

# Format runtime to hh:mm:ss
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

echo "Total run time: ${hours}h ${minutes}m ${seconds}s"
