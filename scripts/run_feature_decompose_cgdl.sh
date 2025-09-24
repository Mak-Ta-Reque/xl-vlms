#!/usr/bin/env bash

# ========================== #
#       CONFIGURATION        #
# ========================== #

set -euo pipefail

# Positional arguments with defaults (override order: CLI > ENV > hardcoded default)
HF_HOME_INPUT=${1:-${HF_HOME:-"/mnt/abka03/huggingface/hub"}}
MODEL_NAME_INPUT=${2:-${VLM_MODEL:-${MODEL_NAME:-"google/gemma-3n-E4B-it"}}}
FEATURE_SAVE_DIR_INPUT=${3:-${FEATURES_DIR:-${FEATURE_SAVE_DIR:-"/mnt/abka03/Projects/xl-vlms/outputs/cdgl/train"}}}
# Resolve Python interpreter
if command -v "${PYTHON:-${PYTHON_BIN:-python}}" >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON:-${PYTHON_BIN:-python}}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

if [ -z "$HF_HOME_INPUT" ] || [ -z "$MODEL_NAME_INPUT" ] || [ -z "$FEATURE_SAVE_DIR_INPUT" ]; then
    echo "Usage: $0 <HF_HOME> <model_name> <feature_save_dir>"
    exit 1
fi

start_time=$(date +%s)

# Export HF_HOME
export HF_HOME="$HF_HOME_INPUT"

# Paths
feature_save_dir="$FEATURE_SAVE_DIR_INPUT"
feature_dir="${feature_save_dir}/features"
# Respect orchestrator's DECOMP_DIR if provided; default to features/concept
analysis_save_dir="${DECOMP_DIR:-${feature_save_dir}/concept}"
cache_dir="/mnt/abka03/xl-vlms/cache"

# Model settings
model_name="$MODEL_NAME_INPUT"
feature_module="model.language_model.norm"
hook_name="save_hidden_states_sentence"

# Concept extraction settings
n_concepts=2
dataset_size="500000"
normalizations=("gl")
max_iterations=1000

# Decomposition methods to run (respect orchestrator-selected method if set)
if [[ -n "${DECOMP_METHOD:-}" ]]; then
    decomposition_methods=("${DECOMP_METHOD}")
else
    decomposition_methods=("snmf" "sae2" "pca" "simple")
fi

# Base analysis names
base_analysis_name="decompose_activations_text_grounding_image_grounding"
base_regrunding_name="redefine_activations_text_grounding"

# ========================== #
#         MAIN LOOP          #
# ========================== #

for decomposition in "${decomposition_methods[@]}"; do
    echo "=== Running decomposition: $decomposition ==="

    analysis_name="${base_analysis_name}_${decomposition}"
    analysis_regrunding_name="${base_regrunding_name}_${decomposition}"

    intermediate_dir="${analysis_save_dir}/intermediate_${decomposition}"
    mkdir -p "$intermediate_dir"

    count=0
    for saved_features_path in "$feature_dir"/*.pth; do
        if (( count >= max_iterations )); then break; fi

        filename=$(basename "$saved_features_path")
        folder_name="${filename%.*}"
        concept="${folder_name//_/ }"

        echo "Processing file: $filename"
        echo "  - Interpreted concept: $concept"

        results_filename="individual_concept_${folder_name}_${decomposition}"

        echo "  - Analyzing features..."
    "$PYTHON_BIN" src/analyse_features.py \
            --model_name "$model_name" \
            --analysis_name "$analysis_name" \
            --features_path "$saved_features_path" \
            --module_to_decompose "$feature_module" \
            --num_concepts "$n_concepts" \
            --decomposition_method "$decomposition" \
            --save_filename "$results_filename" \
            --save_dir "$intermediate_dir"

        count=$((count + 1))
    done

    echo "  - Combining concepts for $decomposition..."
    "$PYTHON_BIN" src/combine_concepts.py \
        --input_dir "$intermediate_dir" \
        --output_path "$analysis_save_dir/combined_concept_${decomposition}.pth" \
        --normalization "${normalizations[@]}"

    echo "  - Cleaning up intermediate files..."
    rm -rf "$intermediate_dir"

    for normalization in "${normalizations[@]}"; do
        echo "  - Regrounding concepts with $decomposition using normalization: $normalization"

    "$PYTHON_BIN" src/analyse_features.py \
            --model_name "$model_name" \
            --analysis_name "$analysis_regrunding_name" \
            --analysis_saving_path "${analysis_save_dir}/combined_concept_${decomposition}_raw.pth" \
            --module_to_decompose "$feature_module" \
            --decomposition_method "$decomposition" \
            --save_filename "combined_concept_${decomposition}_${normalization}_regrounded" \
            --save_dir "$analysis_save_dir" \
            --load_matched_features
    done

    echo "=== Completed decomposition: $decomposition ==="
    echo
done

# ========================== #
#       TIME REPORTING       #
# ========================== #

end_time=$(date +%s)
total_seconds=$((end_time - start_time))
hours=$((total_seconds / 3600))
minutes=$(((total_seconds % 3600) / 60))
seconds=$((total_seconds % 60))

echo "All steps completed successfully."
echo "Total execution time: ${hours}h ${minutes}m ${seconds}s"
