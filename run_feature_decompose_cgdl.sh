#!/bin/bash

# ========================== #
#       CONFIGURATION        #
# ========================== #

start_time=$(date +%s)
export HF_HOME="/mnt/abka03/huggingface/hub"

# Paths
feature_save_dir="/mnt/abka03/concept_extraction_result/gemma3n/MCoX/SNMF/imagenet3/train"
feature_dir="${feature_save_dir}/features"
analysis_save_dir="${feature_save_dir}/concept"
cache_dir="/mnt/abka03/xl-vlms/cache"

# Model settings
model_name="google/gemma-3n-E4B-it"
feature_module="model.language_model.norm"
hook_name="save_hidden_states_sentence"

# Concept extraction settings
n_concepts=2
dataset_size="800"
normalizations=("gl")
max_iterations=5

# Decomposition methods to run
decomposition_methods=("snmf" "sae2" "pca" "simple")

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

    # Create intermediate folder for this decomposition
    intermediate_dir="${analysis_save_dir}/intermediate_${decomposition}"
    mkdir -p "$intermediate_dir"

    count=0
    for saved_features_path in "$feature_dir"/*.pth; do
        if (( count >= max_iterations )); then break; fi

        filename=$(basename "$saved_features_path")
        folder_name="${filename%.*}"  # Remove .pth extension
        concept="${folder_name//_/ }"

        echo "Processing file: $filename"
        echo "  - Interpreted concept: $concept"

        # Rename individual concept file prefix here:
        results_filename="individual_concept_${folder_name}_${decomposition}"

        echo "  - Analyzing features..."
        python src/analyse_features.py \
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

    # === Combine Concepts ===
    echo "  - Combining concepts for $decomposition..."
    python src/combine_concepts.py \
        --input_dir "$intermediate_dir" \
        --output_path "$analysis_save_dir/combined_concept_${decomposition}.pth" \
        --normalization "${normalizations[@]}"

    # === Delete Intermediate Folder ===
    echo "  - Cleaning up intermediate files..."
    rm -rf "$intermediate_dir"

    # === Regrounding ===
    for normalization in "${normalizations[@]}"; do
        echo "  - Regrounding concepts with $decomposition using normalization: $normalization"

        python src/analyse_features.py \
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
