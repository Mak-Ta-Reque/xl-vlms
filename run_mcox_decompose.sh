#!/bin/bash

# Record start time
start_time=$(date +%s)

# Set environment variable
export HF_HOME=/mnt/abka03/huggingface/hub

# Base paths
base_data_dir="/mnt/abka03/xlvlm_data/imagenet_1000_concepts_crop/train"
feature_save_dir="/mnt/abka03/concept_extraction_result/publish/gemma3n/CGDL/SNMF/imagenet1000/train"
analysis_save_dir="/mnt/abka03/concept_extraction_result/publish/gemma3n/CGDL/SNMF/imagenet1000/train/concept"
cache_dir="/mnt/abka03/xl-vlms/cache"
model_name="google/gemma-3n-E4B-it"
feature_module="model.language_model.norm"
hook_name="save_hidden_states_sentence"
analysis_name="decompose_activations_text_grounding_image_grounding_simple"
analysis_regrunding_name="redefine_activations_text_grounding"
decomposition="simple"
n_concepts=2
dataset_size="300"
nomalizations=("gl")
max_iterations=10
count=0

# === STEP 2: Analyze Features ===
for dir in "$base_data_dir"/*; do
    if (( count >= max_iterations )); then
        break
    fi
    folder_name=$(basename "$dir")
    concept="${folder_name//_/ }"

    echo "Processing concept: $concept"

    saved_features_path="${feature_save_dir}/features/${hook_name}_qwen2_image_concept_hidden_states_${folder_name}.pth"
    results_filename="phrase_embeddings_concepts_${folder_name}"

    echo "  - Analyzing features..."
    python src/analyse_features.py \
        --model_name "$model_name" \
        --analysis_name "$analysis_name" \
        --features_path "$saved_features_path" \
        --module_to_decompose "$feature_module" \
        --num_concepts "$n_concepts" \
        --decomposition_method "$decomposition" \
        --save_filename "$results_filename" \
        --save_dir "$analysis_save_dir"
    
    count=$((count + 1))
done

# === STEP 3: Combine Concepts ===
echo "Combining concepts with sudoCAV..."
python src/combine_concepts.py \
    --input_dir "$analysis_save_dir" \
    --output_path "$analysis_save_dir/combined_concept.pth" \
    --normalization "${nomalizations[@]}"

echo "  - Text regrounding..."
for normalization in "${nomalizations[@]}"; do
    echo "  - Regrounding concepts with normalization: $normalization"
    python src/analyse_features.py \
        --model_name "$model_name" \
        --analysis_name "$analysis_regrunding_name" \
        --analysis_saving_path "${analysis_save_dir}/combined_concept_${normalization}.pth" \
        --module_to_decompose "$feature_module" \
        --decomposition_method "$decomposition" \
        --save_filename "combined_concept_${normalization}_regrounded" \
        --save_dir "$analysis_save_dir" \
        --load_matched_features
done

# Record end time and calculate duration
end_time=$(date +%s)
total_seconds=$((end_time - start_time))

# Format and display time
hours=$((total_seconds / 3600))
minutes=$(((total_seconds % 3600) / 60))
seconds=$((total_seconds % 60))

echo "All steps completed successfully."
echo "Total execution time: ${hours}h ${minutes}m ${seconds}s"
