#!/bin/bash

# Set environment variable
export HF_HOME=/mnt/abka03/huggingface/hub

# Base paths
base_data_dir="/mnt/abka03/xlvlm_data/coco_crops/train"
feature_save_dir="/mnt/abka03/concept_extraction_result/MCoX/SNMF/coco_temp/train"
analysis_save_dir="/mnt/abka03/concept_extraction_result/MCoX/SNMF/coco_temp/train/concept"
cache_dir="/mnt/abka03/xl-vlms/cache"
model_name="Qwen/Qwen2-VL-7B-Instruct"
feature_module="model.norm"
hook_name="save_hidden_states_noun_phrase"
analysis_name="decompose_activations_text_grounding_image_grounding"
decomposition="snmf"
n_concepts=2
dataset_size="300"

# Loop through each concept folder
for dir in "$base_data_dir"/*; do
    # Get folder name and clean concept name
    folder_name=$(basename "$dir")
    concept="${folder_name//_/ }"  # Convert underscores to spaces

    echo "Processing concept: $concept"

    # === STEP 1: Save Features ===
    save_filename="qwen2_image_concept_hidden_states_${folder_name}"
    data_dir="${base_data_dir}/${folder_name}"

    echo "  - Saving features..."
    python src/save_features.py \
        --model_name "$model_name" \
        --dataset_name "image" \
        --dataset_size "$dataset_size" \
        --data_dir "$data_dir" \
        --hook_name "$hook_name" \
        --modules_to_hook "$feature_module" \
        --save_dir "$feature_save_dir" \
        --save_filename "$save_filename" \
        --generation_mode \
        --save_only_generated_tokens \
        --slice_prediction \
        --exact_match_modules_to_hook \
        --concept "$concept" \
        --cache_dir "$cache_dir"

    # === STEP 2: Analyze Features ===
    saved_features_path="${feature_save_dir}/features/${hook_name}_${save_filename}.pth"
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
done
