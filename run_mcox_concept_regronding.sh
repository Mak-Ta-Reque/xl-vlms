#!/bin/bash

# Set environment variable
#export HF_HOME=/mnt/abka03/huggingface/hub
export HF_HOME=/mnt/abka03/temp
# Base paths
base_data_dir="/mnt/abka03/xlvlm_data/no_idle_crops/train"
feature_save_dir="/mnt/abka03/concept_extraction_result/MCoX/SNMF/imagenet1000/train"
analysis_save_dir="/mnt/abka03/concept_extraction_result/MCoX/SNMF/imagenet1000/train/concept"
cache_dir="/mnt/abka03/xl-vlms/cache"
model_name="Qwen/Qwen2-VL-7B-Instruct"
feature_module="model.norm"
hook_name="save_hidden_states_sentence"
analysis_name="decompose_activations_text_grounding_image_grounding"
analysis_regrunding_name="redefine_activations_text_grounding"
decomposition="snmf" # Decomposition method
n_concepts=2 # Fixed for Contrastive SNMF
dataset_size="50"
nomalizations=("l1" "gl") #("l1" "zca" "l1zca" "l2" "l2zca" "gl")
# Loop through each concept folder
max_iterations=20
count=0


# === STEP 3: Combine Concepts ===


echo "Combining concepts with sudoCAV..."
python src/combine_concepts.py \
    --input_dir "$analysis_save_dir" \
    --output_path "$analysis_save_dir/combined_concept.pth" \
    --normalization "${nomalizations[@]}" \

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

echo "All steps completed successfully."