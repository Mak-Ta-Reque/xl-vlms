#!/bin/bash

# Set environment variable
export HF_HOME=/mnt/abka03/huggingface/hub

# Define list of concepts
concepts=("baby"  "baseball glove"  "bear"  "bus"  "car"  "cat"  "dog"  "fire hydrant"  "hot dog"  "microwave oven"  "school bus"  "stop sign"  "teddy bear"  "traffic light"  "train")  # Add more concepts as needed

# Paths
base_data_dir="/mnt/abka03/xlvlm_data/coco_crops/train"
save_dir="/mnt/abka03/concept_extraction_result/coco"
cache_dir="/mnt/abka03/xl-vlms/cache"
model_name="Qwen/Qwen2-VL-7B-Instruct"

# Loop through each concept
for concept in "${concepts[@]}"; do
    # Replace spaces with underscores for file/directory names
    concept_clean=$(echo "$concept" | tr ' ' '_')

    data_dir="${base_data_dir}/${concept_clean}"
    save_file_name="qwen2_image_concept_hidden_states_${concept_clean}"

    echo "Running for concept: $concept"

    python src/save_features.py \
        --model_name "$model_name" \
        --dataset_name "image" \
        --dataset_size "300" \
        --data_dir "$data_dir" \
        --hook_name "save_hidden_states_noun_phrase" \
        --modules_to_hook "model.norm" \
        --save_dir "$save_dir" \
        --save_filename "$save_file_name" \
        --generation_mode \
        --save_only_generated_tokens \
        --slice_prediction \
        --exact_match_modules_to_hook \
        --concept "$concept" \
        --cache_dir "$cache_dir"
done
