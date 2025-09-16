#!/bin/bash

# Set environment variables
export TORCH_COMPILE_DISABLE=1
export HF_HOME="/mnt/abka03/huggingface/hub"
export PYTHONPATH="${PWD}"

# Python executable
PYTHON="/mnt/abka03/.conda/envs/rsml/bin/python"

# Program path
PROGRAM="${PWD}/inference/dataset_inference.py"
CONCEPT_MAPPING_SCRIPT="${PWD}/concept_image_mapping.py"
# Run the program
$PYTHON "$PROGRAM" \
    --dataset_path "/mnt/abka03/xlvlm_data/imagenet_1000/train" \
    --model_name "google/gemma-3n-E4B-it" \
    --output_csv "/mnt/abka03/xlvlm_data/imagenet_1000/coco1000_concepts.csv" \
    --prompt "Identify all visible objects, items, textures, colors, materials, and notable visual patterns in the given image. Output only a single-word, comma-separated list. Do not include explanations, sentences, or any extra text—just the detected elements." \
    --batch_size 25 \
    --trust_remote_code \
    --image_size 256 256 \
    --image_budget 50

$PYTHON "$CONCEPT_MAPPING_SCRIPT" --input /mnt/abka03/xlvlm_data/imagenet_1000/coco1000_concepts.csv --output /mnt/abka03/xlvlm_data/imagenet_1000/coco_1000_concept_image_mapping.json