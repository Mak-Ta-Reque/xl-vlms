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
    --dataset_path "/mnt/abka03/Projects/xl-vlms/data/train" \
    --model_name "google/gemma-3n-E4B-it" \
    --output_csv "coco1000_concepts.csv" \
    --prompt "Identify and list all visible objects, items, textures, colors, materials, and notable visual elements in the given image. The output should be a single word comma-separated list without extra explanation, only the detected elements as word." \
    --batch_size 20 \
    --trust_remote_code \
    --image_size 512 512 \
    --image_budget 50

$PYTHON "$CONCEPT_MAPPING_SCRIPT" --input /mnt/abka03/Projects/xl-vlms/coco1000_concepts.csv --output /mnt/abka03/Projects/xl-vlms/coco_1000_concept_image_mapping.json