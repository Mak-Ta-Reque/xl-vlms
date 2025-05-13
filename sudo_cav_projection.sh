#!/bin/bash

# Set environment variable
export HF_HOME=/mnt/abka03/huggingface/hub

# List of concepts
concepts=("a parachute" "dog" "chainsaw" "french horn" "gas station" "cassette player" "charch" "fish" "garbage truck" "golf ball")  # Add or modify as needed

# Paths and constants
save_dir="/mnt/abka03/concept_extraction/concept"
analysis_name="decompose_activations_text_grounding_image_grounding"
model_name="Qwen/Qwen2-VL-7B-Instruct"
feature_module="model.norm"
decomposition="snmf"
n_concepts=2

# Loop through each concept and run the analysis
for concept in "${concepts[@]}"; do
    concept_clean=$(echo "$concept" | tr ' ' '_')

    saved_features_path="/mnt/abka03/concept_extraction/features/save_hidden_states_noun_phrase_qwen2_image_concept_hidden_states_${concept_clean}.pth"
    results_filename="phrase_embeddings_concepts_${concept_clean}"

    echo "Running analysis for concept: $concept"

    python src/analyse_features.py \
        --model_name "$model_name" \
        --analysis_name "$analysis_name" \
        --features_path "$saved_features_path" \
        --module_to_decompose "$feature_module" \
        --num_concepts "$n_concepts" \
        --decomposition_method "$decomposition" \
        --save_filename "$results_filename" \
        --save_dir "$save_dir"
done
