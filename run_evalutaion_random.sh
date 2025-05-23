#!/bin/bash
# Applicable for both method MCoX and CoX
# Variables
PYTHON_EXEC="/mnt/abka03/.conda/envs/xl_vlm/bin/python"
SCRIPT_PATH="${PWD}/src/analyse_features.py"

HF_HOME_PATH="/mnt/abka03/huggingface/hub"

ANALYSIS_NAME="concept_dictionary_evaluation_jaccard_clipscore_overlap_bertscore"
FEATURES_PATH="/mnt/abka03/concept_extraction_result/MCoX/SNMF/cifar100/val/features/combined_features.pth"
MODULE_TO_DECOMPOSE="model.norm"
MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
SAVE_FILENAME="cifar100random"
SAVE_DIR="/mnt/abka03/concept_extraction_result/MCoX/SNMF/cifar100/val/matrics"
ANALYSIS_SAVING_PATH="/mnt/abka03/concept_extraction_result/MCoX/SNMF/cifar100/train/concept/combined_concept_l2zca.pth"

# Use   //"--use_random_grounding_words" for random grounding words
# Export environment variable
export HF_HOME="$HF_HOME_PATH"

# Run the python script with all args
"$PYTHON_EXEC" "$SCRIPT_PATH" \
  --analysis_name "$ANALYSIS_NAME" \
  --features_path "$FEATURES_PATH" \
  --module_to_decompose "$MODULE_TO_DECOMPOSE" \
  --model_name "$MODEL_NAME" \
  --save_filename "$SAVE_FILENAME" \
  --local_files_only \
  --save_dir "$SAVE_DIR" \
  --use_random_grounding_words \
  --analysis_saving_path "$ANALYSIS_SAVING_PATH"
