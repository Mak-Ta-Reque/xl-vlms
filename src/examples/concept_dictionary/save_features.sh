#!/bin/bash

# Path to your xl-vlms repository
#cd ~/xl-vlms
# Set the transformer cache dir
export HF_HOME=/netscratch/kadir/xl-vlms/cache

# --hook_name controls strategy of feature extraction
hook_name=save_hidden_states_for_token_of_interest  # Only save features for a token of interest in the output
#hook_name=save_hidden_states_given_token_start_end_idx   # Save all representations from start to end indices
#hook_name=save_hidden_states_given_token_idx   # Save representations at specific index

token=is
#token=dog # Can also use other nouns that appear in your dataset


#model_name=llava-hf/llava-1.5-7b-hf
#model_name=microsoft/llava-med-v1.5-mistral-7b
model_name=StanfordAIMI/CheXagent-8b
#model_name=allenai/Molmo-7B-D-0924
#model_name=HuggingFaceM4/idefics2-8b
#model_name=Qwen/Qwen2-VL-7B-Instruct

# Directory and filename to store extracted features
results_filename=CheXagent_is_generation_split_train
save_dir=/netscratch/kadir/xl-vlms

# Named modules inside the specific model for which you want to save the representations

# Examples of named modules for LLaVA-v1.5
#feature_modules=language_model.model.norm,language_model.model.layers.30
#feature_modules=language_model.model.layers.28.input_layernorm
#feature_modules=language_model.model.layers.29

# Examples of named modules for Molmo-7B
#feature_modules=model.transformer.blocks.27,model.transformer.ln_f

# Examples of named modules for Idefics2-8B
#feature_modules=model.text_model.norm,model.text_model.layers.30

# Examples of named modules for Qwen2-VL-7B
feature_modules=language_model.model.norm,language_model.model.layers.31


# Dataset specifications. Ensure you modify dataset path (--data_dir command) accordingly
data_dir=/netscratch/kadir/xl-vlms/sample_data # Data directory for COCO dataset
split=train # Which data split to save features for. For COCO: train/val/test options
size=5 #82783 # How many samples of dataset to consider. karpathy train split for COCO is of size 82783 images. Can't be more than dataset size
annotation_file=/home/kadir/xl-vlms/playground/organized_data.json


python src/save_features.py \
--model_name $model_name \
--dataset_name xray \
--dataset_size $size \
--data_dir $data_dir \
--annotation_file $annotation_file \
--split $split \
--hook_name $hook_name \
--modules_to_hook $feature_modules \
--select_token_of_interest_samples \
--token_of_interest $token \
--save_dir $save_dir \
--save_filename $results_filename \
--generation_mode \
--exact_match_modules_to_hook


# We save model representations on both train split (to learn the concepts) and test split (to evaluate)
split=val
size=2
results_filename=chexagent_is_generation_split_test


python src/save_features.py \
--model_name $model_name \
--dataset_name xray \
--dataset_size $size \
--data_dir $data_dir \
--annotation_file $annotation_file \
--split $split \
--hook_name $hook_name \
--modules_to_hook $feature_modules \
--select_token_of_interest_samples \
--token_of_interest $token \
--save_dir /netscratch/kadir/xl-vlms \
--save_filename $results_filename \
--generation_mode \
--exact_match_modules_to_hook
