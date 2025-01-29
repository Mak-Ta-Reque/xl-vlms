#!/bin/bash

# Path to your xl-vlms repository
#cd ~/xl-vlms
export HF_HOME=/netscratch/kadir/xl-vlms/cache
#model_name=llava-hf/llava-1.5-7b-hf
#model_name=allenai/Molmo-7B-D-0924
#model_name=HuggingFaceM4/idefics2-8b
#model_name=Qwen/Qwen2-VL-7B-Instruct
model_name=StanfordAIMI/CheXagent-8b

# Need to specify analysis type. This will decompose features, and extract both text, visual grounding
analysis_name=decompose_activations_text_grounding_image_grounding

# Specify path where you have saved features on training data
saved_features_path=/netscratch/kadir/xl-vlms/features/save_hidden_states_for_token_of_interest_CheXagent_is_generation_split_train.pth

# Where to store details about extracted concepts. Default directory is results/
results_filename=results_is_chexagent

# Specify the specific feature module you want to decompose

# LLaVA-v1.5
#feature_module=language_model.model.norm
#feature_module=language_model.model.layers.30

# Molmo-7B
#feature_module=model.transformer.blocks.27
#feature_module=model.transformer.ln_f
save_dir=/netscratch/kadir/xl-vlms/results
# Idefics2-8B
#feature_module=model.text_model.norm

# Qwen2-VL-7B
feature_module=language_model.model.norm

decomposition=snmf # Current options: snmf, kmeans, pca, simple
n_concepts=2 # Size of dictionary learnt i.e. number of concepts

python src/analyse_features.py \
--model_name $model_name \
--analysis_name $analysis_name \
--features_path $saved_features_path \
--module_to_decompose $feature_module \
--num_concepts $n_concepts \
--decomposition_method $decomposition \
--save_filename $results_filename \
--save_dir $save_dir
