export HF_HOME=/mnt/abka03/xl-vlms/cache

python src/save_features.py \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset_name "coco" \
    --dataset_size "8000" \
    --data_dir "/mnt/abka03/mscoco2014/xl-vlm/combined_dataset_coco" \
    --annotation_file "/mnt/abka03/mscoco2014/xl-vlm/combined_dataset_coco_patch.json" \
    --split "train" \
    --hook_name "save_hidden_states_for_tokens_of_interest" \
    --tokens_of_interest " fur cat eye ears paw leg stripe long small" \
    --modules_to_hook "model.norm,model.layers.27" \
    --save_dir "/mnt/abka03/xl-vlms" \
    --save_filename "qwen2_image_patched_combined_grid_patch_concept_generation_split_train" \
    --generation_mode \
    --save_only_generated_tokens \
    --slice_prediction \
    --exact_match_modules_to_hook \
    --post_process_hidden \
    --cache_dir "/mnt/abka03/xl-vlms/cache"
