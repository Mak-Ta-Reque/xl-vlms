export HF_HOME=/mnt/abka03/huggingface/hub

python src/save_features.py \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset_name "coco" \
    --dataset_size "4000" \
    --data_dir "/mnt/abka03/mscoco2014/" \
    --annotation_file  "/mnt/abka03/mscoco2014/dataset_coco.json" \
    --split "train" \
    --hook_name  "save_hidden_states_noun_phrase" \
    --modules_to_hook "model.norm" \
    --save_dir "/mnt/abka03/xl-vlms" \
    --save_filename "qwen2_image_coco_hidden_states" \
    --generation_mode \
    --save_only_generated_tokens \
    --slice_prediction \
    --exact_match_modules_to_hook \
    --cache_dir "/mnt/abka03/xl-vlms/cache"
