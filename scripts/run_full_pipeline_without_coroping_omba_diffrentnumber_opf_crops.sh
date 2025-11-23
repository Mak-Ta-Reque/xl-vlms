#!/usr/bin/env bash
# Single-script XL-VLMs pipeline: defines constants at top and calls Python directly (no nested bash wrappers).
# Steps:
# 1) Dataset inference -> concepts map
# 2) Build crops JSON from concept→image mapping
# 3) Generate features from crops JSON (on-the-fly cropping)
# 4) Decompose features (one or more methods)
# 5) Run VLM explainer per method
# 6) Concept deletion eval per method
# 7) (Optional) Plots per method + summary

set -Eeuo pipefail

# -------------------------------
# Utility
# -------------------------------
usage() {
  cat <<USAGE
Usage: $(basename "$0") [--input-dir PATH] [--output-dir PATH] [--decomp METHODS] [--plot-ymin VAL] [--plot-ymax VAL]

Options:
  --input-dir PATH     Root dataset/images directory. Default: ${INPUT_DIR:-/mnt/sdz/abka03_data/xl-vlms/data}
  --output-dir PATH    Root output directory. Default: ${OUTPUT_DIR:-$PWD/outputs/run_<timestamp>}
  --decomp METHODS     Comma-separated decomposition methods. Default: ${DECOMP_METHODS:-snmf}
  --plot-ymin VAL      Y-axis min for plots (default: ${PLOT_YMIN:-6.55e-6})
  --plot-ymax VAL      Y-axis max for plots (default: ${PLOT_YMAX:-7.10e-6})
  -h, --help           Show this help

Env overrides supported for all constants below.
USAGE
}

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }
warn() { echo "[$(ts)] [WARN] $*" >&2; }

run_step() {
  local name="$1"; shift
  local cmd="$*"
  local logfile="$OUTPUT_DIR/logs/${name// /_}.log"
  log "START: ${name}"
  {
    eval "$cmd" 2>&1 | tee -a "$logfile"
  }
  log "DONE:  ${name}"
}

trap 'rc=$?; warn "Pipeline failed at line $LINENO (exit $rc). Check logs at: $OUTPUT_DIR/logs"; exit $rc' ERR

# -------------------------------
# Paths and constants (override via env)
# -------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Data & outputs
INPUT_DIR="${INPUT_DIR:-/mnt/abka03/xlvlm_data/dtd_split}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/dtd_concepts/qwen2_5/imnet_colorful_eval_16_grid}"
HF_HOME="${HF_HOME:-/mnt/abka03/huggingface/hub}"
#
#Qwen/Qwen2.5-VL-7B-Instruct
#google/gemma-3n-E4B-it
# Model/runtime knobs
VLM_MODEL="${VLM_MODEL:-google/gemma-3n-E4B-it}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-42}"
DEVICE_ID="${DEVICE_ID:-0}"   # default to GPU 1; override with DEVICE_ID or CUDA_VISIBLE_DEVICES

# Crops JSON generation
CONCEPT_CROPS_PER_IMAGE="${CONCEPT_CROPS_PER_IMAGE:-30}"
PATCH_SIZE="${PATCH_SIZE:-200}"
MIN_IMAGES_PER_TAG="${MIN_IMAGES_PER_TAG:-50}" # This the minimum frequency of objects to be selected for cropping. Bag sisze < image per tag
MAX_IMAGES_PER_TAG="${MAX_IMAGES_PER_TAG:-600}" #  This the upperlimit of images going to crop for further processing 
BAG_SIZE="${BAG_SIZE:-30}" # max number of crop/image chosen for each concept bag,
PATCHES_PER_IMAGE="${PATCHES_PER_IMAGE:-40}"
CONCEPT_MODE="${CONCEPT_MODE:-1}"              # 1: concept-focused k crops/image; 0: random/grid modes
OBJECT_DETECTION="${OBJECT_DETECTION:-1}"      # 1 to enable LangSAM
DETECTION_BATCH_SIZE="${DETECTION_BATCH_SIZE:-2}" # Careful if gpu overflows it goes to 1 obejct per image 
DETECTION_TOPN="${DETECTION_TOPN:-2}"

# Inference prompt and image preproc
PROMPT="${PROMPT:-Identify all visible objects, items, entities, materials, textures, colors, shapes, symbols, text, scenes, actions, and visual patterns present in the image at the most detailed and fine-grained level possible. Include every distinguishable element or concept, even small or background details. Output only a strict, single-word, comma-separated list with no sentences, no explanations, and no extra text.}"
IMAGE_SIZE="${IMAGE_SIZE:-512 512}"             # two ints
IMAGE_BUDGET="${IMAGE_BUDGET:-300}"            # per-subfolder budget

# Decomposition methods
DECOMP_METHODS="${DECOMP_METHODS:-snmf}"

# Explainer/Eval
LAYER_PATH="${LAYER_PATH:-model.language_model.norm}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/abka03/xlvlm_data/colorful/val_grids}"
TOP_N="${TOP_N:-3}"
NUM_POINTS="${NUM_POINTS:-70}"
EXPL_PROMPT_MODE="${EXPL_PROMPT_MODE:-unsupervised}"
EXPL_LABEL="${EXPL_LABEL:-}"
EXPL_CHOICES="${EXPL_CHOICES:-}"

# Plot
PLOT_YMIN="${PLOT_YMIN:-6.5500e-6}"
PLOT_YMAX="${PLOT_YMAX:-7.1000e-6}"

# -------------------------------
# CLI
# -------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)  INPUT_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --decomp)     DECOMP_METHODS="$2"; shift 2;;
    --plot-ymin)  PLOT_YMIN="$2"; shift 2;;
    --plot-ymax)  PLOT_YMAX="$2"; shift 2;;
    -h|--help)    usage; exit 0;;
    *) warn "Unknown arg: $1"; usage; exit 1;;
  esac
done


# -------------------------------
# Prepare directories
# -------------------------------
mkdir -p "$OUTPUT_DIR"/logs
mkdir -p "$OUTPUT_DIR"/inference "$OUTPUT_DIR"/features "$OUTPUT_DIR"/concept \
         "$OUTPUT_DIR"/explanations "$OUTPUT_DIR"/eval "$OUTPUT_DIR"/plots

CONCEPT_MAP_JSON="$OUTPUT_DIR/inference/concepts_to_images.json"
CROPS_JSON="$OUTPUT_DIR/inference/crops.json"
FEATURES_DIR="$OUTPUT_DIR"         # save_features.py controls internal layout
DECOMP_DIR="$OUTPUT_DIR/concept"
EXPLAIN_DIR="$OUTPUT_DIR/explanations"
EVAL_DIR="$OUTPUT_DIR/eval"
PLOTS_DIR="$OUTPUT_DIR/plots"

export HF_HOME
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEVICE_ID}"

log "Root:       $ROOT_DIR"
log "Input dir:  $INPUT_DIR"
log "Output dir: $OUTPUT_DIR"
log "Model:      $VLM_MODEL | Batch: $BATCH_SIZE | Seed: $SEED | Device: cuda:$CUDA_VISIBLE_DEVICES"
log "Decompose:  $DECOMP_METHODS"
log "Crops: data_dir=$INPUT_DIR k=$CONCEPT_CROPS_PER_IMAGE patch=$PATCH_SIZE min=$MIN_IMAGES_PER_TAG max=$MAX_IMAGES_PER_TAG concept_mode=$CONCEPT_MODE det=$OBJECT_DETECTION"
log "Explainer:  layer=$LAYER_PATH image_root=$IMAGE_ROOT top_n=$TOP_N mode=$EXPL_PROMPT_MODE"
log "Plots Y:    [$PLOT_YMIN, $PLOT_YMAX]"

# -------------------------------
# 1) Dataset inference -> concept map
# -------------------------------
OBJECTS_CSV="$OUTPUT_DIR/inference/objects.csv"
if [[ -s "$OBJECTS_CSV" && -s "$CONCEPT_MAP_JSON" ]]; then
  log "Skip Dataset Inference (found $OBJECTS_CSV and $CONCEPT_MAP_JSON)"
else
  mkdir -p "$(dirname "$OBJECTS_CSV")"
  # dataset_inference.py expects --image_size as two ints; pass via eval-expanded string
  run_step "Dataset Inference" \
    "python -u \"$ROOT_DIR/inference/dataset_inference.py\" \
      --dataset_path \"$INPUT_DIR/train\" \
      --model_name \"$VLM_MODEL\" \
      --output_csv \"$OBJECTS_CSV\" \
      --prompt \"$PROMPT\" \
      --batch_size \"$BATCH_SIZE\" \
      --image_size $IMAGE_SIZE \
      --image_budget \"$IMAGE_BUDGET\" \
      --trust_remote_code"

  run_step "Build Concept Map" \
    "python -u \"$ROOT_DIR/concept_image_mapping.py\" --input \"$OBJECTS_CSV\" --output \"$CONCEPT_MAP_JSON\""
fi

# -------------------------------
# 2) Build crops JSON from concept→image map
# -------------------------------
if [[ -s "$CROPS_JSON" ]]; then
  log "Skip Crops JSON (found $CROPS_JSON)"
else
  mkdir -p "$(dirname "$CROPS_JSON")"
  # Build flag strings
  CONCEPT_FLAG=$([[ "$CONCEPT_MODE" == "1" ]] && echo "--concept_mode --concept_crops_per_image $CONCEPT_CROPS_PER_IMAGE" || echo "")
  DETECT_FLAG=$([[ "$OBJECT_DETECTION" == "1" ]] && echo "--object_detection --batch_size $DETECTION_BATCH_SIZE --topn $DETECTION_TOPN" || echo "")
  run_step "Crops JSON" \
    "python -u \"$ROOT_DIR/preprocessing/crops_to_json.py\" \
      --input_root \"$INPUT_DIR\" \
      --json_mapping \"$CONCEPT_MAP_JSON\" \
      --output_json \"$CROPS_JSON\" \
      --patch_size \"$PATCH_SIZE\" \
      --patches_per_image \"$PATCHES_PER_IMAGE\" \
      --min_images_per_tag \"$MIN_IMAGES_PER_TAG\" \
      --max_images_per_tag \"$MAX_IMAGES_PER_TAG\" \
      --seed \"$SEED\" \
      --device cuda:1 \
      $CONCEPT_FLAG \
      $DETECT_FLAG"
fi

# -------------------------------
# 3) Generate features from crops JSON
# -------------------------------
if find "$FEATURES_DIR/features" -type f -name '*.pth' -print -quit | grep -q .; then
  log "Skip Feature Generation (found features under $FEATURES_DIR/features)"
else
run_step "Generate Features" \
  "HF_HOME=\"$HF_HOME\" python -u \"$ROOT_DIR/src/save_features.py\" \
    --model_name \"$VLM_MODEL\" \
    --dataset_name json_crop_map \
    --dataset_size \"$BAG_SIZE\" \
    --data_dir \"$INPUT_DIR\" \
    --annotation_file \"$CROPS_JSON\" \
    --split train \
    --hook_names save_hidden_states_mean \
    --modules_to_hook $LAYER_PATH \
    --prompt_template cgdl \
    --save_dir \"$FEATURES_DIR\" \
    --batch_size \"$BATCH_SIZE\" \
    --generation_mode \
    --save_only_generated_tokens \
    --exact_match_modules_to_hook"
fi

# -------------------------------
# 4) Decompose features across methods (direct Python)
# -------------------------------
IFS=',' read -r -a DECOMP_ARRAY <<< "$DECOMP_METHODS"
for method in "${DECOMP_ARRAY[@]}"; do
  out_raw="$DECOMP_DIR/$method/combined_concept_${method}_raw.pth"
  mkdir -p "$DECOMP_DIR/$method"
  if [[ -s "$out_raw" ]]; then
    log "Skip Decompose ($method) (found $out_raw)"
    continue
  fi

  # Analyse each feature file
  base_analysis_name="decompose_activations_text_grounding_image_grounding"
  feature_module="$LAYER_PATH"
  n_concepts="${n_concepts:-2}"
  DL_ALPHA="${DL_ALPHA:-20}"

  run_step "Decompose:$method (batch)" \
    "python -u \"$ROOT_DIR/src/analyse_features.py\" \
      --model_name \"$VLM_MODEL\" \
      --analysis_name \"${base_analysis_name}_${method}\" \
      --features_path \"$FEATURES_DIR/features\" \
      --module_to_decompose \"$feature_module\" \
      --num_concepts \"$n_concepts\" \
      --dl_alpha \"$DL_ALPHA\" \
      --decomposition_method \"$method\" \
      --save_dir \"$DECOMP_DIR/$method/intermediate_${method}\""

  mkdir -p "$DECOMP_DIR/$method"
  run_step "Combine Concepts ($method)" \
    "python -u \"$ROOT_DIR/src/combine_concepts.py\" \
      --input_dir \"$DECOMP_DIR/$method/intermediate_${method}\" \
      --output_path \"$DECOMP_DIR/$method/combined_concept_${method}.pth\" \
      --normalization gl"

  # Regrounding
  run_step "Reground Concepts ($method)" \
    "python -u \"$ROOT_DIR/src/analyse_features.py\" \
      --model_name \"$VLM_MODEL\" \
      --analysis_name \"redefine_activations_text_grounding_${method}\" \
      --analysis_saving_path \"$DECOMP_DIR/$method/combined_concept_${method}_raw.pth\" \
      --module_to_decompose \"$feature_module\" \
      --decomposition_method \"$method\" \
      --save_filename \"combined_concept_${method}_gl_regrounded\" \
      --save_dir \"$DECOMP_DIR/$method\" \
      --load_matched_features"

  # Cleanup
  rm -rf "$DECOMP_DIR/$method/intermediate_${method}" || true
done

# -------------------------------
# 5) VLM explainer per method
# -------------------------------
for method in "${DECOMP_ARRAY[@]}"; do
  concept_path="$DECOMP_DIR/${method}/combined_concept_${method}_raw.pth"
  out_dir="$EXPLAIN_DIR/$method"; mkdir -p "$out_dir"
  out_json="$out_dir/vlm_explanations.json"
  if [[ -s "$out_json" ]]; then
    log "Skip Explainer ($method) (found $out_json)"
    continue
  fi
  EXTRA_PROMPT_ARGS="--prompt_mode $EXPL_PROMPT_MODE"
  [[ -n "$EXPL_LABEL" ]] && EXTRA_PROMPT_ARGS+=" --prompt_label \"$EXPL_LABEL\""
  [[ -n "$EXPL_CHOICES" ]] && EXTRA_PROMPT_ARGS+=" --choices \"$EXPL_CHOICES\""
  run_step "Explainer ($method)" \
    "HF_HOME=\"$HF_HOME\" python -u \"$ROOT_DIR/inference/vlm_explainer_multibatch.py\" \
      --model_name \"$VLM_MODEL\" \
      --concept_path \"$concept_path\" \
      --layer_path \"$LAYER_PATH\" \
      --image_root \"$IMAGE_ROOT\" \
      --top_n \"$TOP_N\" \
      --out_json \"$out_json\" \
      $EXTRA_PROMPT_ARGS"
done
