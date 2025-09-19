#!/usr/bin/env bash
# Orchestrates the full XL-VLMs pipeline end-to-end with logging.
# Steps:
# 1) Dataset inference -> concepts map
# 2) Crop images
# 3) Generate concept features
# 4) Decompose features (multiple methods)
# 5) Run VLM explainer
# 6) Concept deletion evaluation
# 7) (Optional) Plot final figures

set -Eeuo pipefail

# -------------------------------
# Utility
# -------------------------------
usage() {
  cat <<'USAGE'
Usage: run_full_pipeline.sh [--input-dir PATH] [--output-dir PATH] [--decomp METHODS] [--plot-ymin VAL] [--plot-ymax VAL]

Options:
  --input-dir PATH     Root dataset/images directory to process. Default: $INPUT_DIR or ../data
  --output-dir PATH    Root output directory. Default: ../outputs/run_<timestamp>
  --decomp METHODS     Comma-separated decomposition methods. Default: pca,nmf,ica,svd
  --plot-ymin VAL     Y-axis min for plots (default: ${PLOT_YMIN:-3.62e-6})
  --plot-ymax VAL     Y-axis max for plots (default: ${PLOT_YMAX:-4.0e-6})
  -h, --help          Show this help

You can also override via environment variables:
  INPUT_DIR, OUTPUT_DIR, DECOMP_METHODS, VLM_MODEL, BATCH_SIZE, DEVICE, NUM_WORKERS, SEED, PLOT_YMIN, PLOT_YMAX

All steps stream logs to stdout and to per-step log files under $OUTPUT_DIR/logs.
USAGE
}

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }
warn() { echo "[$(ts)] [WARN] $*" >&2; }

run_step() {
  local name="$1"; shift
  local logfile="$OUTPUT_DIR/logs/${name// /_}.log"
  log "START: ${name}"
  {
    # shellcheck disable=SC2068
    eval $@ 2>&1 | tee -a "$logfile"
  }
  log "DONE:  ${name}"
}

trap 'rc=$?; warn "Pipeline failed at line $LINENO (exit $rc). Check logs at: $OUTPUT_DIR/logs"; exit $rc' ERR

# -------------------------------
# Paths and defaults
# -------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

INPUT_DIR="${INPUT_DIR:-$ROOT_DIR/data/train}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/run_20250918_130709_cdgl}" #run_$TIMESTAMP
DECOMP_METHODS="${DECOMP_METHODS:-snmf,pca,simple}"
HF_HOME="${HF_HOME:-/mnt/abka03/huggingface/hub}"

# Optional tuning knobs used by underlying scripts (if they read env vars)
VLM_MODEL="${VLM_MODEL:-"google/gemma-3n-E4B-it"}"         # e.g., llava, llava-7b, qwen-vl, etc.
BATCH_SIZE="${BATCH_SIZE:-16}"       # e.g., 16
DEVICE="${DEVICE:-cuda}"               # e.g., cuda:0
NUM_WORKERS="${NUM_WORKERS:-}"     # e.g., 8
SEED="${SEED:-42}"

# Cropping controls (single source of truth)
CROP_INPUT_ROOT="${CROP_INPUT_ROOT:-$ROOT_DIR/data}" # e.g., /mnt/abka03/Projects/xl-vlms/data/train
CONCEPT_CROPS_PER_IMAGE="${CONCEPT_CROPS_PER_IMAGE:-100}"
PATCH_SIZE="${PATCH_SIZE:-128}"
RESIZE="${RESIZE:-512}"
MIN_IMAGES_PER_TAG="${MIN_IMAGES_PER_TAG:-20}"
MAX_IMAGES_PER_TAG="${MAX_IMAGES_PER_TAG:-300}"
CONCEPT_MODE="${CONCEPT_MODE:-1}"

# Explainer/Eval controls
LAYER_PATH="${LAYER_PATH:-model.language_model.norm}"
IMAGE_ROOT="${IMAGE_ROOT:-$ROOT_DIR/data/val}"
TOP_N="${TOP_N:-5}"
NUM_POINTS="${NUM_POINTS:-80}"

# Explainer prompt configuration
EXPL_PROMPT_MODE="${EXPL_PROMPT_MODE:-unsupervised}"   # unsupervised | binary | mcq
EXPL_LABEL="${EXPL_LABEL:-}"                            # used when binary
EXPL_CHOICES="${EXPL_CHOICES:-}"                        # CSV list when mcq

# Plot ranges
PLOT_YMIN="${PLOT_YMIN:-3.750e-6}"
PLOT_YMAX="${PLOT_YMAX:-3.990e-6}"

# Dataset inference controls
PROMPT="${PROMPT:-Identify all visible objects, items in the given image. Output only a single-word, comma-separated list. Do not include explanations, sentences, or any extra text—just the detected elements.}"
IMAGE_SIZE="${IMAGE_SIZE:-"512 512"}"     # e.g., "512 512"
IMAGE_BUDGET="${IMAGE_BUDGET:-50}" # e.g., 500

# -------------------------------
# Parse args
# -------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIR="$2"; shift 2;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --decomp)
      DECOMP_METHODS="$2"; shift 2;;
    --plot-ymin)
      PLOT_YMIN="$2"; shift 2;;
    --plot-ymax)
      PLOT_YMAX="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      warn "Unknown arg: $1"; usage; exit 1;;
  esac
done

# -------------------------------
# Prepare directories
# -------------------------------
mkdir -p "$OUTPUT_DIR"/logs
mkdir -p "$OUTPUT_DIR"/inference "$OUTPUT_DIR"/crops "$OUTPUT_DIR"/features \
         "$OUTPUT_DIR"/concept "$OUTPUT_DIR"/explanations "$OUTPUT_DIR"/eval "$OUTPUT_DIR"/plots

CONCEPT_MAP_JSON="$OUTPUT_DIR/inference/concepts_to_images.json"
CROPS_DIR="$OUTPUT_DIR/crops"
FEATURES_DIR="$OUTPUT_DIR" # the algorithm automatically creates the path.
DECOMP_DIR="$OUTPUT_DIR/concept"
EXPLAIN_DIR="$OUTPUT_DIR/explanations"
EVAL_DIR="$OUTPUT_DIR/eval"
PLOTS_DIR="$OUTPUT_DIR/plots"

# Export so child scripts can read them
export INPUT_DIR OUTPUT_DIR CONCEPT_MAP_JSON CROPS_DIR FEATURES_DIR DECOMP_DIR EXPLAIN_DIR EVAL_DIR PLOTS_DIR
export VLM_MODEL BATCH_SIZE DEVICE NUM_WORKERS SEED HF_HOME
export LAYER_PATH IMAGE_ROOT TOP_N NUM_POINTS
export EXPL_PROMPT_MODE EXPL_LABEL EXPL_CHOICES

log "Root:        $ROOT_DIR"
log "Scripts:     $SCRIPTS_DIR"
log "Input dir:   $INPUT_DIR"
log "Output dir:  $OUTPUT_DIR"
log "Decompose:   $DECOMP_METHODS"
log "Env: VLM_MODEL=${VLM_MODEL:-<unset>} BATCH_SIZE=${BATCH_SIZE:-<unset>} DEVICE=${DEVICE:-<unset>} NUM_WORKERS=${NUM_WORKERS:-<unset>} SEED=$SEED HF_HOME=$HF_HOME"
log "Crop: input_root=$CROP_INPUT_ROOT patch=$PATCH_SIZE resize=$RESIZE concept_crops_per_image=$CONCEPT_CROPS_PER_IMAGE min_per_tag=$MIN_IMAGES_PER_TAG max_per_tag=$MAX_IMAGES_PER_TAG concept_mode=$CONCEPT_MODE"
log "Explainer: layer=$LAYER_PATH image_root=$IMAGE_ROOT top_n=$TOP_N num_points=$NUM_POINTS prompt_mode=$EXPL_PROMPT_MODE label='${EXPL_LABEL}' choices='${EXPL_CHOICES}'"
log "Plot: y-range=[$PLOT_YMIN, $PLOT_YMAX]"
log "Inference: prompt='${PROMPT:0:60}...' image_size='${IMAGE_SIZE}' image_budget='${IMAGE_BUDGET}' batch=${BATCH_SIZE}"

# Verify required scripts are present
required=(
  run_dataset_inference.sh
  run_crop_images.sh
  run_feature_gen_cgdl.sh
  run_feature_decompose_cgdl.sh
  run_vlm_explainer_no_gt.sh
  run_concept_deletion_eval_token.sh
)
for f in "${required[@]}"; do
  if [[ ! -f "$SCRIPTS_DIR/$f" ]]; then
    warn "Missing required script: $SCRIPTS_DIR/$f"; exit 1
  fi
done

# -------------------------------
# 1) Dataset inference -> concept map
# -------------------------------
OBJECTS_CSV="$OUTPUT_DIR/inference/objects.csv"
if [[ -s "$OBJECTS_CSV" && -s "$CONCEPT_MAP_JSON" ]]; then
  log "Skip Dataset Inference (found $OBJECTS_CSV and $CONCEPT_MAP_JSON)"
else
  if [[ -s "$OBJECTS_CSV" && ! -s "$CONCEPT_MAP_JSON" ]]; then
    run_step "Build Concept Map (resume)" \
      "python -u \"$ROOT_DIR/concept_image_mapping.py\" --input \"$OBJECTS_CSV\" --output \"$CONCEPT_MAP_JSON\""
  else
    run_step "Dataset Inference" \
      "HF_HOME=\"$HF_HOME\" INPUT_DIR=\"$INPUT_DIR\" OUTPUT_DIR=\"$OUTPUT_DIR\" VLM_MODEL=\"$VLM_MODEL\" BATCH_SIZE=\"$BATCH_SIZE\" PROMPT=\"$PROMPT\" IMAGE_SIZE=\"$IMAGE_SIZE\" IMAGE_BUDGET=\"$IMAGE_BUDGET\" bash \"$SCRIPTS_DIR/run_dataset_inference.sh\""
  fi
fi

# Ensure map exists if produced by the script at a standard path; if not, just warn.
if [[ ! -s "$CONCEPT_MAP_JSON" ]]; then
  warn "Concept map not found at $CONCEPT_MAP_JSON (the step's script may use a different path)."
fi

# -------------------------------
# 2) Crop images using the concept->images map
# -------------------------------
if find "$CROPS_DIR" -type f -name '*.png' -print -quit | grep -q .; then
  log "Skip Crop Images (found crops under $CROPS_DIR)"
else
  run_step "Crop Images" \
    "bash \"$SCRIPTS_DIR/run_crop_images.sh\" \
      --input_root \"$CROP_INPUT_ROOT\" \
      --output_root \"$CROPS_DIR\" \
      --json_mapping \"$CONCEPT_MAP_JSON\" \
      --concept_crops_per_image \"$CONCEPT_CROPS_PER_IMAGE\" \
      --patch_size \"$PATCH_SIZE\" \
      --resize \"$RESIZE\" \
      --seed \"$SEED\" \
      --min_images_per_tag \"$MIN_IMAGES_PER_TAG\" \
      --max_images_per_tag \"$MAX_IMAGES_PER_TAG\" \
      $([[ \"$CONCEPT_MODE\" == \"0\" ]] && echo --no-concept_mode || true)"
fi

# -------------------------------
# 3) Generate concept features
# -------------------------------
if find "$FEATURES_DIR/features" -type f -name '*.pth' -print -quit | grep -q .; then
  log "Skip Feature Generation (found features under $FEATURES_DIR/features)"
else
  run_step "Generate Concept Features" "bash \"$SCRIPTS_DIR/run_feature_gen_cgdl.sh\" \"$VLM_MODEL\" \"$CROPS_DIR\" \"$FEATURES_DIR\" \"${HF_HOME:-}\""
fi

# -------------------------------
# 4) Decompose features across methods
# -------------------------------
IFS=',' read -r -a DECOMP_ARRAY <<< "$DECOMP_METHODS"
for method in "${DECOMP_ARRAY[@]}"; do
  export DECOMP_METHOD="$method"
  mkdir -p "$DECOMP_DIR/$method"
  # Pass HF cache, model, and features dir explicitly
  concept_raw="$DECOMP_DIR/$method/combined_concept_${method}_raw.pth" #
  if [[ -s "$concept_raw" ]]; then
    log "Skip Decompose Features ($method) (found $concept_raw)"
  else
    run_step "Decompose Features ($method)" "DECOMP_DIR=\"$DECOMP_DIR/$method\" bash \"$SCRIPTS_DIR/run_feature_decompose_cgdl.sh\" \"$HF_HOME\" \"$VLM_MODEL\" \"$FEATURES_DIR\""
  fi
done

# -------------------------------
# 5) Run VLM explainer to get explanations (per method)
# -------------------------------
for method in "${DECOMP_ARRAY[@]}"; do
  concept_path="$DECOMP_DIR/${method}/redefine_activations_text_grounding_${method}_combined_concept_${method}_gl_regrounded.pth" # combined_concept_${method}_raw.pth" #
  out_dir="$EXPLAIN_DIR/$method"; mkdir -p "$out_dir"
  out_json="$out_dir/vlm_explanations.json"
  if [[ -s "$out_json" ]]; then
    log "Skip VLM Explainer ($method) (found $out_json)"
  else
    run_step "VLM Explainer ($method)" \
      "HF_HOME=\"$HF_HOME\" VLM_MODEL=\"$VLM_MODEL\" CONCEPT_PATH=\"$concept_path\" LAYER_PATH=\"$LAYER_PATH\" IMAGE_ROOT=\"$IMAGE_ROOT\" TOP_N=\"$TOP_N\" EXPL_PROMPT_MODE=\"$EXPL_PROMPT_MODE\" EXPL_LABEL=\"$EXPL_LABEL\" EXPL_CHOICES=\"$EXPL_CHOICES\" EXPLAIN_DIR=\"$out_dir\" OUT_JSON=\"$out_json\" bash \"$SCRIPTS_DIR/run_vlm_explainer_no_gt.sh\""
  fi
done

# -------------------------------
# 6) Evaluate explanations (concept deletion, token-level) per method
# -------------------------------
for method in "${DECOMP_ARRAY[@]}"; do
  concept_path="$DECOMP_DIR/${method}/combined_concept_${method}_raw.pth"
  in_json="$EXPLAIN_DIR/$method/vlm_explanations.json"
  out_dir="$EVAL_DIR/$method"; mkdir -p "$out_dir"
  if find "$out_dir" -type f -name '*.csv' -print -quit | grep -q .; then
    log "Skip Concept Deletion Eval (Token) - $method (CSV exists in $out_dir)"
  else
    run_step "Concept Deletion Eval (Token) - $method" \
      "HF_HOME=\"$HF_HOME\" RESULTS_JSON=\"$in_json\" CONCEPT_PATH=\"$concept_path\" VLM_MODEL=\"$VLM_MODEL\" LAYER_PATH=\"$LAYER_PATH\" NUM_POINTS=\"$NUM_POINTS\" OUT_DIR=\"$out_dir\" bash \"$SCRIPTS_DIR/run_concept_deletion_eval_token.sh\""
  fi
done

# -------------------------------
# 7) Plot final results (optional, per method)
# -------------------------------
if [[ -f "$SCRIPTS_DIR/plot_concept_deletion_eval_token.py" ]]; then
  for method in "${DECOMP_ARRAY[@]}"; do
    plot_dir="$EVAL_DIR/$method"
    del_png="$plot_dir/c_deletion_token_all_ranks.png"
    ins_png="$plot_dir/c_insertion_token_all_ranks.png"
    if [[ -f "$del_png" && -f "$ins_png" ]]; then
      log "Skip Plot Concept Deletion (Token) - $method (plots exist)"
      continue
    fi
    if find "$plot_dir" -type f -name 'c_*_token_rank*.csv' -print -quit | grep -q .; then
  run_step "Plot Concept Deletion (Token) - $method" "python -u \"$SCRIPTS_DIR/plot_concept_deletion_eval_token.py\" --out_dir \"$plot_dir\" --ymin \"$PLOT_YMIN\" --ymax \"$PLOT_YMAX\""
    else
      warn "No CSVs found in $plot_dir for plotting; skipping $method."
    fi
  done
else
  warn "Plot script not found; skipping plots."
fi

# -------------------------------
# 8) Save combined plots into dedicated eval/plots directory (per method)
# -------------------------------
if [[ -f "$SCRIPTS_DIR/plot_concept_deletion_eval_token.py" ]]; then
  for method in "${DECOMP_ARRAY[@]}"; do
    src_dir="$EVAL_DIR/$method"
    dst_dir="$EVAL_DIR/$method/plots"
    del_png="$dst_dir/c_deletion_token_all_ranks.png"
    ins_png="$dst_dir/c_insertion_token_all_ranks.png"
    if [[ -f "$del_png" && -f "$ins_png" ]]; then
      log "Skip Save Combined Plots - $method (already present in $dst_dir)"
      continue
    fi
    if find "$src_dir" -type f -name 'c_*_token_rank*.csv' -print -quit | grep -q .; then
  mkdir -p "$dst_dir"
      # Copy CSVs to destination so the plotting script can scan them
      find "$src_dir" -maxdepth 1 -type f -name 'c_*_token_rank*.csv' -exec cp -u {} "$dst_dir" \;
  run_step "Save Combined Plots - $method" "python -u \"$SCRIPTS_DIR/plot_concept_deletion_eval_token.py\" --out_dir \"$dst_dir\" --ymin \"$PLOT_YMIN\" --ymax \"$PLOT_YMAX\""
    else
      warn "No CSVs found in $src_dir for plotting; skipping $method."
    fi
  done
fi

log "Pipeline completed. Outputs: $OUTPUT_DIR"
log "Logs: $OUTPUT_DIR/logs"
