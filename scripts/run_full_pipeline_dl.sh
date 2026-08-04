#!/usr/bin/env bash
# Orchestrates the XL-VLMs pipeline (DL variant):
# - Skips dataset inference and cropping
# - Uses DL feature generation and DL decomposition scripts
# Steps executed:
# 1) Generate concept features (DL)
# 2) Decompose features (multiple methods)
# 3) Run VLM explainer (per method)
# 4) Concept deletion evaluation (per method)
# 5) (Optional) Plot final figures

set -Eeuo pipefail

# -------------------------------
# Utility
# -------------------------------
usage() {
  cat <<'USAGE'
Usage: run_full_pipeline_dl.sh [--output-dir PATH] [--decomp METHODS] [--plot-ymin VAL] [--plot-ymax VAL]

Options:
  --output-dir PATH    Root output directory. Default: ../outputs/run_<timestamp>
  --decomp METHODS     Comma-separated decomposition methods. Default: snmf
  --plot-ymin VAL     Y-axis min for plots (default: ${PLOT_YMIN:-3.62e-6})
  --plot-ymax VAL     Y-axis max for plots (default: ${PLOT_YMAX:-4.0e-6})
  -h, --help          Show this help

You can also override via environment variables:
  OUTPUT_DIR, DECOMP_METHODS, VLM_MODEL, BATCH_SIZE, DEVICE, NUM_WORKERS, SEED,
  HF_HOME, LAYER_PATH, IMAGE_ROOT, TOP_N, NUM_POINTS, PLOT_YMIN, PLOT_YMAX

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

# Source .env as single source of truth
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen2_5_10cls_coxlmm/imnet100}"
DECOMP_METHODS="${DECOMP_METHODS:-snmf,simple,random,kmeans,pca}"
HF_HOME="${HF_HOME:-/mnt/abka03/huggingface/hub}"
DEFAULT_DATASET_SIZE="${DEFAULT_DATASET_SIZE:-100}"
# Optional tuning knobs used by underlying scripts (if they read env vars)
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-}"
SEED="${SEED:-42}"

# Dataset controls for DL feature generation
SPLIT="${SPLIT:-train}"
BASE_DATA_DIR="${BASE_DATA_DIR:-/mnt/abka03/xlvlm_data/imagenet_5_class/train}"

# Explainer/Eval controls
LAYER_PATH="${LAYER_PATH:-model.language_model.norm}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/abka03/xlvlm_data/imagenet_5_class/val_grids}"
TOP_N="${TOP_N:-3}"
NUM_POINTS="${NUM_POINTS:-70}"

# Explainer prompt configuration
EXPL_PROMPT_MODE="${EXPL_PROMPT_MODE:-unsupervised}"   # unsupervised | binary | mcq
EXPL_LABEL="${EXPL_LABEL:-}"                            # used when binary
EXPL_CHOICES="${EXPL_CHOICES:-}"                        # CSV list when mcq

# Plot ranges
PLOT_YMIN="${PLOT_YMIN:-6.55e-6}"
PLOT_YMAX="${PLOT_YMAX:-7.10e-6}"

# -------------------------------
# Parse args
# -------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
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
mkdir -p "$OUTPUT_DIR"/features "$OUTPUT_DIR"/concept \
         "$OUTPUT_DIR"/explanations "$OUTPUT_DIR"/eval "$OUTPUT_DIR"/plots

FEATURES_DIR="$OUTPUT_DIR"           # feature_gen_dl writes under this root
SAVE_DIR="$FEATURES_DIR"  # where run_feature_gen_dl.sh should write .pth features
DECOMP_DIR="$OUTPUT_DIR/concept"
EXPLAIN_DIR="$OUTPUT_DIR/explanations"
EVAL_DIR="$OUTPUT_DIR/eval"
PLOTS_DIR="$OUTPUT_DIR/plots"

# Export so child scripts can read them
mkdir -p "$SAVE_DIR" "$SAVE_DIR/features"

export OUTPUT_DIR FEATURES_DIR SAVE_DIR DECOMP_DIR EXPLAIN_DIR EVAL_DIR PLOTS_DIR
export VLM_MODEL BATCH_SIZE DEVICE NUM_WORKERS SEED HF_HOME
export LAYER_PATH IMAGE_ROOT TOP_N NUM_POINTS
export SPLIT BASE_DATA_DIR
export EXPL_PROMPT_MODE EXPL_LABEL EXPL_CHOICES

log "Root:        $ROOT_DIR"
log "Scripts:     $SCRIPTS_DIR"
log "Output dir:  $OUTPUT_DIR"
log "Feature dir: $SAVE_DIR"
log "Decompose:   $DECOMP_METHODS"
log "Env: VLM_MODEL=${VLM_MODEL:-<unset>} BATCH_SIZE=${BATCH_SIZE:-<unset>} DEVICE=${DEVICE:-<unset>} NUM_WORKERS=${NUM_WORKERS:-<unset>} SEED=$SEED HF_HOME=$HF_HOME"
log "Explainer: layer=$LAYER_PATH image_root=$IMAGE_ROOT top_n=$TOP_N num_points=$NUM_POINTS prompt_mode=$EXPL_PROMPT_MODE label='${EXPL_LABEL}' choices='${EXPL_CHOICES}'"
log "Plot: y-range=[$PLOT_YMIN, $PLOT_YMAX]"
log "DL data: split=$SPLIT base_data_dir=$BASE_DATA_DIR"

# Verify required scripts are present
required=(
  run_feature_gen_dl.sh
  run_feature_decompose_dl.sh
  run_vlm_explainer_no_gt.sh
  run_concept_deletion_eval_token.sh
)
for f in "${required[@]}"; do
  if [[ ! -f "$SCRIPTS_DIR/$f" ]]; then
    warn "Missing required script: $SCRIPTS_DIR/$f"; exit 1
  fi
done

# -------------------------------
# 1) Generate concept features (DL)
# -------------------------------
if find "$SAVE_DIR/features" -type f -name '*.pth' -print -quit | grep -q .; then
  log "Skip Feature Generation (found features under $SAVE_DIR/features)"
else
  run_step "Generate Concept Features (DL)" \
    "PYTHON_EXEC=\"${PYTHON_EXEC:-python}\" BATCH_SIZE=\"$BATCH_SIZE\" DEFAULT_DATASET_SIZE=\"$DEFAULT_DATASET_SIZE\" DATA_DIR=\"${DATA_DIR:-}\" TOKEN_OF_INTEREST=\"${TOKEN_OF_INTEREST:-}\" SAVE_FILENAME=\"${SAVE_FILENAME:-}\" HOOK_NAME=\"${HOOK_NAME:-}\" MODULES_TO_HOOK=\"${MODULES_TO_HOOK:-}\" PROMPT_TEMPLATE=\"${PROMPT_TEMPLATE:-}\" DEVICE=\"${DEVICE:-}\" bash \"$SCRIPTS_DIR/run_feature_gen_dl.sh\" \"$VLM_MODEL\" \"$FEATURES_DIR\" \"${HF_HOME:-}\""
fi

# -------------------------------
# 2) Decompose features across methods (DL)
# -------------------------------
IFS=',' read -r -a DECOMP_ARRAY <<< "$DECOMP_METHODS"
for method in "${DECOMP_ARRAY[@]}"; do
  export DECOMP_METHOD="$method"
  mkdir -p "$DECOMP_DIR/$method"
  concept_raw="$DECOMP_DIR/$method/combined_concept_${method}_raw.pth"
  if [[ -s "$concept_raw" ]]; then
    log "Skip Decompose Features ($method) (found $concept_raw)"
  else
    run_step "Decompose Features (DL:$method)" \
      "DECOMP_DIR=\"$DECOMP_DIR/$method\" bash \"$SCRIPTS_DIR/run_feature_decompose_dl.sh\" \"$HF_HOME\" \"$VLM_MODEL\" \"$FEATURES_DIR\""
  fi

done

# -------------------------------
# 3) Run VLM explainer to get explanations (per method)
# -------------------------------
for method in "${DECOMP_ARRAY[@]}"; do
  concept_path="$DECOMP_DIR/${method}/combined_concept_${method}_raw.pth"
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
# 4) Evaluate explanations (concept deletion, token-level) per method
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
# 4.5) Faithfulness AUC summary (insertion/deletion, top TOP_N concepts) per method
# -------------------------------
for method in "${DECOMP_ARRAY[@]}"; do
  out_dir="$EVAL_DIR/$method"
  auc_table="$out_dir/concept_curve_auc_table.csv"
  if [[ -s "$auc_table" ]]; then
    log "Skip Faithfulness AUC ($method) (found $auc_table)"
  else
    if find "$out_dir" -maxdepth 1 -type f -name 'c_*_token_rank*.json' -print -quit | grep -q .; then
      run_step "Faithfulness AUC ($method)" \
        "python -u \"$ROOT_DIR/eval/concept_curve_auc_eval.py\" --out_dir \"$out_dir\" --top_n \"$TOP_N\" --mode token"
    else
      warn "No curve JSONs found in $out_dir for AUC summary; skipping $method."
    fi
  fi
done

# -------------------------------
# 5) Plot final results (optional, per method)
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
# 6) Save combined plots into dedicated eval/plots directory (per method)
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
      find "$src_dir" -maxdepth 1 -type f -name 'c_*_token_rank*.csv' -exec cp -u {} "$dst_dir" \;
  run_step "Save Combined Plots - $method" "python -u \"$SCRIPTS_DIR/plot_concept_deletion_eval_token.py\" --out_dir \"$dst_dir\" --ymin \"$PLOT_YMIN\" --ymax \"$PLOT_YMAX\""
    else
      warn "No CSVs found in $src_dir for plotting; skipping $method."
    fi
  done
fi

log "DL Pipeline completed. Outputs: $OUTPUT_DIR"
log "Logs: $OUTPUT_DIR/logs"
