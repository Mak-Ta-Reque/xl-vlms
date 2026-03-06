#!/usr/bin/env bash
# New pipeline for prompt-based grounding explainer.
# Does not modify or reuse older scripts in-place; lives under scripts_grounding/.
# Steps:
# 1) Run vlm_prompt_grounding_explainer.py on an image root or list
# 2) Evaluate grounding outputs (simple metrics)
# 3) Plot evaluation curves and top names

set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage: run_grounding_full_pipeline.sh \
  --image-root PATH \
  [--model-name NAME] \
  [--out-dir PATH] \
  [--batch-size N] \
  [--max-new-tokens N] \
  [--temperature T]

Environment variables respected:
  HF_HOME (default: /mnt/abka03/huggingface/hub)

Examples:
  bash scripts_grounding/run_grounding_full_pipeline.sh \
    --image-root /mnt/abka03/Projects/xl-vlms/data/val \
    --model-name google/gemma-3n-E4B-it \
    --out-dir /mnt/abka03/Projects/xl-vlms/outputs/grounding_run
USAGE
}

log() { echo "[$(date '+%F %T')] $*"; }
warn() { echo "[$(date '+%F %T')] [WARN] $*" >&2; }
run_step() { local name="$1"; shift; log "START: $name"; { eval $@; } ; log "DONE:  $name"; }
trap 'rc=$?; warn "Pipeline failed at line $LINENO (exit $rc)"; exit $rc' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Source .env as single source of truth
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi

IMAGE_ROOT="/mnt/abka03/Projects/xl-vlms/data/val"
MODEL_NAME="google/gemma-3n-E4B-it"
OUT_DIR="$ROOT_DIR/outputs/grounding_run_20250919_144409"
BATCH_SIZE=1
MAX_NEW_TOKENS=300
TEMPERATURE=0.0

# Track step statuses for a summary
STEP1_STATUS="pending"   # explainer
STEP2_STATUS="pending"   # eval
STEP3_STATUS="pending"   # plot basic
STEP4_STATUS="pending"   # token blur eval
STEP5_STATUS="pending"   # token blur plot

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-root) IMAGE_ROOT="$2"; shift 2;;
    --model-name) MODEL_NAME="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --temperature) TEMPERATURE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) warn "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$IMAGE_ROOT" ]]; then
  warn "--image-root is required"; usage; exit 1
fi

mkdir -p "$OUT_DIR" "$OUT_DIR/logs" "$OUT_DIR/eval"

EXPLAIN_JSON="$OUT_DIR/vlm_groundings.json"
EVAL_DIR="$OUT_DIR/eval"

HF_HOME="${HF_HOME:-/mnt/abka03/huggingface/hub}"
export HF_HOME

# 1) Run explainer (skip if JSON already exists and is non-empty)
if [[ -s "$EXPLAIN_JSON" ]]; then
  log "Skip explainer: output exists -> $EXPLAIN_JSON"
  STEP1_STATUS="skipped"
else
  run_step "Run VLM Prompt Grounding Explainer" \
    "python -u \"$ROOT_DIR/inference/vlm_prompt_grounding_explainer.py\" \
      --model_name \"$MODEL_NAME\" \
      --image_root \"$IMAGE_ROOT\" \
      --batch_size \"$BATCH_SIZE\" \
      --max_new_tokens \"$MAX_NEW_TOKENS\" \
      --temperature \"$TEMPERATURE\" \
      --out_json \"$EXPLAIN_JSON\""
  STEP1_STATUS="ran"
fi

# 2) Evaluate (skip if CSV already exists)
if [[ -s "$EVAL_DIR/grounding_summary.csv" ]]; then
  log "Skip eval: $EVAL_DIR/grounding_summary.csv exists"
  STEP2_STATUS="skipped"
else
  run_step "Evaluate Groundings" \
    "python -u -m eval.grounding_eval --results_json \"$EXPLAIN_JSON\" --out_dir \"$EVAL_DIR\""
  STEP2_STATUS="ran"
fi

# 3) Plot (skip if both images exist)
if [[ -s "$EVAL_DIR/grounding_curve.png" && -s "$EVAL_DIR/top_names.png" ]]; then
  log "Skip plot: grounding_curve.png and top_names.png exist"
  STEP3_STATUS="skipped"
else
  run_step "Plot Grounding Eval" \
    "python -u \"$ROOT_DIR/scripts_grounding/plot_grounding_eval.py\" --eval_dir \"$EVAL_DIR\""
  STEP3_STATUS="ran"
fi

# 4) Token blur eval (probability vs blur strength) - skip if curve CSV exists
if [[ -s "$EVAL_DIR/token_blur_curve.csv" ]]; then
  log "Skip token blur eval: $EVAL_DIR/token_blur_curve.csv exists"
  STEP4_STATUS="skipped"
else
  run_step "Evaluate Token Blur" \
    "python -u -m eval.grounding_token_blur_eval --results_json \"$EXPLAIN_JSON\" --out_dir \"$EVAL_DIR\""
  STEP4_STATUS="ran"
fi

# 5) Plot token blur curve (skip if image exists)
if [[ -s "$EVAL_DIR/token_blur_curve.png" ]]; then
  log "Skip token blur plot: $EVAL_DIR/token_blur_curve.png exists"
  STEP5_STATUS="skipped"
else
  run_step "Plot Token Blur Curve" \
    "python -u \"$ROOT_DIR/scripts_grounding/plot_grounding_blur_token.py\" --eval_dir \"$EVAL_DIR\""
  STEP5_STATUS="ran"
fi

log "All done. Outputs in $OUT_DIR"

# Summary of which steps ran vs skipped
echo "----------------------------------------"
log "Pipeline summary:"
log "  1) Explainer           : $STEP1_STATUS"
log "  2) Grounding Eval      : $STEP2_STATUS"
log "  3) Plot Basic          : $STEP3_STATUS"
log "  4) Token Blur Eval     : $STEP4_STATUS"
log "  5) Token Blur Plot     : $STEP5_STATUS"
echo "----------------------------------------"
