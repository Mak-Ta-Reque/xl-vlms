#!/usr/bin/env bash
set -euo pipefail  # Safer script: exit on errors and unset vars

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source .env as single source of truth
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi

##########################################
# Configuration (prefer environment)
##########################################
# Prefer orchestrator-provided VLM_MODEL, fall back to MODEL_NAME or positional
MODEL_NAME=${VLM_MODEL:-${MODEL_NAME:-${1:-"google/gemma-3n-E4B-it"}}}
## Use crops directory provided by orchestrator; no hard-coded default
if [[ -z "${BASE_DATA_DIR:-}" ]]; then
  if [[ -n "${CROPS_DIR:-}" ]]; then
    BASE_DATA_DIR="${CROPS_DIR}"
  elif [[ -n "${2:-}" ]]; then
    BASE_DATA_DIR="${2}"
  else
    echo "Error: CROPS_DIR (or BASE_DATA_DIR) must be set by the orchestrator or provided as 2nd arg." >&2
    exit 1
  fi
fi
SAVE_DIR=${FEATURES_DIR:-${3:-"/mnt/abka03/Projects/xl-vlms/outputs/cdgl/train"}}
HF_HOME=${HF_HOME:-${4:-"/mnt/abka03/huggingface/hub"}}
# Resolve Python interpreter
if command -v "${PYTHON:-${PYTHON_BIN:-python}}" >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON:-${PYTHON_BIN:-python}}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi
##########################################
# Configuration
##########################################
DATASET_NAME="image"
DEFAULT_DATASET_SIZE=1600
OVERRIDE_DATASET_SIZE=1600
HOOK_NAME="save_hidden_states_mean"
MODULES_TO_HOOK="model.language_model.norm"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
SCRIPT_PATH="$ROOT_DIR/src/save_features.py"
PROMPT_TEMPLATE="cgdl"
SPLIT="train"
BATCH_SIZE=30
MAX_ITERATIONS=2000000
COUNT=0

##########################################
# Start total timer
##########################################
START_TIME=$(date +%s)

for dir_path in "${BASE_DATA_DIR}"/*/; do
  if (( COUNT >= MAX_ITERATIONS )); then
    echo "Reached maximum number of iterations ($MAX_ITERATIONS), stopping."
    break
  fi

  FILE_COUNT=$(find "$dir_path" -type f -name "*.png" | wc -l)
  DATASET_SIZE="$FILE_COUNT"
  if (( DATASET_SIZE > DEFAULT_DATASET_SIZE )); then
    DATASET_SIZE=$OVERRIDE_DATASET_SIZE
  fi

  dir_name=$(basename "$dir_path")

  if [[ "$dir_name" == *"_"* ]]; then
    TOKEN_OF_INTEREST="${dir_name##*_}"
  else
    TOKEN_OF_INTEREST="$dir_name"
  fi

  concept="${dir_name//_/ }"
  SAVE_FILENAME="qwen2_patched_image_${dir_name}_token_of_interest_concept_generation_split_${SPLIT}"

  echo "[${COUNT}/${MAX_ITERATIONS}] Processing: $dir_name (Token: $TOKEN_OF_INTEREST, Concept: \"$concept\")"

  HF_HOME="$HF_HOME" "$PYTHON_BIN" "$SCRIPT_PATH" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --dataset_size "$DATASET_SIZE" \
    --data_dir "$dir_path" \
    --hook_name "$HOOK_NAME" \
    --token_of_interest "$concept" \
    --modules_to_hook "$MODULES_TO_HOOK" \
    --prompt_template "$PROMPT_TEMPLATE" \
    --save_dir "$SAVE_DIR" \
    --save_filename "$SAVE_FILENAME" \
    --batch_size "$BATCH_SIZE" \
    --generation_mode \
    --save_only_generated_tokens \
    --exact_match_modules_to_hook

  COUNT=$((COUNT + 1))
done

##########################################
# End total timer and report
##########################################
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo -e "\n✅ Script completed in ${TOTAL_TIME}s (~$((TOTAL_TIME / 60)) minutes)"
