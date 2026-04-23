#!/usr/bin/env bash
set -euo pipefail

# Run step-2 crop generation debug outputs for all 4 crop modes and
# store each mode in a separate directory.
#
# Usage:
#   bash scripts/test_all_crop_debug_outputs.sh
#   bash scripts/test_all_crop_debug_outputs.sh /custom/output/root

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  echo "[ERROR] .env file not found at $ROOT_DIR/.env"
  exit 1
fi

# Export all vars from .env into current shell.
set -a
# shellcheck disable=SC1091
source .env
set +a

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
BASE_OUT_DEFAULT="${OUTPUT_DIR%/}/debug_all_crop_modes_${TS}"
BASE_OUT="${1:-$BASE_OUT_DEFAULT}"

SHARED_OUT="$BASE_OUT/shared"
SHARED_MAPPING="$SHARED_OUT/inference/concepts_to_images.json"

INPUT_DIR="${INPUT_DIR:-$ROOT_DIR/data}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.5}"
MASKS_PER_IMAGE="${MASKS_PER_IMAGE:-10}"
CONCEPT_MASKS_PER_IMAGE="${CONCEPT_MASKS_PER_IMAGE:-1}"
MIN_IMAGES_PER_TAG="${MIN_IMAGES_PER_TAG:-2}"
MAX_IMAGES_PER_TAG="${MAX_IMAGES_PER_TAG:-10}"
PATCH_SIZE="${PATCH_SIZE:-60}"
DETECTION_BATCH_SIZE="${DETECTION_BATCH_SIZE:-8}"
IMAGE_SIZE_WIDTH="${IMAGE_SIZE_WIDTH:-512}"
POSITIVE_NEGATIVE_SEGMENT="${POSITIVE_NEGATIVE_SEGMENT:-0}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$BASE_OUT"

echo "[INFO] Root dir:   $ROOT_DIR"
echo "[INFO] Python:     $PYTHON_BIN"
echo "[INFO] Base output:$BASE_OUT"
echo "[INFO] Input dir:  $INPUT_DIR"

# Build concept mapping once (Step 1) and reuse for all crop modes.
if [[ ! -f "$SHARED_MAPPING" ]]; then
  echo "[INFO] Building shared concept mapping (pipeline step 1)..."
  env \
    OUTPUT_DIR="$SHARED_OUT" \
    DEBUG_SAVE_VLM_INPUTS=0 \
    "$PYTHON_BIN" scripts/run_full_pipeline.py --only-step 1
else
  echo "[INFO] Reusing shared concept mapping: $SHARED_MAPPING"
fi

if [[ ! -f "$SHARED_MAPPING" ]]; then
  echo "[ERROR] Shared concept mapping was not created: $SHARED_MAPPING"
  exit 1
fi

run_mode() {
  local mode="$1"
  local detector="none"
  if [[ "$mode" == "langsam" || "$mode" == "sam3" ]]; then
    detector="$mode"
  fi

  local mode_out="$BASE_OUT/$mode"
  local mode_infer="$mode_out/inference"
  local mode_crops="$mode_infer/crops.json"
  local mode_debug="$mode_infer/debug_crop_overlays"

  mkdir -p "$mode_infer"

  echo "[INFO] Running mode=$mode detector=$detector"
  env \
    HF_HOME="$HF_HOME" \
    CROP_MODE="$mode" \
    DEBUG_SAVE_VLM_INPUTS=1 \
    "$PYTHON_BIN" preprocessing/crops_to_json.py \
      --mapping_json "$SHARED_MAPPING" \
      --image_root "$INPUT_DIR" \
      --output_json "$mode_crops" \
      --detector "$detector" \
      --masks_per_image "$MASKS_PER_IMAGE" \
      --concept_masks_per_image "$CONCEPT_MASKS_PER_IMAGE" \
      --min_images_per_tag "$MIN_IMAGES_PER_TAG" \
      --max_images_per_tag "$MAX_IMAGES_PER_TAG" \
      --patch_size "$PATCH_SIZE" \
      --batch_size "$DETECTION_BATCH_SIZE" \
      --image_size_width "$IMAGE_SIZE_WIDTH" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --confidence_threshold "$BOX_THRESHOLD" \
      --positive_negative_segment "$POSITIVE_NEGATIVE_SEGMENT"

  local overlay_count="0"
  if [[ -d "$mode_debug" ]]; then
    overlay_count="$(find "$mode_debug" -type f -name '*_crop_overlay.jpg' | wc -l | tr -d ' ')"
  fi

  echo "[DONE] $mode"
  echo "       crops:  $mode_crops"
  echo "       debug:  $mode_debug"
  echo "       overlays: $overlay_count"
}

run_mode "random"
run_mode "sliding_window"
run_mode "langsam"
run_mode "sam3"

echo ""
echo "[SUCCESS] Completed all 4 crop mode debug runs."
echo "[SUCCESS] Output root: $BASE_OUT"
echo "[SUCCESS] Debug directories:"
echo "  - $BASE_OUT/random/inference/debug_crop_overlays"
echo "  - $BASE_OUT/sliding_window/inference/debug_crop_overlays"
echo "  - $BASE_OUT/langsam/inference/debug_crop_overlays"
echo "  - $BASE_OUT/sam3/inference/debug_crop_overlays"
