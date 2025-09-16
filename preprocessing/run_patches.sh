#!/usr/bin/env bash
set -euo pipefail

# run_patches.sh
# Runs random, grid, and combined (merged) patch extraction using preprocessing/random_crops.py
# Requires: bash, python, PIL installed.
# Example:
#   bash run_patches.sh -i /data/imagenet -o /data/patches -p 128 -n 50 -m 0.5 --seed 42
# With JSON mapping:
#   bash run_patches.sh -i /data/imagenet -o /data/patches -j /data/tags.json -p 128 -n 50
# With resize (width only, keeps aspect ratio):
#   bash run_patches.sh -i /data/imagenet -o /data/patches -r 512

usage() {
  echo "Usage: $0 -i INPUT_ROOT -o OUTPUT_ROOT [-j JSON_MAPPING] [-p PATCH_SIZE] [-n PATCHES_PER_IMAGE] [-m MAX_OVERLAP] [-r RESIZE_WIDTH] [--seed SEED]" >&2
  exit 1
}

INPUT_ROOT=""
OUTPUT_ROOT=""
JSON_MAPPING=""
PATCH_SIZE=128
PATCHES_PER_IMAGE=10
MAX_OVERLAP=0.25
RESIZE=""
SEED=""
PYTHON_BIN=${PYTHON:-python}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input_root) INPUT_ROOT="$2"; shift 2;;
    -o|--output_root) OUTPUT_ROOT="$2"; shift 2;;
    -j|--json_mapping) JSON_MAPPING="$2"; shift 2;;
    -p|--patch_size) PATCH_SIZE="$2"; shift 2;;
    -n|--patches_per_image) PATCHES_PER_IMAGE="$2"; shift 2;;
    -m|--max_overlap) MAX_OVERLAP="$2"; shift 2;;
    -r|--resize) RESIZE="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1" >&2; usage;;
  esac
done

[[ -z "$INPUT_ROOT" || -z "$OUTPUT_ROOT" ]] && usage

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PY_SCRIPT="$SCRIPT_DIR/random_crops.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "Error: random_crops.py not found at $PY_SCRIPT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
RANDOM_DIR="$OUTPUT_ROOT/random"
GRID_DIR="$OUTPUT_ROOT/grid"
COMBINED_DIR="$OUTPUT_ROOT/combined"
mkdir -p "$RANDOM_DIR" "$GRID_DIR" "$COMBINED_DIR"

COMMON_ARGS=( --input_root "$INPUT_ROOT" --patch_size "$PATCH_SIZE" --patches_per_image "$PATCHES_PER_IMAGE" --max_overlap "$MAX_OVERLAP" )
if [[ -n "$JSON_MAPPING" ]]; then
  COMMON_ARGS+=( --json_mapping "$JSON_MAPPING" )
fi
if [[ -n "$RESIZE" ]]; then
  COMMON_ARGS+=( --resize "$RESIZE" )
fi
if [[ -n "$SEED" ]]; then
  COMMON_ARGS+=( --seed "$SEED" )
fi

# Random patches
echo "[1/3] Generating random patches -> $RANDOM_DIR"
$PYTHON_BIN "$PY_SCRIPT" "${COMMON_ARGS[@]}" --output_root "$RANDOM_DIR"

# Grid patches
echo "[2/3] Generating grid patches -> $GRID_DIR"
$PYTHON_BIN "$PY_SCRIPT" "${COMMON_ARGS[@]}" --grid --output_root "$GRID_DIR"

# Combine (prefix to avoid filename clashes)
echo "[3/3] Merging into $COMBINED_DIR"
find "$RANDOM_DIR" -type f -name '*.png' -print0 | while IFS= read -r -d '' f; do
  cp "$f" "$COMBINED_DIR/random_$(basename "$f")"
done
find "$GRID_DIR" -type f -name '*.png' -print0 | while IFS= read -r -d '' f; do
  cp "$f" "$COMBINED_DIR/grid_$(basename "$f")"
done

echo "Done. Outputs:"
echo "  Random:   $RANDOM_DIR"
echo "  Grid:     $GRID_DIR"
echo "  Combined: $COMBINED_DIR"
