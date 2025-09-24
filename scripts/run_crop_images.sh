#!/usr/bin/env bash
# Run image cropping based on concept→image JSON mapping.
# Defaults mirror the README example. Override via CLI flags or env vars.
set -euo pipefail

# Default parameters, prefer orchestrator exports
INPUT_ROOT="${INPUT_ROOT:-${INPUT_DIR:-/mnt/abka03/Projects/xl-vlms/data}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CROPS_DIR:-/mnt/abka03/Projects/xl-vlms/crops/train}}"
JSON_MAPPING="${JSON_MAPPING:-${CONCEPT_MAP_JSON:-/mnt/abka03/Projects/xl-vlms/data/coco_10_concept_image_mapping.json}}"
CONCEPT_CROPS_PER_IMAGE="${CONCEPT_CROPS_PER_IMAGE:-48}"
PATCH_SIZE="${PATCH_SIZE:-200}"
RESIZE="${RESIZE:-512}"
SEED="${SEED:-123}"
MIN_IMAGES_PER_TAG="${MIN_IMAGES_PER_TAG:-20}"
MAX_IMAGES_PER_TAG="${MAX_IMAGES_PER_TAG:-300}"
CONCEPT_MODE="${CONCEPT_MODE:-0}"  # 1 to enable --concept_mode, 0 to disable
BATCH_SIZE="${BATCH_SIZE:-8}"
OBJECT_DETECTION="${OBJECT_DETECTION:-1}"  # 1 to enable --object_detection, 0 to disable


usage() {
  cat <<EOF
Usage: bash scripts/$(basename "$0") [options]

Options (override defaults from README example):
  --input_root PATH                 (default: ${INPUT_ROOT})
  --output_root PATH                (default: ${OUTPUT_ROOT})
  --json_mapping FILE               (default: ${JSON_MAPPING})
  --concept_crops_per_image INT     (default: ${CONCEPT_CROPS_PER_IMAGE})
  --patch_size INT                  (default: ${PATCH_SIZE})
  --resize INT                      (default: ${RESIZE})
  --seed INT                        (default: ${SEED})
  --min_images_per_tag INT          (default: ${MIN_IMAGES_PER_TAG})
  --max_images_per_tag INT          (default: ${MAX_IMAGES_PER_TAG})
  --no-concept_mode                 Disable --concept_mode flag
  --no-object_detection             Disable --object_detection flag
  --batch_size INT                  (default: ${BATCH_SIZE})
  -h, --help                        Show this help

Environment variables:
  INPUT_ROOT, OUTPUT_ROOT, JSON_MAPPING, CONCEPT_CROPS_PER_IMAGE,
  PATCH_SIZE, RESIZE, SEED, MIN_IMAGES_PER_TAG, MAX_IMAGES_PER_TAG,
  CONCEPT_MODE (1/0), OBJECT_DETECTION (1/0), BATCH_SIZE, PYTHON

Examples:
  bash scripts/$(basename "$0")
  bash scripts/$(basename "$0") --input_root /data --output_root /crops/train \
    --json_mapping data/coco_10_concept_image_mapping.json --patch_size 128 --resize 512
EOF
}

# Parse CLI args
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_root) INPUT_ROOT="$2"; shift 2;;
    --output_root) OUTPUT_ROOT="$2"; shift 2;;
    --json_mapping) JSON_MAPPING="$2"; shift 2;;
    --concept_crops_per_image) CONCEPT_CROPS_PER_IMAGE="$2"; shift 2;;
    --patch_size) PATCH_SIZE="$2"; shift 2;;
    --resize) RESIZE="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --min_images_per_tag) MIN_IMAGES_PER_TAG="$2"; shift 2;;
    --max_images_per_tag) MAX_IMAGES_PER_TAG="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --no-concept_mode) CONCEPT_MODE=0; shift;;
    --no-object_detection) OBJECT_DETECTION=0; shift;;
    --object_detection) OBJECT_DETECTION=1; shift;;
    -h|--help) usage; exit 0;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Build optional flag
CONCEPT_MODE_FLAG=()
if [[ "${CONCEPT_MODE}" == "1" ]]; then
  CONCEPT_MODE_FLAG+=("--concept_mode")
fi

OBJECT_DETECTION_FLAG=()
if [[ "${OBJECT_DETECTION}" == "1" ]]; then
  OBJECT_DETECTION_FLAG+=("--object_detection")
fi

# Sanity checks
if [[ ! -f "${REPO_ROOT}/preprocessing/random_crops.py" ]]; then
  echo "Error: random_crops.py not found at ${REPO_ROOT}/preprocessing/random_crops.py" >&2
  exit 1
fi
if [[ ! -f "${JSON_MAPPING}" ]]; then
  echo "Error: JSON mapping not found at ${JSON_MAPPING}" >&2
  exit 1
fi

# Run
# Build argv
argv=(
  "${REPO_ROOT}/preprocessing/random_crops.py"
  --input_root  "${INPUT_ROOT}"
  --output_root "${OUTPUT_ROOT}"
  --json_mapping "${JSON_MAPPING}"
  ${CONCEPT_MODE_FLAG[@]:+"${CONCEPT_MODE_FLAG[@]}"}
  ${OBJECT_DETECTION_FLAG[@]:+"${OBJECT_DETECTION_FLAG[@]}"}
  --concept_crops_per_image "${CONCEPT_CROPS_PER_IMAGE}"
  --patch_size "${PATCH_SIZE}"
  --resize "${RESIZE}"
  --seed "${SEED}"
  --min_images_per_tag "${MIN_IMAGES_PER_TAG}"
  --max_images_per_tag "${MAX_IMAGES_PER_TAG}"
  --batch_size "${BATCH_SIZE}"
)

# Append any extra args if present
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  argv+=("${EXTRA_ARGS[@]}")
fi

# Resolve Python interpreter (env PYTHON wins, then python, then python3)
PY_CMD="${PYTHON:-python}"
if ! command -v "${PY_CMD}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PY_CMD="python3"
  else
    echo "Error: No Python interpreter found. Set PYTHON env var or install python/python3." >&2
    exit 1
  fi
fi

# Ensure output directory exists
mkdir -p "${OUTPUT_ROOT}"

"${PY_CMD}" "${argv[@]}"
