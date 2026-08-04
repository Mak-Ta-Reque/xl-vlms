#!/usr/bin/env bash
# Full run of the xl-vlms pipeline on the coco10 dataset (all available images).
# Run scripts/run_coco10_pilot.sh first to validate the chain end-to-end on a
# small subset before launching this -- it takes considerably longer.
#
# Data: data/coco10/{train_all, val_grids} -- ~1050 train images / 50 val
# grids across 10 categories (apple, banana, bird, cake, cat, cup, dog, donut,
# knife, orange), built by preprocessing/build_coco10_dataset.py and
# preprocessing/build_coco10_masked_grids.py from COCO val2017 (real train2017
# images were unavailable on this machine, so val2017 was re-split into our
# own train/test pools -- see manifest.json for the exact split).
#
# Usage:
#   bash scripts/run_coco10_full.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

source /home/abka03/miniforge3/etc/profile.d/conda.sh
conda activate xlvlms

OUTPUT_DIR="outputs/coco10"

INPUT_DIR="data/coco10/train_all" \
IMAGE_ROOT="data/coco10/val_grids" \
CONCEPTS_VOCAB="src/assets/coco10_vocab.txt" \
NUM_CONCEPT=-1 \
CROP_MODE=langsam \
PROMPT_TEMPLATE=cgdl \
DECOMP_METHODS=snmf \
CLEAN_EXAMPLE_RATIO=0.2 \
PATCH_SIZE=160 \
IMAGE_BUDGET=-1 \
EXPL_MAX_IMAGES=-1 \
EXPL_PROMPT_MODE=mcq \
EXPL_CHOICES="apple,banana,bird,cake,cat,cup,dog,donut,knife,orange" \
OUTPUT_DIR="$OUTPUT_DIR" \
python -u scripts/run_full_pipeline.py --output-dir "$OUTPUT_DIR" --decomp snmf

echo "Full run complete. Outputs: $OUTPUT_DIR"
