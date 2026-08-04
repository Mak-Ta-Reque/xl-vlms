#!/usr/bin/env bash
# Pilot/smoke run of the xl-vlms pipeline on the coco10 dataset (small subset).
# Validates the whole chain (VLM captioning -> crops -> features -> SNMF ->
# explainer -> faithfulness eval -> BERT/CLIP) before committing to the full
# run_coco10_full.sh. Uses the same settings validated on imagenet10 today:
# CROP_MODE=langsam (real text-conditioned detection, needed for a healthy
# concept yield) and CLEAN_EXAMPLE_RATIO=0.2.
#
# Data: data/coco10/{train_all, val_grids} (see preprocessing/build_coco10_dataset.py
# and preprocessing/build_coco10_masked_grids.py -- already built).
#
# Usage:
#   bash scripts/run_coco10_pilot.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

source /home/abka03/miniforge3/etc/profile.d/conda.sh
conda activate xlvlms

OUTPUT_DIR="outputs/coco10_pilot"

INPUT_DIR="data/coco10/train_all" \
IMAGE_ROOT="data/coco10/val_grids" \
CONCEPTS_VOCAB="src/assets/coco10_vocab.txt" \
NUM_CONCEPT=-1 \
CROP_MODE=langsam \
PROMPT_TEMPLATE=cgdl \
DECOMP_METHODS=snmf \
CLEAN_EXAMPLE_RATIO=0.2 \
PATCH_SIZE=160 \
IMAGE_BUDGET=30 \
EXPL_MAX_IMAGES=3 \
EXPL_PROMPT_MODE=mcq \
EXPL_CHOICES="apple,banana,bird,cake,cat,cup,dog,donut,knife,orange" \
OUTPUT_DIR="$OUTPUT_DIR" \
python -u scripts/run_full_pipeline.py --output-dir "$OUTPUT_DIR" --decomp snmf

echo "Pilot run complete. Outputs: $OUTPUT_DIR"
