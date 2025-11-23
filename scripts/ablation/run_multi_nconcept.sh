#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<USAGE
Usage: $0 --resource-dir RESOURCE_OUT --output-dir BASE_OUT
          [--n-list 2,10,20] [--pipeline ./run_pipeline.sh]

RESOURCE_OUT: directory that contains precomputed inference/ (source)
BASE_OUT:     directory where per-n runs will be created

Example:
  $0 --resource-dir outputs/qwen2_5_10cls_sam/imnet100 \
     --output-dir   outputs/qwen2_5_10cls_sam/imnet100_runs \
     --n-list 2,10
USAGE
  exit 1
}

PIPELINE_SCRIPT="./scripts/run_full_pipeline_without_coroping_omba_diffrentnumber_opf_crops.sh"
RESOURCE_OUT=""
BASE_OUT=""
N_LIST="2,10"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resource-dir) RESOURCE_OUT="$2"; shift 2;;
    --output-dir)   BASE_OUT="$2"; shift 2;;
    --n-list)       N_LIST="$2"; shift 2;;
    --pipeline)     PIPELINE_SCRIPT="$2"; shift 2;;
    -h|--help)      usage;;
    *) echo "[ERROR] Unknown arg: $1"; usage;;
  esac
done

[[ -z "$RESOURCE_OUT" || -z "$BASE_OUT" ]] && echo "[ERROR] --resource-dir and --output-dir are required" && usage
[[ ! -f "$PIPELINE_SCRIPT" ]] && echo "[ERROR] Pipeline script not found: $PIPELINE_SCRIPT" && exit 1

RESOURCE_INF="$RESOURCE_OUT/inference"
RESOURCE_FEAT="$RESOURCE_OUT/features"
if [[ ! -d "$RESOURCE_INF" ]]; then
  echo "[ERROR] Resource inference dir not found: $RESOURCE_INF"
  echo "Make sure RESOURCE_OUT has inference/ with objects.csv, concepts_to_images.json, crops.json, etc."
  exit 1
fi
if [[ ! -d "$RESOURCE_FEAT" ]]; then
  echo "[ERROR] Resource features dir not found: $RESOURCE_FEAT"
  echo "Make sure RESOURCE_OUT has features/ with precomputed features."
  exit 1
fi

mkdir -p "$BASE_OUT"

IFS=',' read -r -a N_ARR <<< "$N_LIST"

for n in "${N_ARR[@]}"; do
  OUT_DIR="$BASE_OUT/n_${n}"

  echo "=============================="
  echo "Running for n_concepts=$n"
  echo "RESOURCE=$RESOURCE_OUT"
  echo "OUT_DIR=$OUT_DIR"
  echo "=============================="

  mkdir -p "$OUT_DIR/inference"

  # Copy precomputed inference from resource -> per-n dir (don’t overwrite existing)
  rsync -a --ignore-existing "$RESOURCE_INF/" "$OUT_DIR/inference/"
  rsync -a --ignore-existing "$RESOURCE_FEAT/" "$OUT_DIR/features/"

  # Run pipeline with overridden OUTPUT_DIR + n_concepts
  n_concepts="$n" OUTPUT_DIR="$OUT_DIR" bash "$PIPELINE_SCRIPT" --output-dir "$OUT_DIR"
done
