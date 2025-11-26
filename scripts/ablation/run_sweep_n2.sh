#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<USAGE
Alpha ablation (fixed n_concepts=2)

Usage:
  $(basename "$0") --resource-dir RESOURCE_OUT --output-dir OUT_ROOT
                   [--pipeline PIPELINE.sh]
                   [--alpha-min 5 --alpha-max 50 --alpha-step 5]
                   [--alpha-list 5,10,15,...,50]

Required:
  --resource-dir   Preprocessed run dir that contains inference/ (source)
  --output-dir     Output root where alpha_* runs will be created

Optional:
  --pipeline       Path to your pipeline script (default: ./run_pipeline.sh)
  --alpha-min      Min DL_ALPHA (default: 5)
  --alpha-max      Max DL_ALPHA (default: 50)
  --alpha-step     Step size (default: 5)
  --alpha-list     Comma list overrides min/max/step (e.g., 5,20,50)

Example:
  ./run_alpha_ablation_n2.sh \
    --resource-dir outputs/qwen2_5_10cls_sam/imnet100 \
    --output-dir   outputs/qwen2_5_10cls_sam/imnet100_alpha_runs \
    --alpha-min 5 --alpha-max 50 --alpha-step 5
USAGE
  exit 1
}

PIPELINE="scripts/run_full_pipeline_without_coroping_omba_diffrentnumber_opf_crops.sh"
RESOURCE_DIR=""
OUT_ROOT=""
ALPHA_MIN=5
ALPHA_MAX=50
ALPHA_STEP=5
ALPHA_LIST=""
N_CONCEPTS_FIXED=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resource-dir) RESOURCE_DIR="$2"; shift 2;;
    --output-dir)   OUT_ROOT="$2"; shift 2;;
    --pipeline)     PIPELINE="$2"; shift 2;;
    --alpha-min)    ALPHA_MIN="$2"; shift 2;;
    --alpha-max)    ALPHA_MAX="$2"; shift 2;;
    --alpha-step)   ALPHA_STEP="$2"; shift 2;;
    --alpha-list)   ALPHA_LIST="$2"; shift 2;;
    -h|--help)      usage;;
    *) echo "[ERROR] Unknown arg: $1"; usage;;
  esac
done

[[ -z "$RESOURCE_DIR" || -z "$OUT_ROOT" ]] && echo "[ERROR] --resource-dir and --output-dir are required" && usage
[[ ! -f "$PIPELINE" ]] && echo "[ERROR] Pipeline not found: $PIPELINE" && exit 1
[[ ! -d "$RESOURCE_DIR/inference" ]] && echo "[ERROR] Resource inference not found: $RESOURCE_DIR/inference" && exit 1
[[ ! -d "$RESOURCE_DIR/features" ]] && echo "[ERROR] Resource features not found: $RESOURCE_DIR/features" && exit 1

mkdir -p "$OUT_ROOT"

# Build alpha list if not provided
if [[ -z "$ALPHA_LIST" ]]; then
  tmp=()
  a=$ALPHA_MIN
  while [[ $a -le $ALPHA_MAX ]]; do
    tmp+=("$a")
    a=$((a + ALPHA_STEP))
  done
  ALPHA_LIST="$(IFS=','; echo "${tmp[*]}")"
fi
IFS=',' read -r -a ALPHAS <<< "$ALPHA_LIST"

echo "[INFO] RESOURCE_DIR = $RESOURCE_DIR"
echo "[INFO] OUT_ROOT     = $OUT_ROOT"
echo "[INFO] n_concepts   = $N_CONCEPTS_FIXED (fixed)"
echo "[INFO] alpha sweep  = ${ALPHAS[*]}"

# -----------------------------
# 1) Alpha ablation runs (copy per alpha like run_multi)
# -----------------------------
RESOURCE_INF="$RESOURCE_DIR/inference"
RESOURCE_FEAT="$RESOURCE_DIR/features"

for a in "${ALPHAS[@]}"; do
  RUN_DIR="$OUT_ROOT/alpha_${a}"
  mkdir -p "$RUN_DIR/inference"
  echo "------------------------------------"
  echo "[RUN] n_concepts=2  DL_ALPHA=$a"
  echo "[DIR] $RUN_DIR"
  echo "------------------------------------"

  # copy precomputed inference/features for this alpha run (no overwrite)
  rsync -a --ignore-existing "$RESOURCE_INF/" "$RUN_DIR/inference/"
  rsync -a --ignore-existing "$RESOURCE_FEAT/" "$RUN_DIR/features/"

  # run pipeline with fixed n=2 and this alpha
  n_concepts="$N_CONCEPTS_FIXED" DL_ALPHA="$a" OUTPUT_DIR="$RUN_DIR" \
    bash "$PIPELINE" --output-dir "$RUN_DIR"
done

echo "[DONE] Alpha ablation finished under: $OUT_ROOT"
