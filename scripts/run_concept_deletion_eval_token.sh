#!/usr/bin/env bash
set -euo pipefail

# Runs Concept Deletion Eval - Token (insertion + deletion)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source .env as single source of truth
if [[ -f "$WORKSPACE/.env" ]]; then
  set -a; source "$WORKSPACE/.env"; set +a
fi

# Resolve Python interpreter
if command -v "${PYTHON:-${PYTHON_BIN:-python}}" >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON:-${PYTHON_BIN:-python}}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi
PROGRAM="${WORKSPACE}/eval/concept_deletion_eval.py"

# Environment
export HF_HOME="${HF_HOME:-/mnt/abka03/huggingface/hub}"
: "${HF_TOKEN:=}"

# Args (prefer orchestrator outputs)
RESULTS_JSON="${RESULTS_JSON:-${EXPLAIN_DIR:-${WORKSPACE}/outputs}/vlm_explanations.json}"
CONCEPT_PATH="${CONCEPT_PATH:-${DECOMP_DIR:-${WORKSPACE}/outputs/cdgl/train/concept}/combined_concept_snmf_raw.pth}"
MODEL_NAME="${VLM_MODEL:-${MODEL_NAME:-google/gemma-3n-E4B-it}}"
LAYER_PATH="${LAYER_PATH:-model.language_model.norm}"
MODE="token"
NUM_POINTS="${NUM_POINTS:-20}"
OUT_DIR="${OUT_DIR:-${EVAL_DIR:-${WORKSPACE}/outputs/concept_deletion_token}}"

mkdir -p "${OUT_DIR}"

run_common_args=(
  --results_json "${RESULTS_JSON}"
  --concept_path "${CONCEPT_PATH}"
  --model_name "${MODEL_NAME}"
  --layer_path "${LAYER_PATH}"
  --mode "${MODE}"
  --num_points "${NUM_POINTS}"
  --out_dir "${OUT_DIR}"
)

for RANK in 1 2 3; do
  echo "[Concept Deletion Eval - Token] Insertion run (rank ${RANK})"
  "${PYTHON_BIN}" "${PROGRAM}" "${run_common_args[@]}" --rank "${RANK}" --insertion

  echo "[Concept Deletion Eval - Token] Deletion run (rank ${RANK})"
  "${PYTHON_BIN}" "${PROGRAM}" "${run_common_args[@]}" --rank "${RANK}"

done

