#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Launch the Interactive VLM Concept Explorer demo            ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Prerequisites:
#   conda activate xlvlm-v1
#   pip install streamlit spacy pycocotools
#   python -m spacy download en_core_web_sm
#
# Usage:
#   bash demo/run_demo.sh                    # uses defaults from .env
#   CONCEPT_PATH=/path/to/raw.pth bash demo/run_demo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Use conda xlvlm-v1 environment
CONDA_RUN="conda run --no-capture-output -n xvlm-clean"

# Ensure spaCy model is available (download silently if missing)
$CONDA_RUN python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null \
  || $CONDA_RUN python -m spacy download en_core_web_sm

exec $CONDA_RUN streamlit run demo/vlm_binary_demo.py \
    --server.port "${DEMO_PORT:-8501}" \
    --server.headless true \
    "$@"
