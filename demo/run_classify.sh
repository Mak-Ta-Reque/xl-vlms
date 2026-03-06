#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────
#  Run the VLM Classify demo (FastAPI backend + React frontend).
#
#  Usage:
#    bash demo/run_classify.sh
#
#  Environment variables:
#    CLASSIFY_API_PORT  — backend port  (default 8501)
#    FRONTEND_PORT      — Vite dev port (default 5173)
#    CONDA_ENV          — conda env     (default xlvlm-v1)
# ───────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Source .env for DEVICE and other settings
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

# ─── Configurable vars ────────────────────────────────────────────────
API_PORT="${CLASSIFY_API_PORT:-8501}"
FE_PORT="${FRONTEND_PORT:-5173}"
CONDA_ENV="${CONDA_ENV:-xvlm-demo}"
CONDA_RUN="conda run --no-capture-output -n $CONDA_ENV"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# ─── Ensure spaCy model is available ──────────────────────────────────
echo "» Checking spaCy model..."
$CONDA_RUN python -c "
import spacy
try:
    spacy.load('en_core_web_sm')
except OSError:
    print('  Downloading en_core_web_sm...')
    spacy.cli.download('en_core_web_sm')
print('  spaCy model OK')
" 2>/dev/null || true

# ─── Ensure frontend deps are installed ───────────────────────────────
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "» Installing frontend dependencies..."
    (cd "$SCRIPT_DIR/frontend" && npm install)
fi

# ─── PIDs for cleanup ────────────────────────────────────────────────
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    echo "» Shutting down..."
    [ -n "$BACKEND_PID" ]  && kill "$BACKEND_PID"  2>/dev/null || true
    [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "» Done."
}
trap cleanup EXIT INT TERM

# ─── Start FastAPI backend ────────────────────────────────────────────
echo "» Starting backend on port $API_PORT..."
$CONDA_RUN python -m uvicorn demo.classify_api:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --log-level info &
BACKEND_PID=$!

# Wait for backend to be ready
echo "  Waiting for backend health check..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$API_PORT/api/health" > /dev/null 2>&1; then
        echo "  Backend ready."
        break
    fi
    if [ $i -eq 60 ]; then
        echo "  WARNING: Backend health check timed out (may still be loading model)."
    fi
    sleep 2
done

# ─── Start Vite dev server ────────────────────────────────────────────
echo "» Starting frontend on port $FE_PORT..."
(cd "$SCRIPT_DIR/frontend" && npx vite --port "$FE_PORT" --host) &
FRONTEND_PID=$!

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Backend:   http://localhost:$API_PORT/api/health"
echo "  Frontend:  http://localhost:$FE_PORT"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ─── Wait for both processes ──────────────────────────────────────────
wait
