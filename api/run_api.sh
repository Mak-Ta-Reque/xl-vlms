#!/bin/bash
# Script to run the FastAPI server with conda environment

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source .env as single source of truth
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi

echo "=================================================="
echo "XL-VLMS Pipeline API Server"
echo "=================================================="
echo "Root Directory: $ROOT_DIR"
echo "API Directory: $SCRIPT_DIR"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not available in PATH"
    echo "Please ensure conda is installed and initialized"
    exit 1
fi

# Check if running in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "WARNING: No conda environment is activated"
    echo "Please activate your conda environment first:"
    echo "  conda activate your_env_name"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
fi

# Check if FastAPI dependencies are installed
echo ""
echo "Checking dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo "FastAPI not found. Installing API dependencies..."
    pip install -r "$ROOT_DIR/requirements.txt"
else
    echo "Dependencies OK"
fi

# Check if pipeline script exists
PIPELINE_SCRIPT="$ROOT_DIR/scripts/run_full_pipeline_without_coroping.sh"
if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "ERROR: Pipeline script not found at: $PIPELINE_SCRIPT"
    exit 1
fi

# Make pipeline script executable
chmod +x "$PIPELINE_SCRIPT"

# Create necessary directories
mkdir -p "$ROOT_DIR/data/train"
mkdir -p "$ROOT_DIR/outputs"

echo ""
echo "=================================================="
echo "Starting API Server"
echo "=================================================="
echo "Server will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"
echo "=================================================="
echo ""

# Run with uvicorn from project root
cd "$ROOT_DIR"
python -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info
