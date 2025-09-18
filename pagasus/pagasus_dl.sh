#!/usr/bin/env bash

# Robust SLURM-aware install wrapper
# - Installs optional apt and conda packages if available and permitted
# - Always upgrades pip and installs from requirements.txt at repo root
# - Ensures only SLURM local rank 0 per node performs installation; others wait
# - Runs the wrapped command passed as arguments after installation

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

# Optional overrides via env vars
#   APT_PACKAGES     : space-separated apt packages to install (default: empty)
#   EXTRA_PIP_PACKAGES: extra pip packages (space-separated) to install after base set (default: empty)
#   USE_CONDA        : force using conda if available (default: auto)
#   PYTHON_VERSION   : python version to create env with (default: 3.9)
#   VENV_PATH        : path to create venv if conda is unavailable (default: $repo_root/.venv)
#   TORCH_INDEX_URL  : index URL for torch/torchvision (default: https://download.pytorch.org/whl/cu126)
#   TORCH_VERSION    : optional torch version to install (e.g., 2.4.0)
#   TORCHVISION_VERSION : optional torchvision version to install
#   FORCE_TORCH_INSTALL : set to 1 to force reinstall even if found (default: 0)
#   SKIP_TORCH       : set to 1 to skip torch/torchvision installation (default: 0)
APT_PACKAGES="${APT_PACKAGES:-}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-}"
USE_CONDA="${USE_CONDA:-auto}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
VENV_PATH="${VENV_PATH:-$repo_root/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
FORCE_TORCH_INSTALL="${FORCE_TORCH_INSTALL:-0}"
SKIP_TORCH="${SKIP_TORCH:-0}"

JOBID="${SLURM_JOBID:-${SLURM_JOB_ID:-$$}}"
LOCALID="${SLURM_LOCALID:-0}"
NODEID="${SLURM_NODEID:-0}"

# Per-node done flag in /tmp (node-local)
DONEFILE="/tmp/install_done_${JOBID}_${NODEID}"
LOCKDIR="${DONEFILE}.lock"

log() { echo "[pagasus-install][node=${NODEID} job=${JOBID}] $*"; }

run_apt() {
  # Try to run apt-get only if available and we have privileges
  if ! command -v apt-get >/dev/null 2>&1; then
    log "apt-get not found; skipping apt installs"
    return 0
  fi

  if [[ -z "$APT_PACKAGES" ]]; then
    log "No APT_PACKAGES specified; skipping apt installs"
    return 0
  fi

  local apt_cmd="apt-get"
  if (( EUID != 0 )); then
    if command -v sudo >/dev/null 2>&1; then
      apt_cmd="sudo -n apt-get"
    else
      log "Not root and sudo not available; skipping apt installs"
      return 0
    fi
  fi

  log "Installing apt packages: $APT_PACKAGES"
  DEBIAN_FRONTEND=noninteractive $apt_cmd update
  DEBIAN_FRONTEND=noninteractive $apt_cmd install -y $APT_PACKAGES
  $apt_cmd clean
  # Best-effort cache cleanup
  rm -rf /var/lib/apt/lists/* || true
}

activate_conda_env() {
  # Initialize conda in this shell and activate env
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    if conda env list | awk '{print $1}' | grep -qx "xlvlms"; then
      log "Conda env 'xlvlms' exists; activating"
    else
      log "Creating conda env 'xlvlms' with Python ${PYTHON_VERSION}"
      conda create -y -n xlvlms "python=${PYTHON_VERSION}"
    fi
    conda activate xlvlms
    return 0
  fi
  return 1
}

activate_venv_env() {
  local py_bin="python3"
  if command -v python${PYTHON_VERSION} >/dev/null 2>&1; then
    py_bin="python${PYTHON_VERSION}"
  elif command -v python3 >/dev/null 2>&1; then
    py_bin="python3"
  elif command -v python >/dev/null 2>&1; then
    py_bin="python"
  fi

  if [[ ! -d "$VENV_PATH" ]]; then
    log "Creating venv at $VENV_PATH with ${py_bin}"
    "$py_bin" -m venv "$VENV_PATH"
  else
    log "Using existing venv at $VENV_PATH"
  fi
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
}

setup_python_env() {
  case "$USE_CONDA" in
    force|true)
      if activate_conda_env; then return 0; else log "Conda forced but not available; falling back to venv"; fi
      ;;
    auto|*)
      if activate_conda_env; then return 0; fi
      ;;
  esac
  activate_venv_env
}

install_readme_deps() {
  # We assume an environment is already active
  log "Upgrading pip3"
  python -m pip3 install --upgrade pip3

  install_torch_if_needed

  log "Installing core Python dependencies"
  pip install tqdm git+https://github.com/bckim92/language-evaluation.git bert-score clip psutil spacy timm accelerate

  log "Downloading spaCy model en_core_web_sm"
  python -m spacy download en_core_web_sm || true

  log "Installing Qwen model utils"
  pip install qwen-vl-utils

  # Optional Java via conda if conda env is active
  if command -v conda >/dev/null 2>&1 && conda info --envs | grep -q "* xlvlms"; then
    log "Installing openjdk via conda-forge"
    conda install -y -c conda-forge openjdk || true
  else
    log "Conda not active; skipping openjdk installation"
  fi

  log "Downloading COCO evaluation data via language_evaluation"
  python - <<'PY'
import language_evaluation
try:
    language_evaluation.download('coco')
except Exception as e:
    print(f"Warning: COCO data download failed: {e}")
PY

  if [[ -n "$EXTRA_PIP_PACKAGES" ]]; then
    log "Installing extra pip packages: $EXTRA_PIP_PACKAGES"
    pip install $EXTRA_PIP_PACKAGES || true
  fi
}

install_torch_if_needed() {
  if [[ "$SKIP_TORCH" == "1" ]]; then
    log "Skipping torch/torchvision installation as requested (SKIP_TORCH=1)"
    return 0
  fi

  local have_torch=0
  python - <<'PY' && have_torch=1 || have_torch=0
try:
    import torch
    import torchvision
    print("torch:", torch.__version__)
    try:
        print("cuda:", torch.version.cuda)
    except Exception:
        pass
except Exception as e:
    raise SystemExit(1)
PY

  if [[ "$have_torch" -eq 1 && "$FORCE_TORCH_INSTALL" != "1" ]]; then
    log "torch/torchvision already present; skipping install (set FORCE_TORCH_INSTALL=1 to override)"
    return 0
  fi

  log "Installing torch/torchvision from $TORCH_INDEX_URL"
  if ! pip install --index-url "$TORCH_INDEX_URL" torch torchvision; then
    log "Non-fatal: torch install failed (likely container has pinned NV torch). Continuing with existing torch."
  fi
}

do_install() {
  log "Starting installation"
  run_apt
  setup_python_env
  install_readme_deps
  log "Installation completed"
}

under_slurm="false"
if [[ -n "${SLURM_JOBID:-${SLURM_JOB_ID:-}}" ]]; then
  under_slurm="true"
fi

if [[ "$under_slurm" == "true" ]]; then
  if [[ "$LOCALID" == "0" ]]; then
    # Local rank 0 performs install per node
    if [[ -f "$DONEFILE" ]]; then
      log "Install already done for this node; continuing"
    else
      # Simple lock to avoid duplicate installs if multiple rank 0 somehow race
      trap 'rm -rf "$LOCKDIR"' EXIT
      if mkdir "$LOCKDIR" 2>/dev/null; then
        do_install
        touch "$DONEFILE"
      else
        log "Another process is installing; waiting for done flag..."
        while [[ ! -f "$DONEFILE" ]]; do sleep 2; done
      fi
    fi
  else
    # Non-zero local ranks just wait
    log "Waiting for local rank 0 to finish installation..."
    while [[ ! -f "$DONEFILE" ]]; do sleep 2; done
  fi
else
  # Not under SLURM -> do install once
  do_install
fi

# Run wrapped command
if [[ $# -gt 0 ]]; then
  log "Executing: $*"
  exec "$@"
else
  log "No command provided after install. Exiting."
fi
