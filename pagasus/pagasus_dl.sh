#!/usr/bin/env bash

# Robust SLURM-aware install wrapper
# - Host installs only (no conda/venv): use system Python with pip --user
# - Optional apt packages (e.g., openjdk) if permissions allow
# - Ensures only SLURM local rank 0 per node performs installation; others wait
# - Runs the wrapped command passed as arguments after installation

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
ACTIVE_ENV="system" # fixed to system host installs

# Optional overrides via env vars
#   APT_PACKAGES     : space-separated apt packages to install (default: empty)
#   EXTRA_PIP_PACKAGES: extra pip packages (space-separated) to install after base set (default: empty)
#   INSTALL_OPENJDK  : set to 1 to try installing openjdk via apt (default: 0)
#   TORCH_INDEX_URL  : index URL for torch/torchvision (default: https://download.pytorch.org/whl/cu126)
#   TORCH_VERSION    : optional torch version to install (e.g., 2.4.0)
#   TORCHVISION_VERSION : optional torchvision version to install
#   FORCE_TORCH_INSTALL : set to 1 to force reinstall even if found (default: 0)
#   SKIP_TORCH       : set to 1 to skip torch/torchvision installation (default: 0)
APT_PACKAGES="${APT_PACKAGES:-}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-}"
INSTALL_OPENJDK="${INSTALL_OPENJDK:-0}"
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

setup_python_env() { ACTIVE_ENV="system"; }

install_readme_deps() {
  # We assume an environment is already active
  local PIP_USER_FLAG="--user"

  log "Upgrading pip"
  python -m pip install --upgrade pip $PIP_USER_FLAG || true

  reinstall_torch "$PIP_USER_FLAG"

  log "Installing core Python dependencies"
  pip install $PIP_USER_FLAG tqdm git+https://github.com/bckim92/language-evaluation.git bert-score clip psutil spacy timm accelerate

  log "Downloading spaCy model en_core_web_sm"
  python -m spacy download en_core_web_sm || true

  log "Installing Qwen model utils"
  pip install $PIP_USER_FLAG qwen-vl-utils

  # Optional Java via apt if requested and permitted
  if [[ "$INSTALL_OPENJDK" == "1" ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      log "Attempting to install openjdk via apt"
      local apt_cmd="apt-get"
      if (( EUID != 0 )) && command -v sudo >/dev/null 2>&1; then
        apt_cmd="sudo -n apt-get"
      fi
      DEBIAN_FRONTEND=noninteractive $apt_cmd update || true
      DEBIAN_FRONTEND=noninteractive $apt_cmd install -y openjdk-17-jre-headless || true
      $apt_cmd clean || true
      rm -rf /var/lib/apt/lists/* || true
    else
      log "apt-get not available; skipping openjdk installation"
    fi
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
    pip install $PIP_USER_FLAG $EXTRA_PIP_PACKAGES || true
  fi
}

reinstall_torch() {
  local PIP_USER_FLAG="$1"
  if [[ "$SKIP_TORCH" == "1" ]]; then
    log "Skipping torch/torchvision install (SKIP_TORCH=1)"
    return 0
  fi

  log "Purging existing torch packages (pip)"
  pip uninstall -y torch torchvision torchaudio torchtext || true

  log "Installing torch/torchvision from $TORCH_INDEX_URL"
  if ! pip install $PIP_USER_FLAG --index-url "$TORCH_INDEX_URL" torch torchvision; then
    log "Non-fatal: torch install failed; continuing."
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
