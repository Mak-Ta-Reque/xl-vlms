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
#   APT_PACKAGES   : space-separated apt packages to install (default: empty)
#   CONDA_PACKAGES : space-separated conda packages to install (default: empty)
#   REQUIREMENTS_FILE : path to requirements file (default: $repo_root/requirements.txt)
#   PYTHON_EXE     : python executable to use (default: python or python3)
APT_PACKAGES="${APT_PACKAGES:-}"
CONDA_PACKAGES="${CONDA_PACKAGES:-}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$repo_root/requirements.txt}"

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

run_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    log "conda not found; skipping conda installs"
    return 0
  fi
  if [[ -z "$CONDA_PACKAGES" ]]; then
    log "No CONDA_PACKAGES specified; skipping conda installs"
    return 0
  fi

  log "Installing conda packages: $CONDA_PACKAGES"
  conda install -y $CONDA_PACKAGES
}

run_pip() {
  local py="${PYTHON_EXE:-python}"
  if ! command -v "$py" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
      py="python3"
    fi
  fi

  log "Upgrading pip"
  "$py" -m pip install --upgrade pip

  if [[ -f "$REQUIREMENTS_FILE" ]]; then
    log "Installing pip requirements from $REQUIREMENTS_FILE"
    "$py" -m pip install -r "$REQUIREMENTS_FILE"
  else
    log "Requirements file not found at $REQUIREMENTS_FILE; skipping"
  fi
}

do_install() {
  log "Starting installation"
  run_apt
  run_conda
  run_pip
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
