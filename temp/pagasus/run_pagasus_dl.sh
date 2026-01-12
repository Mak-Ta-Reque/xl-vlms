#!/usr/bin/env bash

# Wrapper to run pagasus_dl.sh inside the specified container via srun.
# You can pass any command after '--' and it will be executed after installation.

set -euo pipefail

this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$this_dir/.." && pwd)"

# Defaults from user request; adjust via env vars if needed
PARTITION="${PARTITION:-A100-IML}"
NTASKS="${NTASKS:-1}"
GPUS_PER_TASK="${GPUS_PER_TASK:-0}"
MEM_PER_CPU="${MEM_PER_CPU:-16G}"
CONTAINER_IMAGE_PATH="${CONTAINER_IMAGE_PATH:-/netscratch/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh}"
SOFTWARE_MOUNT="${SOFTWARE_MOUNT:-/netscratch/software:/netscratch/software:ro}"
USER_MOUNT="${USER_MOUNT:-/netscratch/kadir:/netscratch/kadir}"
WORKDIR="${WORKDIR:-$repo_root}"
TIME="${TIME:-0:30:00}"

# Split args at '--' to forward to pagasus_dl.sh
WRAPPED_CMD=( )
if [[ "$#" -gt 0 ]]; then
  # collect all args as the wrapped command
  WRAPPED_CMD=("$@")
fi

srun \
  -p "$PARTITION" \
  --ntasks "$NTASKS" \
  --gpus-per-task "$GPUS_PER_TASK" \
  --mem-per-cpu "$MEM_PER_CPU" \
  --container-image="$CONTAINER_IMAGE_PATH" \
  --container-mounts="$SOFTWARE_MOUNT","$USER_MOUNT","$(pwd)":"$(pwd)" \
  --container-workdir="$(pwd)" \
  --time="$TIME" \
  "$this_dir/pagasus_dl.sh" "${WRAPPED_CMD[@]}"
