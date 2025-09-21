#!/usr/bin/env bash
# For each run under outputs/ matching given tokens, generate method-overlaid
# mean curves across ranks (deletion and insertion) with interpolation.
# Uses scripts/plot_eval_summary_across_methods.py.
#
# Usage:
#   scripts/plot_methods_mean_across_runs.sh \
#     [--outputs-dir PATH] \
#     [--runs "cdgl,dl"] \
#     [--methods "pca,simple,random,snmf"] \
#     [--ymin 3.62e-6] [--ymax 4.0e-6]
#
# Output: For each run, saves into <run>/plots:
#   - c_deletion_token_methods_mean.(png|pdf)
#   - c_insertion_token_methods_mean.(png|pdf)

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_AGG="$SCRIPT_DIR/plot_eval_summary_across_methods.py"

OUTPUTS_DIR="$ROOT_DIR/outputs"
RUN_TOKENS="cdgl,dl"
METHODS=""
YMIN="3.62e-6"
YMAX="4.0e-6"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--outputs-dir PATH] [--runs "cdgl,dl"] [--methods CSV] [--ymin VAL] [--ymax VAL]

Options:
  --outputs-dir PATH   Root outputs directory (default: $OUTPUTS_DIR)
  --runs CSV           Comma-separated tokens to match run dirs (default: $RUN_TOKENS)
  --methods CSV        Comma-separated method names to include (default: auto-detect)
  --ymin VAL           Lower y-axis limit (default: $YMIN)
  --ymax VAL           Upper y-axis limit (default: $YMAX)
  -h, --help           Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outputs-dir) OUTPUTS_DIR="$2"; shift 2 ;;
    --runs) RUN_TOKENS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --ymin) YMIN="$2"; shift 2 ;;
    --ymax) YMAX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -f "$PY_AGG" ]]; then
  echo "Missing: $PY_AGG" >&2
  exit 1
fi

IFS=',' read -r -a TOKENS <<<"$RUN_TOKENS"

mapfile -t RUN_DIRS < <(find "$OUTPUTS_DIR" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | while read -r d; do
  for t in "${TOKENS[@]}"; do
    if [[ "$d" == *"$t"* ]]; then
      echo "$OUTPUTS_DIR/$d"
      break
    fi
  done
done | sort)

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "No run directories found under $OUTPUTS_DIR matching tokens: $RUN_TOKENS" >&2
  exit 1
fi

PY_BIN=""
if command -v python3 >/dev/null 2>&1; then PY_BIN=python3; elif command -v python >/dev/null 2>&1; then PY_BIN=python; else echo "Python not found" >&2; exit 1; fi

for RUN in "${RUN_DIRS[@]}"; do
  EVAL_DIR="$RUN/eval"
  PLOTS_DIR="$RUN/plots"
  [[ -d "$EVAL_DIR" ]] || { echo "Skip $(basename "$RUN"): no eval dir" >&2; continue; }
  mkdir -p "$PLOTS_DIR"

  echo "[+] Run: $(basename "$RUN") -> $PLOTS_DIR"
  if [[ -n "$METHODS" ]]; then
    "$PY_BIN" "$PY_AGG" --eval_dir "$EVAL_DIR" --out_dir "$PLOTS_DIR" --methods "$METHODS" --ymin "$YMIN" --ymax "$YMAX"
  else
    "$PY_BIN" "$PY_AGG" --eval_dir "$EVAL_DIR" --out_dir "$PLOTS_DIR" --ymin "$YMIN" --ymax "$YMAX"
  fi
done

echo "Done."
