#!/usr/bin/env bash
# Summarize rank-1 c_insertion/c_deletion mean_prob across methods for selected runs and plot.
#
# - Scans outputs/<run>/eval/<method>/ for:
#     c_insertion_token_rank1.csv
#     c_deletion_token_rank1.csv
# - Computes the mean of the `mean_prob` column per method, then averages across methods per run.
# - Writes a TSV with columns: run\tinsertion_mean\tdeletion_mean
# - Produces two bar plots saved in $OUT_DIR:
#     rank1_insertion_summary.(png|pdf)
#     rank1_deletion_summary.(png|pdf)
#
# Usage:
#   scripts/plot_rank1_summary.sh \
#     [--outputs-dir PATH] \
#     [--runs "cdgl,dl"] \
#     [--methods "pca,simple,random,snmf"] \
#     [--out-dir PATH]
#
# Defaults:
#   outputs-dir: <repo_root>/outputs
#   runs: cdgl,dl     (matches run directories containing these tokens)
#   methods: auto-detect under each run's eval/ (fallback to pca,simple,random,snmf)
#   out-dir: <run>/plots for each run and a combined file/plots in <outputs>/plots

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUTS_DIR="$ROOT_DIR/outputs"
RUN_TOKENS="cdgl,dl"
METHODS=""
OUT_DIR_GLOBAL=""  # if empty, will default to $OUTPUTS_DIR/plots

usage() {
  cat <<EOF
Usage: $(basename "$0") [--outputs-dir PATH] [--runs "cdgl,dl"] [--methods "pca,simple,random,snmf"] [--out-dir PATH]

Options:
  --outputs-dir PATH   Root outputs directory (default: $OUTPUTS_DIR)
  --runs CSV           Comma-separated tokens to match run dirs (default: $RUN_TOKENS)
  --methods CSV        Comma-separated method names; if omitted, auto-detect per run
  --out-dir PATH       Directory for combined plots/TSV (default: <outputs>/plots)
  -h, --help           Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outputs-dir) OUTPUTS_DIR="$2"; shift 2 ;;
    --runs) RUN_TOKENS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --out-dir) OUT_DIR_GLOBAL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "$OUTPUTS_DIR"
OUT_DIR_COMBINED="${OUT_DIR_GLOBAL:-$OUTPUTS_DIR/plots}"
mkdir -p "$OUT_DIR_COMBINED"

IFS=',' read -r -a TOKENS <<<"$RUN_TOKENS"

# Find run directories matching any token
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

SUMMARY_TSV="$OUT_DIR_COMBINED/rank1_summary.tsv"
echo -e "run\tinsertion_mean\tdeletion_mean" > "$SUMMARY_TSV"

csv_mean_prob_col() {
  local csv="$1"
  awk -F',' 'NR==1{for(i=1;i<=NF;i++){if($i=="mean_prob"){col=i}} next} NR>1 && col>0 {sum+=$col; n++} END{if(n>0){printf "%.10f\n", sum/n}else{print "nan"}}' "$csv"
}

detect_methods() {
  local eval_dir="$1"
  if [[ -d "$eval_dir" ]]; then
    find "$eval_dir" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort
  fi
}

for RUN in "${RUN_DIRS[@]}"; do
  EVAL_DIR="$RUN/eval"
  [[ -d "$EVAL_DIR" ]] || { echo "Skip $RUN (no eval dir)" >&2; continue; }

  # Determine methods list for this run
  if [[ -n "$METHODS" ]]; then
    IFS=',' read -r -a MLIST <<<"$METHODS"
  else
    mapfile -t MLIST < <(detect_methods "$EVAL_DIR")
    if [[ ${#MLIST[@]} -eq 0 ]]; then
      MLIST=(pca simple random snmf)
    fi
  fi

  ins_vals=()
  del_vals=()

  for m in "${MLIST[@]}"; do
    mdir="$EVAL_DIR/$m"
    [[ -d "$mdir" ]] || continue

    ins_csv="$mdir/c_insertion_token_rank1.csv"
    del_csv="$mdir/c_deletion_token_rank1.csv"

    if [[ -f "$ins_csv" ]]; then
      if val=$(csv_mean_prob_col "$ins_csv"); then
        ins_vals+=("$val")
      fi
    fi
    if [[ -f "$del_csv" ]]; then
      if val=$(csv_mean_prob_col "$del_csv"); then
        del_vals+=("$val")
      fi
    fi
  done

  avg_list() {
    local arr=("$@")
    local total=0; local count=0
    for v in "${arr[@]}"; do
      if [[ "$v" != "nan" && -n "$v" ]]; then
        total=$(python3 - <<PY || python - <<PY
import sys
print({total}+{v})
PY
)
        count=$((count+1))
      fi
    done
    if [[ $count -gt 0 ]]; then
      python3 - <<PY || python - <<PY
print({total}/$count)
PY
    else
      echo "nan"
    fi
  }

  INS_MEAN=$(avg_list "${ins_vals[@]}" | tr -d '\n' || echo nan)
  DEL_MEAN=$(avg_list "${del_vals[@]}" | tr -d '\n' || echo nan)

  echo -e "$(basename "$RUN")\t$INS_MEAN\t$DEL_MEAN" >> "$SUMMARY_TSV"
done

# Plot using Python if available
PY_BIN=""
if command -v python3 >/dev/null 2>&1; then PY_BIN=python3; elif command -v python >/dev/null 2>&1; then PY_BIN=python; fi

if [[ -n "$PY_BIN" ]]; then
  "$PY_BIN" - <<'PY'
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tsv_path = Path(sys.argv[1])
out_dir = tsv_path.parent

runs, ins, dele = [], [], []
with tsv_path.open() as f:
    r = csv.reader(f, delimiter='\t')
    header = next(r, None)
    for row in r:
        if len(row) < 3: continue
        runs.append(row[0])
        try:
            ins.append(float(row[1]))
        except: ins.append(float('nan'))
        try:
            dele.append(float(row[2]))
        except: dele.append(float('nan'))

def make_bar(values, title, stem):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(values)), values, tick_label=runs)
    ax.set_title(title)
    ax.set_ylabel('mean_prob (rank1, mean across methods)')
    ax.set_xticklabels(runs, rotation=20, ha='right')
    fig.tight_layout()
    fig.savefig(out_dir / f'{stem}.png', dpi=160)
    fig.savefig(out_dir / f'{stem}.pdf', dpi=160)
    plt.close(fig)

make_bar(ins, 'Rank-1 C-Insertion (mean across methods)', 'rank1_insertion_summary')
make_bar(dele, 'Rank-1 C-Deletion (mean across methods)', 'rank1_deletion_summary')
print(f'Wrote: {(out_dir / "rank1_insertion_summary.png").as_posix()}')
print(f'Wrote: {(out_dir / "rank1_deletion_summary.png").as_posix()}')
PY
else
  echo "Python not found; plots were not generated. Data saved to $SUMMARY_TSV" >&2
fi

echo "Summary TSV: $SUMMARY_TSV"
echo "Done."
