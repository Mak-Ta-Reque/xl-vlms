#!/usr/bin/env bash

# DL Feature Decomposition Orchestrator
# Accepts the same interface as CGDL variant for pipeline compatibility.
# Usage:
#   run_feature_decompose_dl.sh <HF_HOME> <model_name> <feature_save_dir>
# Falls back to env: HF_HOME, VLM_MODEL/MODEL_NAME, FEATURES_DIR/FEATURE_SAVE_DIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source .env as single source of truth
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi

# --------------------------
# Resolve inputs
# --------------------------
HF_HOME_INPUT=${1:-${HF_HOME:-"/mnt/abka03/huggingface/hub"}}
MODEL_NAME_INPUT=${2:-${VLM_MODEL:-${MODEL_NAME:-"google/gemma-3n-E4B-it"}}}
FEATURE_SAVE_DIR_INPUT=${3:-${FEATURES_DIR:-${FEATURE_SAVE_DIR:-"/mnt/abka03/Projects/xl-vlms/outputs/dl/train"}}}

# Resolve Python interpreter
if command -v "${PYTHON:-${PYTHON_BIN:-python}}" >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON:-${PYTHON_BIN:-python}}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

if [ -z "$HF_HOME_INPUT" ] || [ -z "$MODEL_NAME_INPUT" ] || [ -z "$FEATURE_SAVE_DIR_INPUT" ]; then
    echo "Usage: $0 <HF_HOME> <model_name> <feature_save_dir>"
    exit 1
fi

start_time=$(date +%s)

export HF_HOME="$HF_HOME_INPUT"

# Paths
# Prefer explicit SAVE_DIR (used by pipeline) if provided; else assume <root>/features
if [[ -n "${SAVE_DIR:-}" ]]; then
    feature_dir="$SAVE_DIR/features"
    feature_save_dir="$(dirname "$SAVE_DIR")"
else
    feature_save_dir="$FEATURE_SAVE_DIR_INPUT"
    if [[ -d "${feature_save_dir}/features" ]]; then
        feature_dir="${feature_save_dir}/features"
    else
        feature_dir="$feature_save_dir"
    fi
fi
# Respect orchestrator's DECOMP_DIR if provided; default to features/concept
analysis_save_dir="${DECOMP_DIR:-${feature_save_dir}/concept}"
mkdir -p "$analysis_save_dir"

# Model settings
model_name="$MODEL_NAME_INPUT"
feature_module="model.language_model.norm"

# Concept extraction settings (DL defaults)
n_concepts=${NUM_CONCEPTS:-1000}
max_iterations=${MAX_ITERATIONS:-100}
normalizations=("gl")

# Sparsity regularization: higher = sparser decomposition.
# dl_alpha feeds snmf/nndl (sklearn DictionaryLearning L1 alpha, default 20).
# sae_sparsity_lambda feeds sae/sae2 soft L1 penalty (default 0.0005).
# sae_target_sparsity enables hard top-k activation on sae/sae2, guaranteeing
# this exact zero-fraction (e.g. 0.99) instead of relying on the soft penalty.
dl_alpha="${DL_ALPHA:-20}"
sae_sparsity_lambda="${SAE_SPARSITY_LAMBDA:-0.0005}"
sae_target_sparsity="${SAE_TARGET_SPARSITY:-}"

# Decomposition methods to run (respect orchestrator-selected method if set)
if [[ -n "${DECOMP_METHOD:-}" ]]; then
    decomposition_methods=("${DECOMP_METHOD}")
else
    decomposition_methods=("snmf" "sae2" "pca" "simple")
fi

# Base analysis names
base_analysis_name="decompose_activations_text_grounding_image_grounding"
base_regrunding_name="redefine_activations_text_grounding"

# --------------------------
# Combine Features (required before decomposition)
# --------------------------
echo "========================"
echo "Combine Features"
echo "========================"
echo "Combining all .pth files in $feature_dir ..."
# If only combined_features.pth exists, skip combine and don't delete anything
combined_path="$feature_dir/combined_features.pth"
mapfile -t original_pths < <(find "$feature_dir" -maxdepth 1 -type f -name '*.pth' ! -name 'combined_features.pth')
if [[ -f "$combined_path" && ${#original_pths[@]} -eq 0 ]]; then
    echo "Only combined_features.pth present; skipping combine and deletion."
else
    del_flag=""
    if [[ "${DELETE_AFTER_COMBINE:-1}" -eq 1 ]]; then
        del_flag="--delete"
    fi
    "$PYTHON_BIN" src/combine_features.py "$feature_dir" $del_flag
fi
echo

# --------------------------
# MAIN LOOP (single-pass on combined features)
# --------------------------
for decomposition in "${decomposition_methods[@]}"; do
    echo "=== Running DL decomposition: $decomposition ==="

    # Per-method concept-count override, e.g. NUM_CONCEPTS_SNMF=460,
    # NUM_CONCEPTS_SAE2=100. Falls back to the generic n_concepts above.
    method_upper=$(echo "$decomposition" | tr '[:lower:]' '[:upper:]')
    method_n_concepts_var="NUM_CONCEPTS_${method_upper}"
    method_n_concepts="${!method_n_concepts_var:-$n_concepts}"

    sparsity_extra_args=(--dl_alpha "$dl_alpha" --sae_sparsity_lambda "$sae_sparsity_lambda")
    if [[ -n "$sae_target_sparsity" ]]; then
        sparsity_extra_args+=(--sae_target_sparsity "$sae_target_sparsity")
    fi

    analysis_name="${base_analysis_name}_${decomposition}"

    combined_pth="$feature_dir/combined_features.pth"
    if [[ ! -f "$combined_pth" ]]; then
        echo "[WARN] combined_features.pth not found in $feature_dir. Falling back to all .pth files."
        # Fallback: pick all .pth and process the first one
        shopt -s nullglob
        pths=( "$feature_dir"/*.pth )
        if [[ ${#pths[@]} -eq 0 ]]; then
            echo "[ERROR] No .pth features found in $feature_dir" >&2
            exit 1
        fi
        combined_pth="${pths[0]}"
    fi

    # Decompose once on the combined features and save directly to target path
    target_path="${analysis_save_dir}/combined_concept_${decomposition}_raw.pth"

    "$PYTHON_BIN" src/analyse_features.py \
        --model_name "$model_name" \
        --analysis_name "$analysis_name" \
        --features_path "$combined_pth" \
        --module_to_decompose "$feature_module" \
        --num_concepts "$method_n_concepts" \
        --decomposition_method "$decomposition" \
        --save_filename "combined_concept_${decomposition}_raw" \
        --save_dir "$analysis_save_dir" \
        "${sparsity_extra_args[@]}"

    # Move/normalize filename to expected target (handle various prefixes and locations)
    if [[ ! -f "$target_path" ]]; then
        shopt -s nullglob
        candidates=(
            "${analysis_save_dir}/decompose_activations_${decomposition}_combined_concept_${decomposition}_raw.pth"
            "${analysis_save_dir}/decompose_activations_text_grounding_image_grounding_${decomposition}_combined_concept_${decomposition}_raw.pth"
            "${analysis_save_dir}/"*"_${decomposition}_combined_concept_${decomposition}_raw.pth"
            "${analysis_save_dir}/"*"combined_concept_${decomposition}_raw.pth"
            "${analysis_save_dir}/analysis/decompose_activations_${decomposition}_combined_concept_${decomposition}_raw.pth"
            "${analysis_save_dir}/analysis/decompose_activations_text_grounding_image_grounding_${decomposition}_combined_concept_${decomposition}_raw.pth"
            "${analysis_save_dir}/analysis/"*"_${decomposition}_combined_concept_${decomposition}_raw.pth"
            "${analysis_save_dir}/analysis/"*"combined_concept_${decomposition}_raw.pth"
        )
        moved=0
        for cand in "${candidates[@]}"; do
            if [[ -f "$cand" ]]; then
                mv -f "$cand" "$target_path"
                moved=1
                break
            fi
        done
        if [[ $moved -eq 0 && ! -f "$target_path" ]]; then
            echo "[WARN] Could not locate saved analysis file to move; expected at $target_path" >&2
        else
            echo "Saved concepts at $target_path"
        fi
    fi

    echo "=== Completed DL decomposition: $decomposition ==="
    echo
done

# --------------------------
# TIME REPORTING
# --------------------------
end_time=$(date +%s)
seconds=$((end_time - start_time))
hours=$((seconds / 3600))
minutes=$(((seconds % 3600) / 60))
secs=$((seconds % 60))
echo "All DL decompositions completed in ${hours}h ${minutes}m ${secs}s"