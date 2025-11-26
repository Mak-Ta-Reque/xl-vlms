#!/usr/bin/env bash
set -euo pipefail

SRC="/mnt/abka03/xlvlm_data/imagenet_1000/val"
DST="/mnt/abka03/xlvlm_data/colorful/val"

mkdir -p "$DST"

# List of class keywords to match
CLASSES=(
    "zebra"
    "tiger"
    "dalmatian"
    "leopard"
    "cheetah"
    "ladybug"
    "pinecone"
    "barnacle"
    "coral"
    "rock beauty"
    "pineapple"
    "anemone"
    "soap bubble"
    "wicker"
    "sweater"
    "carpet"
    "bath towel"
    "cardigan"
    "crocodile"
    "iguana"
    "armadillo"
    "peacock"
    "mushroom"
    "tree"
    "coconut"
    "acorn"
    "urchin"
)

echo "Copying matching ImageNet classes..."

for keyword in "${CLASSES[@]}"; do
    # search folder names and class name files
    match=$(grep -ril "$keyword" "$SRC")

    if [ -n "$match" ]; then
        for m in $match; do
            class_dir=$(dirname "$m")
            class_name=$(basename "$class_dir")

            echo "→ Copying: $class_name  (matched: $keyword)"
            mkdir -p "$DST/$class_name"
            cp "$class_dir"/* "$DST/$class_name"/
        done
    else
        echo "× No match found for: $keyword"
    fi
done

echo "✔ Done!"
