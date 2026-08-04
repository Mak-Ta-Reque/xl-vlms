#!/usr/bin/env python3
"""Filter a concept->images mapping to the vocab-filtered top-N concepts.

Standalone port of the concept-selection stage from scripts/run_full_pipeline.py
so the shell orchestrator can apply NUM_CONCEPT / CONCEPTS_VOCAB too.

Prints the path of the mapping to use on stdout (the filtered file, or the
original mapping when --num_concept <= 0), so callers can do:

    FILTERED=$(python preprocessing/select_top_concepts.py --mapping_json ... )
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import inflect

p = inflect.engine()


def _normalize_concept_name(name: str) -> str:
    """Normalize concept names for case-insensitive singular/plural matching."""
    value = name.strip().casefold()
    if not value:
        return value
    return " ".join(p.singular_noun(word) or word for word in value.split())


def _load_concept_vocab(vocab_path: Path) -> Optional[set]:
    if not vocab_path.exists():
        return None
    vocab = set()
    with open(vocab_path, "r", encoding="utf-8") as handle:
        for line in handle:
            concept = line.strip()
            if not concept or concept.startswith("#"):
                continue
            vocab.add(_normalize_concept_name(concept))
    return vocab


def select_top_concepts(
    mapping_json: Path,
    num_concept: int,
    concepts_vocab: Optional[Path] = None,
) -> Path:
    """Create and return a filtered concept map containing vocab-filtered top-N concepts."""
    if num_concept <= 0 and concepts_vocab is None:
        # No vocab filter and no top-N cap requested: nothing to do.
        return mapping_json

    if not mapping_json.exists():
        raise FileNotFoundError(f"Concept map not found: {mapping_json}")

    with open(mapping_json, "r", encoding="utf-8") as handle:
        concept_mapping = json.load(handle)

    if not isinstance(concept_mapping, dict):
        raise ValueError(f"Expected a JSON object in {mapping_json}")

    vocab = None
    if concepts_vocab is not None:
        vocab = _load_concept_vocab(concepts_vocab)
        if vocab is None:
            print(f"WARNING: Concept vocab not found: {concepts_vocab}. Using all concepts.", file=sys.stderr)
        else:
            print(f"Loaded {len(vocab)} concept names from {concepts_vocab}", file=sys.stderr)

    filtered_candidates = concept_mapping.items()
    if vocab is not None:
        filtered_candidates = [
            (tag, images)
            for tag, images in concept_mapping.items()
            if _normalize_concept_name(tag) in vocab
        ]
        print(
            f"Vocab filter kept {len(filtered_candidates)} of {len(concept_mapping)} concepts",
            file=sys.stderr,
        )

    sorted_concepts = sorted(
        filtered_candidates,
        key=lambda item: (-len(item[1]), item[0]),
    )
    # num_concept <= 0 means "no top-N cap" (keep every vocab-filtered
    # concept), not "keep zero/negative" — min(num_concept, len(...)) would
    # otherwise slice off the last concept for num_concept=-1.
    keep_n = len(sorted_concepts) if num_concept <= 0 else min(num_concept, len(sorted_concepts))
    selected_concepts = sorted_concepts[:keep_n]
    filtered_mapping = {tag: images for tag, images in selected_concepts}

    vocab_suffix = ""
    if concepts_vocab is not None:
        vocab_suffix = f"_{concepts_vocab.stem}"
    filtered_path = mapping_json.with_name(
        f"{mapping_json.stem}{vocab_suffix}_top{len(filtered_mapping)}{mapping_json.suffix}"
    )
    if filtered_path.exists():
        return filtered_path

    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    with open(filtered_path, "w", encoding="utf-8") as handle:
        json.dump(filtered_mapping, handle, indent=2, ensure_ascii=False)

    print(
        f"Filtered concept map saved to {filtered_path} "
        f"(top {len(filtered_mapping)} of {len(sorted_concepts)} candidates)",
        file=sys.stderr,
    )
    print(f"Selected tags: {[tag for tag, _ in selected_concepts]}", file=sys.stderr)
    return filtered_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapping_json", type=Path, required=True)
    ap.add_argument("--num_concept", type=int, default=-1, help="-1 keeps all concepts")
    ap.add_argument("--concepts_vocab", type=Path, default=None)
    args = ap.parse_args()

    result = select_top_concepts(args.mapping_json, args.num_concept, args.concepts_vocab)
    print(result)


if __name__ == "__main__":
    main()
