#!/usr/bin/env python3
"""
Build a token-filtered COCO dataset layout from a COCO captions JSON.

The script reads a captions-style COCO JSON with this structure:

    {
      "images": [
        {
          "filepath": "train2014" | "val2014",
          "filename": "COCO_train2014_000000123456.jpg",
          "split": "train" | "val" | "restval" | "test",
          "sentences": [
            {"tokens": ["a", "dog", ...], "raw": "A dog ..."}
          ]
        }
      ]
    }

For each requested token, matching images are copied into:

    output/
      train/<token>/...
      val/<token>/...

The script also writes a split-level JSON manifest:

    output/train/train.json
    output/val/val.json

Each manifest groups records by token and stores the image name plus raw
caption(s) for later evaluation.

If the input directory does not contain the expected COCO files, the script
downloads the 2014 COCO images and annotations and, when needed, synthesizes a
compatible dataset_coco.json from the official caption annotations.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set


COCO_IMAGE_ZIPS = {
    "train2014": "http://images.cocodataset.org/zips/train2014.zip",
    "val2014": "http://images.cocodataset.org/zips/val2014.zip",
}
COCO_ANNOTATIONS_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"


def _normalize_token(token: str) -> str:
    return token.strip().lower()


def _split_token_args(tokens: Sequence[str]) -> List[str]:
    parsed: List[str] = []
    for token in tokens:
        for piece in re.split(r"[\s,]+", token.strip()):
            normalized = _normalize_token(piece)
            if normalized:
                parsed.append(normalized)
    return list(dict.fromkeys(parsed))


def _sentence_matches_token(sentence_tokens: Sequence[str], token: str) -> bool:
    normalized_sentence = [_normalize_token(part) for part in sentence_tokens if _normalize_token(part)]
    target_parts = [_normalize_token(part) for part in token.split() if _normalize_token(part)]
    if not normalized_sentence or not target_parts:
        return False
    if len(target_parts) == 1:
        return target_parts[0] in normalized_sentence
    window = len(target_parts)
    for start in range(0, len(normalized_sentence) - window + 1):
        if normalized_sentence[start : start + window] == target_parts:
            return True
    return False


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(destination)


def _extract_zip(zip_path: Path, destination_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination_dir)


def _download_coco_assets(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)

    for split_name, url in COCO_IMAGE_ZIPS.items():
        split_dir = input_dir / split_name
        if split_dir.exists() and any(split_dir.glob("*.jpg")):
            continue

        zip_path = input_dir / f"{split_name}.zip"
        if not zip_path.exists():
            print(f"Downloading {split_name} images...")
            _download_file(url, zip_path)
        print(f"Extracting {split_name} images...")
        _extract_zip(zip_path, input_dir)

    annotations_dir = input_dir / "annotations"
    train_captions = annotations_dir / "captions_train2014.json"
    val_captions = annotations_dir / "captions_val2014.json"
    if not (train_captions.exists() and val_captions.exists()):
        annotations_zip = input_dir / "annotations_trainval2014.zip"
        if not annotations_zip.exists():
            print("Downloading COCO annotations...")
            _download_file(COCO_ANNOTATIONS_ZIP, annotations_zip)
        print("Extracting COCO annotations...")
        _extract_zip(annotations_zip, input_dir)


def _tokenize_caption(raw_caption: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", raw_caption.lower())


def _build_dataset_json_from_captions(input_dir: Path, dataset_json_path: Path) -> None:
    annotations_dir = input_dir / "annotations"
    caption_files = [
        (annotations_dir / "captions_train2014.json", "train", "train2014"),
        (annotations_dir / "captions_val2014.json", "val", "val2014"),
    ]

    images: List[Dict[str, object]] = []
    imgid = 0
    for captions_path, split, filepath in caption_files:
        if not captions_path.exists():
            continue
        with open(captions_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        annotations_by_image: Dict[int, List[dict]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        for image_info in payload.get("images", []):
            image_id = int(image_info["id"])
            sentences = []
            for annotation in annotations_by_image.get(image_id, []):
                raw_caption = str(annotation.get("caption", "")).strip()
                if not raw_caption:
                    continue
                sentences.append(
                    {
                        "tokens": _tokenize_caption(raw_caption),
                        "raw": raw_caption,
                        "imgid": imgid,
                        "sentid": int(annotation["id"]),
                    }
                )

            if not sentences:
                continue

            images.append(
                {
                    "filepath": filepath,
                    "sentids": [sentence["sentid"] for sentence in sentences],
                    "filename": image_info["file_name"],
                    "imgid": imgid,
                    "split": split,
                    "sentences": sentences,
                    "cocoid": image_id,
                }
            )
            imgid += 1

    dataset_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_json_path, "w", encoding="utf-8") as handle:
        json.dump({"images": images}, handle, indent=2)


def _ensure_dataset_json(input_dir: Path, dataset_json_path: Path) -> Path:
    if dataset_json_path.exists():
        return dataset_json_path

    _download_coco_assets(input_dir)
    if dataset_json_path.exists():
        return dataset_json_path

    print("Synthesizing dataset_coco.json from COCO caption annotations...")
    _build_dataset_json_from_captions(input_dir, dataset_json_path)
    return dataset_json_path


def _resolve_image_path(input_dir: Path, image_entry: Mapping[str, object]) -> Path:
    filepath = str(image_entry.get("filepath", "")).strip()
    filename = str(image_entry.get("filename", "")).strip()

    candidates: List[Path] = []
    if filepath and filename:
        candidates.append(input_dir / filepath / filename)
    if filename:
        candidates.append(input_dir / "train2014" / filename)
        candidates.append(input_dir / "val2014" / filename)
        candidates.append(input_dir / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not locate image file for {filename}")


def _target_split(source_split: str, train_source_splits: Set[str], val_source_splits: Set[str]) -> Optional[str]:
    if source_split in train_source_splits:
        return "train"
    if source_split in val_source_splits:
        return "val"
    return None


def _copy_image(source_path: Path, destination_path: Path, overwrite: bool) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() and not overwrite:
        return
    shutil.copy2(source_path, destination_path)


def build_split_manifests(
    dataset_json_path: Path,
    input_dir: Path,
    output_dir: Path,
    tokens: Sequence[str],
    train_source_splits: Set[str],
    val_source_splits: Set[str],
    overwrite: bool,
) -> None:
    with open(dataset_json_path, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    tokens = list(tokens)
    manifest_entries: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        "train": {token: [] for token in tokens},
        "val": {token: [] for token in tokens},
    }

    for image_entry in dataset.get("images", []):
        source_split = str(image_entry.get("split", "")).strip()
        target_split = _target_split(source_split, train_source_splits, val_source_splits)
        if target_split is None:
            continue

        sentences = image_entry.get("sentences", [])
        if not isinstance(sentences, list) or not sentences:
            continue

        matched_raw_captions: Dict[str, List[str]] = {token: [] for token in tokens}
        for sentence in sentences:
            sentence_tokens = sentence.get("tokens", []) if isinstance(sentence, dict) else []
            raw_caption = str(sentence.get("raw", "")).strip() if isinstance(sentence, dict) else ""
            if not sentence_tokens or not raw_caption:
                continue
            for token in tokens:
                if _sentence_matches_token(sentence_tokens, token):
                    matched_raw_captions[token].append(raw_caption)

        matched_tokens = [token for token in tokens if matched_raw_captions[token]]
        if not matched_tokens:
            continue

        source_path = _resolve_image_path(input_dir, image_entry)
        filename = str(image_entry.get("filename", source_path.name))

        for token in matched_tokens:
            destination_path = output_dir / target_split / token / filename
            _copy_image(source_path, destination_path, overwrite=overwrite)

            manifest_entries[target_split][token].append(
                {
                    "image_name": filename,
                    "image_path": str(destination_path),
                    "raw_caption": matched_raw_captions[token][0],
                    "raw_captions": matched_raw_captions[token],
                    "source_split": source_split,
                    "cocoid": image_entry.get("cocoid"),
                }
            )

    for split_name in ("train", "val"):
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = split_dir / f"{split_name}.json"
        payload = {
            "split": split_name,
            "tokens": tokens,
            "entries": manifest_entries[split_name],
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a token-filtered COCO dataset layout.")
    parser.add_argument("--input_dir", required=True, help="COCO root directory containing dataset_coco.json and images.")
    parser.add_argument("--output_dir", required=True, help="Where the filtered train/val directories will be written.")
    parser.add_argument("--tokens", nargs="+", required=True, help="Tokens or phrases to filter on, for example dog cat bus train.")
    parser.add_argument("--dataset_json", default="", help="Optional path to dataset_coco.json. Defaults to <input_dir>/dataset_coco.json.")
    parser.add_argument(
        "--train_source_splits",
        nargs="*",
        default=["train", "restval"],
        help="Source split names that should map to output/train.",
    )
    parser.add_argument(
        "--val_source_splits",
        nargs="*",
        default=["val", "test"],
        help="Source split names that should map to output/val.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing copied images.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_json_path = Path(args.dataset_json).expanduser().resolve() if args.dataset_json else input_dir / "dataset_coco.json"
    dataset_json_path = _ensure_dataset_json(input_dir, dataset_json_path)

    tokens = _split_token_args(args.tokens)
    if not tokens:
        raise ValueError("No valid tokens were provided.")

    build_split_manifests(
        dataset_json_path=dataset_json_path,
        input_dir=input_dir,
        output_dir=output_dir,
        tokens=tokens,
        train_source_splits={_normalize_token(split) for split in args.train_source_splits},
        val_source_splits={_normalize_token(split) for split in args.val_source_splits},
        overwrite=bool(args.overwrite),
    )

    print(f"Done. Wrote filtered COCO data to {output_dir}")


if __name__ == "__main__":
    main()