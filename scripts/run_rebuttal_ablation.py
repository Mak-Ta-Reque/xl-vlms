#!/usr/bin/env python3
"""
TGCL rebuttal ablation: 3-class (apple/cat/bird) fast protocol per the user's
spec ("TGCL Rebuttal Ablation -- 3-Class Fast Protocol", 5 seeds reduced from
the original 5-class version). See docs/coco10_ablation_methods.md for the
existing {cgdl,non_contrastive,null} x {none,sliding_window,langsam} x
{per_tag,pooled} ablation this is a SEPARATE, standalone driver from -- this
one uses a different condition axis (5 named conditions, not
template x strategy) and a fixed evaluation image set, so it does not extend
scripts/run_ablation.py (whose config-naming code is specific to that axis).

Grid (45 runs):
  Tier 0: P_null, P_open, P_bin        x {sliding_window, langsam} x seeds 1-5  = 30
  Tier 1: P_null, P_open, P_bin        x {none}                    x seed  1   =  3
  Tier 2: P_bin_shuf                   x {sliding_window, langsam} x seeds 1-3 =  6
  Tier 3: P_bin_fullpool               x {sliding_window, langsam} x seeds 1-3 =  6

Non-negotiable invariants forced for every run (see docs/coco10_ablation_methods.md
and the approved plan): DECOMP_STRATEGY=per_tag (P_null/P_open would otherwise
default to pooled), DECOMP_COMPONENTS=2, DL_ALPHA=20 (not .env's 23),
CLEAN_EXAMPLE_RATIO=0.8 (not .env's 0.5), BAG_SIZE=2000 (2x the 1000 train
images/category, not .env's flat 400), fixed IMAGE_ROOT=val_masked_rebuttal
(single-object crops only -- grid quadrants aren't distinguishable in the
explainer's output JSON, so per-image pairing needs one real image = one
file), CONCEPTS_VOCAB restricted to apple/cat/bird.

LOGIT_LENS_LAYER_SELECTION is deliberately left at its .env default (1,
per-tag layer selection via logit lens) -- that's the same mechanism already
used for every other ablation this session ("same as main results"); it's an
identical algorithm applied uniformly across all 5 conditions, not a factor
that varies between them.

Usage:
    python scripts/run_rebuttal_ablation.py --dry-run
    python scripts/run_rebuttal_ablation.py --smoke-test        # P_bin + P_null x langsam seed 1, EXPL_MAX_IMAGES=5
    python scripts/run_rebuttal_ablation.py --devices cuda:0,cuda:1,cuda:2,cuda:3
"""

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()

CLASSES = ["apple", "cat", "bird"]

CONDITIONS = {
    "P_null": dict(
        PROMPT_TEMPLATE="null",
        HOOK_NAMES="save_hidden_states_mean",
        MAX_NEW_TOKENS="1",
    ),
    "P_open": dict(
        PROMPT_TEMPLATE="non_contrastive",
        HOOK_NAMES="save_hidden_states_mean",
    ),
    "P_bin": dict(
        PROMPT_TEMPLATE="cgdl",
        HOOK_NAMES="save_hidden_states_for_token_of_interest",
    ),
    "P_bin_shuf": dict(
        PROMPT_TEMPLATE="cgdl",
        HOOK_NAMES="save_hidden_states_for_token_of_interest",
        SHUFFLE_CONCEPT_PROMPT="1",
        SHUFFLE_CONCEPT_VOCAB=",".join(CLASSES),
    ),
    "P_bin_fullpool": dict(
        PROMPT_TEMPLATE="cgdl",
        HOOK_NAMES="save_hidden_states_mean",
    ),
}

TIER0_CONDITIONS = ["P_null", "P_open", "P_bin"]
TIER0_CROP_MODES = ["sliding_window", "langsam"]
TIER0_SEEDS = [1, 2, 3, 4, 5]

TIER1_CONDITIONS = ["P_null", "P_open", "P_bin"]
TIER1_CROP_MODE = "none"
TIER1_SEED = 1

TIER23_CROP_MODES = ["sliding_window", "langsam"]
TIER23_SEEDS = [1, 2, 3]

# Files that require an actual VLM forward pass over train_all to produce;
# identical across all 47 runs since INPUT_DIR/IMAGE_BUDGET/CONCEPTS_VOCAB
# never change -- cached once and reused, same trick as run_ablation.py.
SHARED_INFERENCE_FILES = ["objects.csv", "concepts_to_images.json"]

ABLATION_ROOT_NAME = "outputs/rebuttal_ablation"


def build_grid():
    """Return the 45-run (condition, crop_mode, seed) tier table."""
    configs = []
    for condition in TIER0_CONDITIONS:
        for crop_mode in TIER0_CROP_MODES:
            for seed in TIER0_SEEDS:
                configs.append((condition, crop_mode, seed))
    for condition in TIER1_CONDITIONS:
        configs.append((condition, TIER1_CROP_MODE, TIER1_SEED))
    for crop_mode in TIER23_CROP_MODES:
        for seed in TIER23_SEEDS:
            configs.append(("P_bin_shuf", crop_mode, seed))
    for crop_mode in TIER23_CROP_MODES:
        for seed in TIER23_SEEDS:
            configs.append(("P_bin_fullpool", crop_mode, seed))
    return configs


def config_name(condition: str, crop_mode: str, seed: int) -> str:
    return f"{condition}_{crop_mode}_seed{seed}"


def shared_inference_dir(ablation_root: Path, image_budget: str) -> Path:
    tag = str(image_budget).replace("-", "neg")
    return ablation_root / f"_shared_inference_budget{tag}"


def prepare_shared_inference(ablation_root: Path, output_dir: Path, image_budget: str) -> None:
    cache_dir = shared_inference_dir(ablation_root, image_budget)
    if not cache_dir.exists() or not any((cache_dir / f).exists() for f in SHARED_INFERENCE_FILES):
        return
    dest = output_dir / "inference"
    dest.mkdir(parents=True, exist_ok=True)
    for fname in SHARED_INFERENCE_FILES:
        src = cache_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)


def populate_shared_inference_cache(ablation_root: Path, output_dir: Path, image_budget: str) -> None:
    cache_dir = shared_inference_dir(ablation_root, image_budget)
    if cache_dir.exists() and any((cache_dir / f).exists() for f in SHARED_INFERENCE_FILES):
        return
    src = output_dir / "inference"
    if not all((src / f).exists() for f in SHARED_INFERENCE_FILES):
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    for fname in SHARED_INFERENCE_FILES:
        shutil.copy2(src / fname, cache_dir / fname)


def run_one(
    ablation_root: Path,
    condition: str,
    crop_mode: str,
    seed: int,
    env_overrides: dict,
    dry_run: bool = False,
    device: str = None,
    print_lock=None,
) -> bool:
    output_dir = ablation_root / config_name(condition, crop_mode, seed)

    def _log(msg: str) -> None:
        if print_lock is not None:
            with print_lock:
                print(msg)
        else:
            print(msg)

    cond_cfg = CONDITIONS[condition]
    _log(f"=== condition={condition} crop={crop_mode} seed={seed} device={device or 'env-default'} -> {output_dir}")

    if dry_run:
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_shared_inference(ablation_root, output_dir, env_overrides["IMAGE_BUDGET"])

    env = os.environ.copy()
    env.update(env_overrides)
    env.update(cond_cfg)
    env["CROP_MODE"] = crop_mode
    env["SEED"] = str(seed)
    env["OUTPUT_DIR"] = str(output_dir)
    if device:
        env["DEVICE"] = device

    cmd = [
        sys.executable, "-u", str(ROOT_DIR / "scripts" / "run_full_pipeline.py"),
        "--output-dir", str(output_dir),
        "--decomp", "snmf",
    ]

    log_path = output_dir / "ablation_run.log"
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, env=env, cwd=str(ROOT_DIR), stdout=logf, stderr=subprocess.STDOUT)

    ok = proc.returncode == 0
    if ok:
        populate_shared_inference_cache(ablation_root, output_dir, env_overrides["IMAGE_BUDGET"])
    _log(f"    -> {'OK' if ok else f'FAILED (exit {proc.returncode})'} (log: {log_path}) [device={device or 'env-default'}]")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run only P_bin and P_null at langsam/seed1, EXPL_MAX_IMAGES=5, "
        "before committing to the full 45-run grid.",
    )
    parser.add_argument("--image-budget", default="-1", help="IMAGE_BUDGET for step 1 (train_all captioning). -1 = all.")
    parser.add_argument("--expl-max-images", default="-1", help="EXPL_MAX_IMAGES for the explainer/eval steps. -1 = all 384 fixed eval images.")
    parser.add_argument(
        "--devices", default=None,
        help="Comma-separated CUDA devices to parallelize across (e.g. 'cuda:0,cuda:1,cuda:2,cuda:3').",
    )
    args = parser.parse_args()

    ablation_root = ROOT_DIR / ABLATION_ROOT_NAME
    image_root = ROOT_DIR / "data" / "coco10" / "val_masked_rebuttal"
    if not image_root.exists():
        raise RuntimeError(
            f"{image_root} does not exist -- run "
            "preprocessing/build_rebuttal_eval_set.py first."
        )
    manifest_path = ROOT_DIR / "data" / "coco10" / "eval_images_rebuttal.json"
    n_manifest = 0
    if manifest_path.exists():
        import json
        with open(manifest_path) as f:
            manifest = json.load(f)
        n_manifest = sum(len(v) for v in manifest.get("images", {}).values())
    n_actual = sum(1 for _ in image_root.rglob("*.jpg"))
    if manifest_path.exists() and n_actual != n_manifest:
        raise RuntimeError(
            f"Fixed eval set drift detected: {image_root} has {n_actual} files "
            f"but the manifest ({manifest_path}) lists {n_manifest}. Re-run "
            "preprocessing/build_rebuttal_eval_set.py before launching -- "
            "every condition must see the byte-identical image set."
        )

    env_overrides = {
        "DECOMP_STRATEGY": "per_tag",
        "DECOMP_METHODS": "snmf",
        "DECOMP_COMPONENTS": "2",
        "DL_ALPHA": "20",
        "CLEAN_EXAMPLE_RATIO": "0.8",
        "BAG_SIZE": "2000",
        "IMAGE_BUDGET": str(args.image_budget),
        "EXPL_MAX_IMAGES": str(args.expl_max_images),
        "INPUT_DIR": "data/coco10/train_all",
        "IMAGE_ROOT": str(image_root),
        "SINGLE_OBJECT": "1",
        "CONCEPTS_VOCAB": "src/assets/coco10_vocab_rebuttal3.txt",
        "NUM_CONCEPT": "-1",
        "EXPL_PROMPT_MODE": "mcq",
        "EXPL_CHOICES": ",".join(CLASSES),
    }
    print(f"Ablation root: {ablation_root}")
    print(f"Fixed IMAGE_ROOT: {image_root} ({n_actual} images)")
    print(f"Forced invariants: {env_overrides}")

    if args.smoke_test:
        all_configs = [("P_bin", "langsam", 1), ("P_null", "langsam", 1)]
        env_overrides["EXPL_MAX_IMAGES"] = "5"
        env_overrides["IMAGE_BUDGET"] = "50"
        print("SMOKE TEST: P_bin + P_null only, langsam, seed 1, EXPL_MAX_IMAGES=5, IMAGE_BUDGET=50")
    else:
        all_configs = build_grid()
    print(f"Total configs: {len(all_configs)}")

    devices = [d.strip() for d in args.devices.split(",") if d.strip()] if args.devices else None
    results = []

    if devices and len(devices) > 1 and not args.dry_run:
        print(f"Parallelizing across devices: {devices}")
        first_condition, first_crop, first_seed = all_configs[0]
        ok = run_one(ablation_root, first_condition, first_crop, first_seed, env_overrides, device=devices[0])
        results.append((first_condition, first_crop, first_seed, ok))

        remaining = all_configs[1:]
        device_queue: "queue.Queue[str]" = queue.Queue()
        for d in devices:
            device_queue.put(d)
        print_lock = threading.Lock()

        def _worker(cfg):
            condition, crop_mode, seed = cfg
            device = device_queue.get()
            try:
                ok = run_one(ablation_root, condition, crop_mode, seed, env_overrides, device=device, print_lock=print_lock)
            finally:
                device_queue.put(device)
            return (condition, crop_mode, seed, ok)

        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            for res in executor.map(_worker, remaining):
                results.append(res)
    else:
        device = devices[0] if devices else None
        for condition, crop_mode, seed in all_configs:
            ok = run_one(ablation_root, condition, crop_mode, seed, env_overrides, dry_run=args.dry_run, device=device)
            results.append((condition, crop_mode, seed, ok))

    print("\n=== Rebuttal ablation summary ===")
    for condition, crop_mode, seed, ok in results:
        print(f"{condition:15s} {crop_mode:15s} seed={seed} {'OK' if ok else 'FAILED'}")

    n_failed = sum(1 for *_, ok in results if not ok)
    if n_failed:
        print(f"\n{n_failed} of {len(results)} runs failed — check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
