#!/usr/bin/env python3
"""
Ablation harness: {cgdl, non_contrastive} x {none, sliding_window, langsam}
crop modes, 2 repeats (seeds) each, SNMF-only.

Defaults are a fast smoke run (100 training images, 5 explainer images,
CLEAN_EXAMPLE_RATIO=0.5) to validate the full matrix runs cleanly before
committing to a full-dataset pass. For the later full run, just override
--image-budget (and optionally --expl-max-images / --clean-example-ratio) --
everything else (crop-fallback fix, gradient-importance fix, relative-AUC
fix, per-rank + mean/std reporting) is already wired in.

Optimization: step 1 (VLM captioning -> concept map) is identical across all
12 runs (it only depends on INPUT_DIR/IMAGE_BUDGET, not CROP_MODE or
PROMPT_TEMPLATE). It's computed once per --image-budget value (cached under
outputs/ablation/_shared_inference_budget<N>/, keyed by budget so changing
--image-budget later doesn't silently reuse a stale, differently-sized
cache) and copied into each run's output dir so step 1's skip-check finds it
and doesn't redo the VLM captioning 12 times.

Usage:
    python scripts/run_ablation.py                      # smoke run (100 images, 5 explainer, cr=0.5)
    python scripts/run_ablation.py --dry-run             # print the plan, don't launch anything
    python scripts/run_ablation.py --image-budget -1 --expl-max-images -1   # full dataset, later
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

PROMPT_TEMPLATES = ["cgdl", "non_contrastive", "null"]
CROP_MODES = ["none", "sliding_window", "langsam"]
# DECOMP_STRATEGY is an independent axis from PROMPT_TEMPLATE (see
# scripts/run_full_pipeline.py's PipelineConfig.decomp_strategy): "per_tag"
# decomposes each tag's features separately then purity-filters + merges;
# "pooled" concatenates every tag's features once and decomposes globally,
# no purity filter. Any prompt_template can use either.
DECOMP_STRATEGIES = ["per_tag", "pooled"]
SEEDS = [42, 43]

# "null" is a literal empty instruction (see src/models/constants.py's
# null_prompt / TASK_PROMPTS "null" entry); it gets MAX_NEW_TOKENS=1 in
# run_one() below, so feature extraction does a single decoder forward pass
# instead of the usual multi-token generation.
NULL_PROMPT_MAX_NEW_TOKENS = "1"


def default_decomp_strategy(prompt_template: str) -> str:
    """Each template's DECOMP_STRATEGY default when not explicitly overridden
    -- matches PipelineConfig's own default in run_full_pipeline.py, so a run
    that doesn't ask for a specific strategy behaves the same as before this
    axis existed."""
    return "per_tag" if prompt_template == "cgdl" else "pooled"

# Only cache the two files that require an actual VLM forward pass to
# produce (objects.csv, concepts_to_images.json). The vocab-filtered file
# (concepts_to_images_<vocab>_top<N>.json) is cheap CPU-only filtering and
# its exact name varies by dataset/vocab/discovered-concept-count, so it's
# just left to recompute fresh each time rather than caching a fragile name.
SHARED_INFERENCE_FILES = ["objects.csv", "concepts_to_images.json"]

DATASET_PRESETS = {
    "imagenet10": {
        "input_dir": "data/imagenet10/train_all",
        "image_root": "data/imagenet10/val_grids",
        "concepts_vocab": "src/assets/vocab.txt",
        "ablation_root": "outputs/ablation",
    },
    "coco10": {
        "input_dir": "data/coco10/train_all",
        "image_root": "data/coco10/val_grids",
        "concepts_vocab": "src/assets/coco10_vocab.txt",
        "ablation_root": "outputs/ablation_coco10",
        # Grid validation images are busy/cluttered real-scene crops; closed-set
        # (multiple-choice) selection from the known 10 category names is more
        # reliable than open-ended naming for this dataset.
        "expl_prompt_mode": "mcq",
        "expl_choices": "apple,banana,bird,cake,cat,cup,dog,donut,knife,orange",
    },
}


def config_name(prompt_template: str, crop_mode: str, decomp_strategy: str, seed: int) -> str:
    # Only append the strategy when it's NOT this template's default, so the
    # 16 pre-existing directories (built before DECOMP_STRATEGY existed) keep
    # their exact names and are still recognized as already-complete.
    suffix = "" if decomp_strategy == default_decomp_strategy(prompt_template) else f"_{decomp_strategy}"
    return f"{crop_mode}_{prompt_template}{suffix}_seed{seed}"


def seed_output_dir(ablation_root: Path, prompt_template: str, crop_mode: str, decomp_strategy: str, seed: int) -> Path:
    return ablation_root / config_name(prompt_template, crop_mode, decomp_strategy, seed)


def shared_inference_dir(ablation_root: Path, image_budget: str) -> Path:
    # Keyed by image_budget so a later run with a different budget (e.g. the
    # full-dataset pass) doesn't silently reuse a smaller/stale cache.
    tag = str(image_budget).replace("-", "neg")
    return ablation_root / f"_shared_inference_budget{tag}"


def prepare_shared_inference(ablation_root: Path, output_dir: Path, image_budget: str, logger=print) -> None:
    """Copy the cached VLM captioning output (for this image_budget) into
    output_dir/inference so run_full_pipeline.py's step 1 skip-check finds
    it and reuses it instead of redoing VLM captioning."""
    cache_dir = shared_inference_dir(ablation_root, image_budget)
    if not cache_dir.exists() or not any((cache_dir / f).exists() for f in SHARED_INFERENCE_FILES):
        return  # nothing cached yet -- step 1 will run fresh and populate the cache afterward
    dest = output_dir / "inference"
    dest.mkdir(parents=True, exist_ok=True)
    for fname in SHARED_INFERENCE_FILES:
        src = cache_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)


def populate_shared_inference_cache(ablation_root: Path, output_dir: Path, image_budget: str) -> None:
    """After a run computes step 1 fresh, seed the shared cache from its
    output so subsequent configs (same image_budget) can reuse it."""
    cache_dir = shared_inference_dir(ablation_root, image_budget)
    if cache_dir.exists() and any((cache_dir / f).exists() for f in SHARED_INFERENCE_FILES):
        return  # already cached by an earlier config in this run
    src = output_dir / "inference"
    if not all((src / f).exists() for f in SHARED_INFERENCE_FILES):
        return  # this run didn't produce a complete inference set (e.g. failed early)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for fname in SHARED_INFERENCE_FILES:
        shutil.copy2(src / fname, cache_dir / fname)


def run_one(
    ablation_root: Path,
    prompt_template: str,
    crop_mode: str,
    decomp_strategy: str,
    seed: int,
    env_overrides: dict,
    dry_run: bool = False,
    device: str = None,
    print_lock=None,
) -> bool:
    output_dir = seed_output_dir(ablation_root, prompt_template, crop_mode, decomp_strategy, seed)

    def _log(msg: str) -> None:
        if print_lock is not None:
            with print_lock:
                print(msg)
        else:
            print(msg)

    _log(f"=== prompt={prompt_template} crop={crop_mode} decomp={decomp_strategy} seed={seed} device={device or 'env-default'} -> {output_dir}")

    if dry_run:
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_shared_inference(ablation_root, output_dir, env_overrides["IMAGE_BUDGET"])

    env = os.environ.copy()
    env.update(env_overrides)
    env["PROMPT_TEMPLATE"] = prompt_template
    env["CROP_MODE"] = crop_mode
    env["DECOMP_STRATEGY"] = decomp_strategy
    env["SEED"] = str(seed)
    env["OUTPUT_DIR"] = str(output_dir)
    if device:
        env["DEVICE"] = device
    if prompt_template == "null":
        env["MAX_NEW_TOKENS"] = NULL_PROMPT_MAX_NEW_TOKENS

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
    parser.add_argument("--dataset", choices=list(DATASET_PRESETS.keys()), default="imagenet10")
    parser.add_argument("--dry-run", action="store_true", help="Print the run plan without launching anything")
    parser.add_argument("--image-budget", default="100", help="IMAGE_BUDGET for step 1 (default: 100, smoke run). -1 = all images.")
    parser.add_argument("--expl-max-images", default="5", help="EXPL_MAX_IMAGES for the explainer/eval steps (default: 5, smoke run). -1 = all val_grids images.")
    parser.add_argument("--clean-example-ratio", default="0.5", help="CLEAN_EXAMPLE_RATIO purity threshold (default: 0.5).")
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Comma-separated CUDA devices to parallelize the configs across "
            "(e.g. 'cuda:0,cuda:1,cuda:2'). Each device runs one config's full "
            "pipeline subprocess at a time; configs are pulled off a shared "
            "queue as devices free up. Default: sequential single-device run "
            "using the DEVICE env var (unchanged legacy behavior)."
        ),
    )
    parser.add_argument(
        "--decomp-strategies",
        choices=["default", "all"],
        default="default",
        help=(
            "'default' (16 configs, unchanged from before DECOMP_STRATEGY "
            "existed): each prompt_template x crop_mode x seed combo runs "
            "once, at that template's default strategy (cgdl -> per_tag, "
            "non_contrastive/null -> pooled). "
            "'all' (36 configs): every prompt_template x crop_mode x seed "
            "combo runs at BOTH per_tag and pooled, including the "
            "non-default combinations (cgdl+pooled, non_contrastive+per_tag, "
            "null+per_tag) and null+none (previously only run with "
            "langsam/sliding_window crops)."
        ),
    )
    args = parser.parse_args()

    preset = DATASET_PRESETS[args.dataset]
    ablation_root = ROOT_DIR / preset["ablation_root"]

    env_overrides = {
        "DECOMP_METHODS": "snmf",
        "IMAGE_BUDGET": str(args.image_budget),
        "EXPL_MAX_IMAGES": str(args.expl_max_images),
        "CLEAN_EXAMPLE_RATIO": str(args.clean_example_ratio),
        "INPUT_DIR": preset["input_dir"],
        "IMAGE_ROOT": preset["image_root"],
        "CONCEPTS_VOCAB": preset["concepts_vocab"],
        "NUM_CONCEPT": "-1",
    }
    if "expl_prompt_mode" in preset:
        env_overrides["EXPL_PROMPT_MODE"] = preset["expl_prompt_mode"]
    if "expl_choices" in preset:
        env_overrides["EXPL_CHOICES"] = preset["expl_choices"]
    print(f"Dataset: {args.dataset}  ablation_root: {ablation_root}")
    print(f"Ablation settings: {env_overrides}")

    devices = [d.strip() for d in args.devices.split(",") if d.strip()] if args.devices else None

    if args.decomp_strategies == "all":
        strategies_for = lambda pt: DECOMP_STRATEGIES  # noqa: E731 -- every template gets both strategies
    else:
        strategies_for = lambda pt: [default_decomp_strategy(pt)]  # noqa: E731 -- unchanged 16-config behavior

    all_configs = [
        (prompt_template, crop_mode, decomp_strategy, seed)
        for prompt_template in PROMPT_TEMPLATES
        for crop_mode in CROP_MODES
        for decomp_strategy in strategies_for(prompt_template)
        for seed in SEEDS
    ]
    print(f"Total configs: {len(all_configs)} (decomp-strategies={args.decomp_strategies})")

    results = []
    if devices and len(devices) > 1 and not args.dry_run:
        print(f"Parallelizing across devices: {devices}")
        # Run the first config alone, on the first device, to populate the
        # shared step-1 (VLM captioning) cache before fanning out -- avoids
        # every worker racing to compute step 1 fresh the moment they all
        # find the cache empty simultaneously.
        first_pt, first_cm, first_ds, first_seed = all_configs[0]
        ok = run_one(ablation_root, first_pt, first_cm, first_ds, first_seed, env_overrides, device=devices[0])
        results.append((first_pt, first_cm, first_ds, first_seed, ok))

        remaining = all_configs[1:]
        device_queue: "queue.Queue[str]" = queue.Queue()
        for d in devices:
            device_queue.put(d)
        print_lock = threading.Lock()

        def _worker(cfg):
            pt, cm, ds, seed = cfg
            device = device_queue.get()
            try:
                ok = run_one(
                    ablation_root, pt, cm, ds, seed, env_overrides,
                    device=device, print_lock=print_lock,
                )
            finally:
                device_queue.put(device)
            return (pt, cm, ds, seed, ok)

        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            for res in executor.map(_worker, remaining):
                results.append(res)
    elif devices and len(devices) > 1 and args.dry_run:
        # Dry-run with multiple devices: just display the round-robin
        # assignment each parallel worker would actually get; don't launch
        # anything (mirrors the real dispatch order: config 0 pinned to
        # devices[0], the rest pulled off the shared device queue).
        for idx, (prompt_template, crop_mode, decomp_strategy, seed) in enumerate(all_configs):
            device = devices[idx % len(devices)]
            ok = run_one(ablation_root, prompt_template, crop_mode, decomp_strategy, seed, env_overrides, dry_run=True, device=device)
            results.append((prompt_template, crop_mode, decomp_strategy, seed, ok))
    else:
        device = devices[0] if devices else None
        for prompt_template, crop_mode, decomp_strategy, seed in all_configs:
            ok = run_one(ablation_root, prompt_template, crop_mode, decomp_strategy, seed, env_overrides, dry_run=args.dry_run, device=device)
            results.append((prompt_template, crop_mode, decomp_strategy, seed, ok))

    print("\n=== Ablation summary ===")
    for prompt_template, crop_mode, decomp_strategy, seed, ok in results:
        print(f"{prompt_template:15s} {crop_mode:15s} {decomp_strategy:8s} seed={seed} {'OK' if ok else 'FAILED'}")

    n_failed = sum(1 for *_, ok in results if not ok)
    if n_failed:
        print(f"\n{n_failed} of {len(results)} runs failed — check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
