#!/usr/bin/env python3
"""
TGCL rebuttal ablation, 10-class variant: same tier structure as
scripts/run_rebuttal_ablation.py (the 3-class apple/cat/bird run) but with
Tier 0's seed count reduced 5->3 (33 configs total: 18+3+6+6), plus:
  - all 10 coco10 categories (apple, banana, bird, cake, cat, cup, dog,
    donut, knife, orange), not just 3
  - 300 train images/category (data/coco10/train_all rebuilt with
    --train-cap 300, down from the 3-class run's 1000/category)
  - BAG_SIZE=900 flat cap (not the 3-class run's 2x-train-images formula)
  - LOGIT_LENS_LAYER_SELECTION=0 (disabled per explicit request -- every
    tag uses the same static LAYER_PATH default, model.language_model.norm,
    instead of a per-tag logit-lens-selected layer)
  - a separate fixed eval set (data/coco10/val_masked_rebuttal10, all 10
    classes, capped at 50 images/category = 500 total) and a separate
    output root (outputs/rebuttal_ablation_10class) so this run doesn't
    touch the completed 3-class results

Reuses the shared-inference-cache helpers from run_rebuttal_ablation.py
(those don't depend on the class list) but defines its own CONDITIONS/
run_one/build_grid since CONDITIONS embeds the class list at construction
time.

Usage:
    python scripts/run_rebuttal_ablation_10class.py --dry-run
    python scripts/run_rebuttal_ablation_10class.py --smoke-test
    python scripts/run_rebuttal_ablation_10class.py --devices cuda:0,cuda:1,cuda:2,cuda:3
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(Path(__file__).parent))
from run_rebuttal_ablation import (  # noqa: E402
    prepare_shared_inference,
    populate_shared_inference_cache,
)

# crops.json (step 2) doesn't depend on PROMPT_TEMPLATE/condition at all --
# only on crop_mode and the underlying image/concept-map pool, which are
# identical across every condition here. Confirmed empirically: langsam's
# crops.json was byte-identical across seed1/seed3 and differed from seed2
# only by a few hundred bytes (segmentation-model noise, not a real seed
# effect) -- so one shared crops.json per crop_mode is reused across every
# condition AND every seed, instead of each of the 15 langsam configs
# independently re-running ~55min of LangSAM segmentation.
CROPS_CACHE_FILES = ["crops.json", "crop_status.log"]


def shared_crops_dir(ablation_root: Path, crop_mode: str) -> Path:
    return ablation_root / f"_shared_crops_{crop_mode}"


def prepare_shared_crops(ablation_root: Path, output_dir: Path, crop_mode: str) -> None:
    cache_dir = shared_crops_dir(ablation_root, crop_mode)
    crops_cache = cache_dir / "crops.json"
    if not crops_cache.exists():
        return
    dest = output_dir / "inference"
    dest.mkdir(parents=True, exist_ok=True)
    for fname in CROPS_CACHE_FILES:
        src = cache_dir / fname
        if src.exists():
            import shutil
            shutil.copy2(src, dest / fname)


def populate_shared_crops_cache(ablation_root: Path, output_dir: Path, crop_mode: str) -> None:
    cache_dir = shared_crops_dir(ablation_root, crop_mode)
    if (cache_dir / "crops.json").exists():
        return
    src = output_dir / "inference"
    if not (src / "crops.json").exists():
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for fname in CROPS_CACHE_FILES:
        if (src / fname).exists():
            shutil.copy2(src / fname, cache_dir / fname)

CLASSES = ["apple", "banana", "bird", "cake", "cat", "cup", "dog", "donut", "knife", "orange"]

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
        # Shuffled control: the prompt asks about a WRONG concept (one of the
        # other 9 classes), so the model answers "No" for ~all samples and no
        # direction passes the default CLEAN_EXAMPLE_RATIO=0.4 purity filter
        # -> empty bank -> pipeline aborts. Per decision, keep 0 filtering for
        # this condition ONLY (0.0 = keep every direction) so a bank forms and
        # its faithfulness AUC is comparable to P_bin's, rather than
        # collapsing. This per-condition entry overrides the global
        # env_overrides CLEAN_EXAMPLE_RATIO=0.4 (env.update(cond_cfg) runs
        # after env.update(env_overrides) in run_one).
        CLEAN_EXAMPLE_RATIO="0.0",
    ),
    "P_bin_fullpool": dict(
        PROMPT_TEMPLATE="cgdl",
        HOOK_NAMES="save_hidden_states_mean",
    ),
}

TIER0_CONDITIONS = ["P_null", "P_open", "P_bin"]
TIER0_CROP_MODES = ["sliding_window", "langsam"]
TIER0_SEEDS = [1, 2, 3]

TIER1_CONDITIONS = ["P_null", "P_open", "P_bin"]
TIER1_CROP_MODE = "none"
TIER1_SEED = 1

TIER23_CROP_MODES = ["sliding_window", "langsam"]
TIER23_SEEDS = [1, 2, 3]

ABLATION_ROOT_NAME = "outputs/rebuttal_ablation_10class"


def build_grid():
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

    # run_full_pipeline.py's own per-step skip-checks cover every step except
    # the BERT/CLIP top-k table (no existence check there at all -- it always
    # recomputes), so re-launching an already-finished config wastes real
    # time re-scoring instead of exiting immediately. Confirmed empirically:
    # 4 already-complete configs each sat in that recompute for 10+ minutes,
    # occupying all 4 GPU workers with zero-benefit work. Skip at the
    # orchestrator level instead of relying on the subprocess to notice.
    #
    # Use the step-8 plot file as the completion marker, not the
    # "Pipeline completed" log line: ablation_run.log gets truncated the
    # moment a relaunch starts (open(..., "w")), so if that relaunch is then
    # killed mid-recompute (as happened here), the marker line is gone even
    # though the underlying data (concept bank, eval CSVs) is still valid --
    # confirmed this happened to 8 configs above. The plot file only exists
    # once every step has actually finished, and step 8 itself never
    # overwrites it once present.
    done_marker = output_dir / "eval" / "snmf" / "c_insertion_token_all_ranks.png"
    if done_marker.exists():
        _log(f"    -> ALREADY COMPLETE, skipping (found {done_marker})")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_shared_inference(ablation_root, output_dir, env_overrides["IMAGE_BUDGET"])
    prepare_shared_crops(ablation_root, output_dir, crop_mode)

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
        populate_shared_crops_cache(ablation_root, output_dir, crop_mode)
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
    parser.add_argument("--expl-max-images", default="-1", help="EXPL_MAX_IMAGES for the explainer/eval steps. -1 = all fixed eval images.")
    parser.add_argument(
        "--devices", default=None,
        help="Comma-separated CUDA devices to parallelize across (e.g. 'cuda:0,cuda:1,cuda:2,cuda:3').",
    )
    args = parser.parse_args()

    ablation_root = ROOT_DIR / ABLATION_ROOT_NAME
    image_root = ROOT_DIR / "data" / "coco10" / "val_masked_rebuttal10"
    if not image_root.exists():
        raise RuntimeError(
            f"{image_root} does not exist -- run "
            "preprocessing/build_rebuttal_eval_set.py --classes "
            f"{','.join(CLASSES)} --dest-dir {image_root} "
            f"--manifest data/coco10/eval_images_rebuttal10.json first."
        )
    manifest_path = ROOT_DIR / "data" / "coco10" / "eval_images_rebuttal10.json"
    n_manifest = 0
    if manifest_path.exists():
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
        "CLEAN_EXAMPLE_RATIO": "0.4",
        "BAG_SIZE": "900",
        # Disabled per explicit request: every tag uses the same static
        # LAYER_PATH (unset here -> code default model.language_model.norm)
        # instead of a per-tag logit-lens-selected layer.
        "LOGIT_LENS_LAYER_SELECTION": "0",
        "IMAGE_BUDGET": str(args.image_budget),
        "EXPL_MAX_IMAGES": str(args.expl_max_images),
        "INPUT_DIR": "data/coco10/train_all",
        "IMAGE_ROOT": str(image_root),
        "SINGLE_OBJECT": "1",
        "CONCEPTS_VOCAB": "src/assets/coco10_vocab.txt",
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

    print("\n=== Rebuttal ablation summary (10-class) ===")
    for condition, crop_mode, seed, ok in results:
        print(f"{condition:15s} {crop_mode:15s} seed={seed} {'OK' if ok else 'FAILED'}")

    n_failed = sum(1 for *_, ok in results if not ok)
    if n_failed:
        print(f"\n{n_failed} of {len(results)} runs failed — check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
