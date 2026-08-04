#!/usr/bin/env python3
"""
Re-run the coco10 ablation's explainer/eval/plots (steps 6-8) against
single-object crops (data/coco10/val_masked) instead of 2x2 grids
(data/coco10/val_grids), reusing each config's already-built concept bank
from outputs/ablation_coco10/<config>/ (steps 1-5 are NOT re-run -- the
upstream directories are symlinked in so run_full_pipeline.py's own
skip-checks find them and only steps 6-8 execute fresh).

Usage:
    python scripts/run_ablation_singleobj.py --dry-run
    python scripts/run_ablation_singleobj.py --devices cuda:0,cuda:1,cuda:2
"""

import argparse
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(Path(__file__).parent))
# Single source of truth for the config matrix + naming convention, shared
# with the script that runs the grid ablation this reuses concept banks from.
from run_ablation import (  # noqa: E402
    PROMPT_TEMPLATES,
    CROP_MODES,
    DECOMP_STRATEGIES,
    SEEDS,
    default_decomp_strategy,
    config_name,
)

SOURCE_ROOT = ROOT_DIR / "outputs" / "ablation_coco10"
DEST_ROOT = ROOT_DIR / "outputs" / "ablation_coco10_singleobj"
REUSED_SUBDIRS = ["inference", "logitlens", "features", "concept"]

CHOICES = "apple,banana,bird,cake,cat,cup,dog,donut,knife,orange"


def prepare_dest(cfg: str, logger=print) -> bool:
    """Symlink the reused steps 1-5 subdirectories from the grid ablation's
    output into the single-object ablation's output dir. Returns False if the
    source config isn't complete (missing concept bank), so the caller can
    skip it."""
    src_dir = SOURCE_ROOT / cfg
    dest_dir = DEST_ROOT / cfg
    concept_dir = src_dir / "concept"
    if not concept_dir.exists() or not any(concept_dir.glob("**/*_raw.pth")):
        logger(f"  [skip] {cfg}: no concept bank found under {concept_dir}")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    for sub in REUSED_SUBDIRS:
        src_sub = src_dir / sub
        dest_sub = dest_dir / sub
        if dest_sub.is_symlink() or dest_sub.exists():
            continue
        if src_sub.exists():
            dest_sub.symlink_to(src_sub, target_is_directory=True)
    return True


def run_one(prompt_template: str, crop_mode: str, decomp_strategy: str, seed: int, env_overrides: dict,
            dry_run: bool = False, device: str = None, print_lock=None) -> bool:
    cfg = config_name(prompt_template, crop_mode, decomp_strategy, seed)
    output_dir = DEST_ROOT / cfg

    def _log(msg: str) -> None:
        if print_lock is not None:
            with print_lock:
                print(msg)
        else:
            print(msg)

    _log(f"=== prompt={prompt_template} crop={crop_mode} decomp={decomp_strategy} seed={seed} device={device or 'env-default'} -> {output_dir}")

    if dry_run:
        return True

    if not prepare_dest(cfg, logger=_log):
        return False

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
        env["MAX_NEW_TOKENS"] = "1"

    cmd = [
        sys.executable, "-u", str(ROOT_DIR / "scripts" / "run_full_pipeline.py"),
        "--output-dir", str(output_dir),
        "--decomp", "snmf",
    ]

    log_path = output_dir / "ablation_singleobj_run.log"
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, env=env, cwd=str(ROOT_DIR), stdout=logf, stderr=subprocess.STDOUT)

    ok = proc.returncode == 0
    _log(f"    -> {'OK' if ok else f'FAILED (exit {proc.returncode})'} (log: {log_path}) [device={device or 'env-default'}]")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expl-max-images", default="-1", help="EXPL_MAX_IMAGES for the explainer/eval steps on val_masked (-1 = all available).")
    parser.add_argument(
        "--devices", default=None,
        help="Comma-separated CUDA devices to parallelize across (e.g. 'cuda:0,cuda:1,cuda:2').",
    )
    parser.add_argument(
        "--decomp-strategies",
        choices=["default", "all"],
        default="default",
        help="Must match the --decomp-strategies value the grid ablation (run_ablation.py) "
             "was run with -- otherwise the expected source config directories won't exist "
             "to symlink from. 'default' = 18 configs, 'all' = 36.",
    )
    args = parser.parse_args()

    env_overrides = {
        "DECOMP_METHODS": "snmf",
        "IMAGE_BUDGET": "-1",
        "EXPL_MAX_IMAGES": str(args.expl_max_images),
        "CLEAN_EXAMPLE_RATIO": "0.2",
        "INPUT_DIR": "data/coco10/train_all",
        "IMAGE_ROOT": "data/coco10/val_masked",
        "CONCEPTS_VOCAB": "src/assets/coco10_vocab.txt",
        "NUM_CONCEPT": "-1",
        "EXPL_PROMPT_MODE": "mcq",
        "EXPL_CHOICES": CHOICES,
        "SINGLE_OBJECT": "1",
    }
    print(f"Dest root: {DEST_ROOT}")
    print(f"Settings: {env_overrides}")

    if args.decomp_strategies == "all":
        strategies_for = lambda pt: DECOMP_STRATEGIES  # noqa: E731
    else:
        strategies_for = lambda pt: [default_decomp_strategy(pt)]  # noqa: E731

    all_configs = [
        (pt, cm, ds, seed)
        for pt in PROMPT_TEMPLATES
        for cm in CROP_MODES
        for ds in strategies_for(pt)
        for seed in SEEDS
    ]
    print(f"Total configs: {len(all_configs)} (decomp-strategies={args.decomp_strategies})")

    devices = [d.strip() for d in args.devices.split(",") if d.strip()] if args.devices else None
    results = []

    if devices and len(devices) > 1 and not args.dry_run:
        print(f"Parallelizing across devices: {devices}")
        device_queue: "queue.Queue[str]" = queue.Queue()
        for d in devices:
            device_queue.put(d)
        print_lock = threading.Lock()

        def _worker(cfg):
            pt, cm, ds, seed = cfg
            device = device_queue.get()
            try:
                ok = run_one(pt, cm, ds, seed, env_overrides, device=device, print_lock=print_lock)
            finally:
                device_queue.put(device)
            return (pt, cm, ds, seed, ok)

        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            for res in executor.map(_worker, all_configs):
                results.append(res)
    else:
        device = devices[0] if devices else None
        for pt, cm, ds, seed in all_configs:
            ok = run_one(pt, cm, ds, seed, env_overrides, dry_run=args.dry_run, device=device)
            results.append((pt, cm, ds, seed, ok))

    print("\n=== Single-object ablation summary ===")
    for pt, cm, ds, seed, ok in results:
        print(f"{pt:15s} {cm:15s} {ds:8s} seed={seed} {'OK' if ok else 'FAILED'}")

    n_failed = sum(1 for *_, ok in results if not ok)
    if n_failed:
        print(f"\n{n_failed} of {len(results)} runs failed/skipped — check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
