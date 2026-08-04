#!/usr/bin/env python3
"""
One-off retry for the 6 P_bin_shuf configs that failed under the pre-fix
token-of-interest hook (it searched every sample for the crop's TRUE tag
word, but P_bin_shuf's shuffled prompt asks about a DIFFERENT concept, so
the model's response -- correctly -- never mentions the true tag, giving 0
matches for every sample and an empty concept bank). Fixed in
src/helpers/utils.py (per-sample token-of-interest override via
kwargs["prompt_concept"]).

Cleans each config's stale concept/ dir (which holds an EMPTY-but-present
combined_concept_snmf_cr0.8_raw.pth from the failed attempt -- run_full_
pipeline.py's step 5 skip-check would otherwise see that file exists and
skip recomputation, silently keeping the empty bank) before re-running.
Steps 1-4 (VLM captioning, crops, features) are untouched/reused.

Usage:
    python scripts/retry_pbin_shuf.py --devices cuda:0,cuda:1,cuda:2,cuda:3
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_rebuttal_ablation import (  # noqa: E402
    ABLATION_ROOT_NAME, CLASSES, TIER23_CROP_MODES, TIER23_SEEDS,
    ROOT_DIR, run_one,
)

CONFIGS = [("P_bin_shuf", cm, seed) for cm in TIER23_CROP_MODES for seed in TIER23_SEEDS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    args = parser.parse_args()

    ablation_root = ROOT_DIR / ABLATION_ROOT_NAME
    image_root = ROOT_DIR / "data" / "coco10" / "val_masked_rebuttal"

    # Some P_bin_shuf configs may have already succeeded on their own: the
    # main grid driver spawns each config as a fresh subprocess that
    # re-imports from disk, so any config dispatched AFTER the token-of-
    # interest fix was saved picked it up automatically, with no retry
    # needed. Only clean up + retry configs that actually failed --
    # deleting and redoing an already-correct run wastes real GPU time for
    # no benefit.
    configs_to_retry = []
    for condition, crop_mode, seed in CONFIGS:
        cfg_dir = ablation_root / f"{condition}_{crop_mode}_seed{seed}"
        log = cfg_dir / "ablation_run.log"
        log_text = log.read_text() if log.exists() else ""
        if "Pipeline completed" in log_text:
            print(f"Skip {cfg_dir.name}: already completed successfully, leaving as-is")
            continue
        if "Pipeline failed" not in log_text:
            print(f"Skip {cfg_dir.name}: not failed (still running or not yet started) -- leaving alone")
            continue
        configs_to_retry.append((condition, crop_mode, seed))
        # features/ MUST be deleted too, not just concept/ -- the actual
        # bug lived inside step 4 (feature generation), where the
        # token-of-interest hook runs. run_full_pipeline.py's step 4
        # skip-check only looks for features/ existing, so leaving it in
        # place (as an earlier version of this script did) silently reuses
        # the pre-fix features and step 5 fails identically every time,
        # confirmed empirically: the first retry attempt did exactly this.
        for stale in ("concept", "features", "explanations", "eval", "logitlens_decompose"):
            p = cfg_dir / stale
            if p.exists():
                print(f"Removing stale {p}")
                shutil.rmtree(p)

    if not configs_to_retry:
        print("Nothing to retry -- all P_bin_shuf configs already succeeded.")
        return

    env_overrides = {
        "DECOMP_STRATEGY": "per_tag",
        "DECOMP_METHODS": "snmf",
        "DECOMP_COMPONENTS": "2",
        "DL_ALPHA": "20",
        "CLEAN_EXAMPLE_RATIO": "0.8",
        "BAG_SIZE": "2000",
        "IMAGE_BUDGET": "-1",
        "EXPL_MAX_IMAGES": "-1",
        "INPUT_DIR": "data/coco10/train_all",
        "IMAGE_ROOT": str(image_root),
        "SINGLE_OBJECT": "1",
        "CONCEPTS_VOCAB": "src/assets/coco10_vocab_rebuttal3.txt",
        "NUM_CONCEPT": "-1",
        "EXPL_PROMPT_MODE": "mcq",
        "EXPL_CHOICES": ",".join(CLASSES),
    }

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    device_queue: "queue.Queue[str]" = queue.Queue()
    for d in devices:
        device_queue.put(d)
    print_lock = threading.Lock()
    results = []

    def _worker(cfg):
        condition, crop_mode, seed = cfg
        device = device_queue.get()
        try:
            ok = run_one(ablation_root, condition, crop_mode, seed, env_overrides, device=device, print_lock=print_lock)
        finally:
            device_queue.put(device)
        return (condition, crop_mode, seed, ok)

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        for res in executor.map(_worker, configs_to_retry):
            results.append(res)

    print("\n=== P_bin_shuf retry summary ===")
    for condition, crop_mode, seed, ok in results:
        print(f"{condition:15s} {crop_mode:15s} seed={seed} {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
