#!/usr/bin/env python3
"""
One-off retry for the 6 P_bin_shuf configs of the 10-class rebuttal ablation
(outputs/rebuttal_ablation_10class) that FAILED at step 5 with an empty
concept bank.

Root cause: those 6 configs were dispatched by the main grid driver
(run_rebuttal_ablation_10class.py) BEFORE the P_bin_shuf condition was given
its CLEAN_EXAMPLE_RATIO="0.0" override (CONDITIONS["P_bin_shuf"] in
run_rebuttal_ablation_10class.py). They therefore ran under the global
CLEAN_EXAMPLE_RATIO=0.4 purity filter. But P_bin_shuf is the *shuffled*
control: every sample's prompt asks about a WRONG concept, so the VLM answers
"No [concept]" for the top-activating regions of every direction ->
positive_ratio ~0.01 for all -> 0/20 directions pass the 0.4 filter -> empty
bank -> step 5 aborts (RuntimeError). This is the documented, intended
behavior of the shuffle control; the fix is to keep every direction (cr=0.0),
exactly what the current source config already specifies.

This driver re-runs those 6 configs through the SAME 10-class run_one, which
now applies CONDITIONS["P_bin_shuf"] = {..., CLEAN_EXAMPLE_RATIO: "0.0"}.
Because ratio_tag = f"cr{ratio:g}", the new bank is written as
combined_concept_snmf_cr0_raw.pth (distinct from the stale, empty
combined_concept_snmf_cr0.4_raw.pth), so step 5's skip-check does NOT skip.
Steps 1-4 are reused untouched: the 10 per-tag feature files under
<dir>/features/ were generated correctly WITH the shuffled prompt
(shuffle_concept_prompt: True in each config's logs.log) -- only step 5's
filter threshold was wrong -- so step 5 recomputes decomposition + combine
with cr0.0, then steps 6-8 (regrounding, explainer, deletion eval, plots) run
normally to produce a non-empty bank whose faithfulness AUC is comparable to
P_bin's.

Runs ONLY on the devices you pass. Verify the target GPUs are free with
nvidia-smi before launching.

Usage:
    python scripts/retry_pbin_shuf_10class.py --devices cuda:1,cuda:3
"""
import argparse
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_rebuttal_ablation_10class import (  # noqa: E402
    ABLATION_ROOT_NAME, CLASSES, ROOT_DIR,
    TIER23_CROP_MODES, TIER23_SEEDS, run_one,
)

CONFIGS = [("P_bin_shuf", cm, seed) for cm in TIER23_CROP_MODES for seed in TIER23_SEEDS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="cuda:1,cuda:3")
    args = parser.parse_args()

    ablation_root = ROOT_DIR / ABLATION_ROOT_NAME
    image_root = ROOT_DIR / "data" / "coco10" / "val_masked_rebuttal10"

    # Exact copy of run_rebuttal_ablation_10class.main()'s env_overrides (the
    # forced invariants). The per-condition CLEAN_EXAMPLE_RATIO="0.0" override
    # for P_bin_shuf is applied INSIDE run_one via CONDITIONS["P_bin_shuf"]
    # (env.update(cond_cfg) after env.update(env_overrides)), so it correctly
    # wins over the 0.4 below.
    env_overrides = {
        "DECOMP_STRATEGY": "per_tag",
        "DECOMP_METHODS": "snmf",
        "DECOMP_COMPONENTS": "2",
        "DL_ALPHA": "20",
        "CLEAN_EXAMPLE_RATIO": "0.4",
        "BAG_SIZE": "900",
        "LOGIT_LENS_LAYER_SELECTION": "0",
        "IMAGE_BUDGET": "-1",
        "EXPL_MAX_IMAGES": "-1",
        "INPUT_DIR": "data/coco10/train_all",
        "IMAGE_ROOT": str(image_root),
        "SINGLE_OBJECT": "1",
        "CONCEPTS_VOCAB": "src/assets/coco10_vocab.txt",
        "NUM_CONCEPT": "-1",
        "EXPL_PROMPT_MODE": "mcq",
        "EXPL_CHOICES": ",".join(CLASSES),
    }

    # Only retry configs not already complete (run_one's own done-marker check
    # would skip them anyway, but be explicit).
    configs_to_retry = []
    for condition, crop_mode, seed in CONFIGS:
        cfg_dir = ablation_root / f"{condition}_{crop_mode}_seed{seed}"
        done_marker = cfg_dir / "eval" / "snmf" / "c_insertion_token_all_ranks.png"
        if done_marker.exists():
            print(f"Skip {cfg_dir.name}: already complete (found {done_marker.name})")
            continue
        configs_to_retry.append((condition, crop_mode, seed))

    if not configs_to_retry:
        print("Nothing to retry -- all P_bin_shuf configs already complete.")
        return

    print(f"Retrying {len(configs_to_retry)} P_bin_shuf configs on {args.devices}:")
    for c in configs_to_retry:
        print(f"  {c[0]}_{c[1]}_seed{c[2]}")

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    device_queue: "queue.Queue[str]" = queue.Queue()
    for d in devices:
        device_queue.put(d)
    print_lock = threading.Lock()
    results = []

    def _worker(cfg):
        condition, crop_mode, seed = cfg
        device = device_queue.get()
        try:
            ok = run_one(ablation_root, condition, crop_mode, seed, env_overrides,
                         device=device, print_lock=print_lock)
        finally:
            device_queue.put(device)
        return (condition, crop_mode, seed, ok)

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        for res in executor.map(_worker, configs_to_retry):
            results.append(res)

    print("\n=== P_bin_shuf (10-class) retry summary ===")
    for condition, crop_mode, seed, ok in results:
        print(f"{condition:15s} {crop_mode:15s} seed={seed} {'OK' if ok else 'FAILED'}")

    n_failed = sum(1 for *_, ok in results if not ok)
    if n_failed:
        print(f"\n{n_failed} of {len(results)} retries failed -- check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
