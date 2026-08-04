#!/usr/bin/env python3
"""
Script 1 of 2: generate + STORE the raw predicted-token probability curves
for concept insertion/deletion, over 100% of the concept-vector coordinates,
for every config in outputs/rebuttal_ablation_10class.

This is the ONE expensive, model-dependent step. It decouples the forward
passes (raw probabilities) from the AUC/scaling/plotting math (done post-hoc
by scripts/posthoc_auc_curves.py, no model). After this runs once, any change
to the AUC definition, normalization, or plot is a cheap re-run of Script 2 --
never the model again. This is the fix for the recurring "changing the AUC
metric forces a full GPU re-run" problem.

For each config, using <config>/explanations/snmf/vlm_explanations.json
(+ the concept bank), it computes, for ranks 1/2/3, insertion & deletion,
value & random order:
  - the raw per-image mean probability curve over 100% of coordinates
  - a full fraction axis in [0, 1]
and writes one compact NPZ per config:
  <config>/eval/snmf/raw_prob_curves.npz
containing, per (rank, kind, order):
  fracs_r{R}                       [P]         fraction axis (identical across kinds; kept per rank)
  {kind}_{order}_r{R}_curves       [N_img, P]  per-image mean prob curves
  {kind}_{order}_r{R}_images       [N_img]     image paths (basenames)
  {kind}_{order}_r{R}_y0           [N_img]     each image's own fraction=0 value (the deletion baseline)
  {kind}_{order}_r{R}_ntokens      [N_img]     tokens averaged per image
where kind in {ins, del}, order in {value, random}, R in {1,2,3}.

Raw probabilities are stored (not AUC), so Script 2 can compute AUC any way.

Multi-GPU: one SUBPROCESS per config, round-robin across --devices, up to
len(devices) running in parallel. Deliberately subprocess-per-config (not
threads in one process) -- confirmed empirically that loading 2+ HF models
concurrently via ThreadPoolExecutor threads races on transformers'
device_map='auto' meta-tensor init ("Cannot copy out of meta tensor")
and cascades into failing every subsequent config in that process. Separate
processes each get their own independent Python/CUDA context, same pattern
already used by scripts/run_rebuttal_ablation_10class.py and every other
multi-GPU driver in this repo.

Usage:
    python scripts/gen_raw_prob_curves.py --devices cuda:0,cuda:1,cuda:2,cuda:3
    python scripts/gen_raw_prob_curves.py --devices cuda:0 --only P_bin_sliding_window_seed1
    # internal, invoked by the driver itself:
    python scripts/gen_raw_prob_curves.py --worker --config NAME --device cuda:0
"""
import argparse
import glob
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
ABLATION_ROOT = ROOT_DIR / "outputs" / "rebuttal_ablation_10class"
METHOD = "snmf"
NUM_POINTS = 100.0     # FULL 100% of coordinates (vs the pipeline's default 70)
CURVE_POINTS = 100     # curve resolution (>= the old 64 for a cleaner integral)
RANKS = [1, 2, 3]
ORDERS = ["value", "random"]


def find_configs():
    out = []
    for d in sorted(ABLATION_ROOT.glob("P_*")):
        if not d.is_dir():
            continue
        rj = d / "explanations" / METHOD / "vlm_explanations.json"
        banks = [
            p for p in glob.glob(str(d / "concept" / METHOD / f"combined_concept_{METHOD}_*_raw.pth"))
            if "negative" not in Path(p).name
        ]
        if rj.exists() and banks:
            out.append((d.name, str(rj), banks[0]))
    return out


def _stash(store, kind, order, rank, ev):
    import numpy as np
    fracs = np.asarray(ev._last_fracs, dtype=np.float32)
    per_img = ev._last_per_image_curves  # {image_path: {"curve": arr, "n_tokens": int}}
    imgs = sorted(per_img.keys())
    curves = (np.stack([np.asarray(per_img[k]["curve"], dtype=np.float32) for k in imgs], axis=0)
              if imgs else np.zeros((0, len(fracs)), np.float32))
    y0 = curves[:, 0].copy() if curves.size else np.zeros((0,), np.float32)
    ntok = np.asarray([per_img[k]["n_tokens"] for k in imgs], dtype=np.int32)
    pre = f"{kind}_{order}_r{rank}"
    store[f"fracs_r{rank}"] = fracs
    store[f"{pre}_curves"] = curves
    store[f"{pre}_images"] = np.asarray([Path(k).name for k in imgs])
    store[f"{pre}_y0"] = y0
    store[f"{pre}_ntokens"] = ntok


def run_worker(config_name: str, device: str) -> int:
    """Runs IN-PROCESS for exactly one config -- invoked as its own
    subprocess by the driver, never called concurrently with another
    worker in the same process."""
    import numpy as np
    sys.path.insert(0, str(ROOT_DIR / "eval"))
    sys.path.insert(0, str(ROOT_DIR))
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
    import torch
    from concept_deletion_eval import ConceptDeletionEvaluator

    d = ABLATION_ROOT / config_name
    existing_npz = d / "eval" / METHOD / "raw_prob_curves.npz"
    if existing_npz.exists():
        # Skip-check: lets re-launching the driver (e.g. to add a newly
        # freed GPU to the round-robin) pick up only the configs it hasn't
        # done yet, instead of silently re-spending ~44s/config on ones
        # already finished by a previous run.
        print(f"[{config_name}] SKIP: {existing_npz.name} already exists", flush=True)
        return 0
    results_json = d / "explanations" / METHOD / "vlm_explanations.json"
    banks = [p for p in glob.glob(str(d / "concept" / METHOD / f"combined_concept_{METHOD}_*_raw.pth"))
             if "negative" not in Path(p).name]
    if not results_json.exists() or not banks:
        print(f"[{config_name}] SKIP: missing results_json or concept bank", flush=True)
        return 1
    concept_path = banks[0]

    out_npz = d / "eval" / METHOD / "raw_prob_curves.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    with open(results_json) as f:
        layer_path = json.load(f).get("layer_path") or "model.language_model.norm"

    print(f"[{config_name}] loading model on {device}", flush=True)
    try:
        ev = ConceptDeletionEvaluator(
            model_name=__import__("os").environ.get("VLM_MODEL", "google/gemma-3n-E4B-it"),
            layer_path=layer_path,
            concept_path=concept_path,
            results_json=str(results_json),
            device=device,
            grad_top_zero_frac=0.0,
            concept_mutiply=True,
        )
    except torch.cuda.OutOfMemoryError:
        print(f"[{config_name}] OOM on {device}", flush=True)
        return 1

    store = {}
    for rank in RANKS:
        for order in ORDERS:
            ev.evaluate_token(rank=rank, num_points=NUM_POINTS, curve_points=CURVE_POINTS, order_mode=order)
            _stash(store, "del", order, rank, ev)
            ev.evaluate_token_insertion(rank=rank, num_points=NUM_POINTS, curve_points=CURVE_POINTS, order_mode=order)
            _stash(store, "ins", order, rank, ev)
            print(f"[{config_name}] rank{rank} {order}: ins+del curves captured", flush=True)

    np.savez_compressed(out_npz, **store)
    print(f"[{config_name}] DONE -> {out_npz.name} ({out_npz.stat().st_size // 1024} KB)", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    ap.add_argument("--only", default=None, help="process just this config name (debug)")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--config", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--device", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        sys.exit(run_worker(args.config, args.device))

    configs = find_configs()
    if args.only:
        configs = [c for c in configs if c[0] == args.only]
    print(f"Configs to process: {len(configs)}", flush=True)

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    print_lock = threading.Lock()
    dq: "queue.Queue[str]" = queue.Queue()
    for dv in devices:
        dq.put(dv)

    def launch_one(cfg):
        name, _rj, _bank = cfg
        device = dq.get()
        try:
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--worker", "--config", name, "--device", device]
            proc = subprocess.run(cmd, cwd=str(ROOT_DIR))
            ok = proc.returncode == 0
        finally:
            dq.put(device)
        with print_lock:
            print(f"[{name}] subprocess {'OK' if ok else 'FAILED'} (device {device})", flush=True)
        return (name, ok)

    from concurrent.futures import ThreadPoolExecutor
    results = []
    # ThreadPoolExecutor here only supervises `subprocess.run` calls (each a
    # real, isolated OS process) -- it is NOT loading models in-thread, so
    # the meta-tensor race from the earlier version doesn't apply.
    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        for res in ex.map(launch_one, configs):
            results.append(res)

    print("\n=== raw-curve generation summary ===", flush=True)
    for name, ok in results:
        print(f"  {name}: {'OK' if ok else 'FAILED'}")
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n{n_ok}/{len(results)} configs produced raw_prob_curves.npz")


if __name__ == "__main__":
    main()
