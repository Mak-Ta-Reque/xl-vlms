# combine_features.py

import torch
import os
import sys
import argparse

def combine_pth_files(root_dir, delete=False):
    pth_files = sorted(
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(".pth") and f != "combined_features.pth"
    )

    if not pth_files:
        print(f"No .pth files found in {root_dir}")
        return

    print(f"Found {len(pth_files)} .pth files.")

    def _is_per_sample_dict(d):
        # Nested per-module structure (e.g. hidden_states: {module_name: [tensors]}) --
        # every value is itself a per-sample list. Scalar-metadata dicts (e.g.
        # logit_lens_selection: one summary per tag/file, with int/str leaves)
        # aren't a per-sample structure and can't be merged the same way.
        return bool(d) and all(isinstance(v, list) for v in d.values())

    combined_data = torch.load(pth_files[0], map_location="cpu")
    for key in combined_data:
        if isinstance(combined_data[key], dict) and _is_per_sample_dict(combined_data[key]):
            # list(dict) would return its keys, not its values — merge per sub-key instead.
            combined_data[key] = {mk: list(mv) for mk, mv in combined_data[key].items()}
        elif isinstance(combined_data[key], (dict, str)):
            # Per-tag scalar (e.g. selected_layer, a string, or
            # logit_lens_selection, a scalar-leaf dict): keep one entry per
            # source file instead of list()-ing it, which would either crash
            # on a dict of ints or silently shred a string into characters.
            combined_data[key] = [combined_data[key]]
        else:
            combined_data[key] = list(combined_data[key])

    for path in pth_files[1:]:
        print(f"Loading {path}...")
        data = torch.load(path, map_location="cpu")
        for key in data:
            if isinstance(data[key], dict) and _is_per_sample_dict(data[key]):
                for mk, mv in data[key].items():
                    combined_data[key].setdefault(mk, []).extend(mv)
            elif isinstance(data[key], (dict, str)):
                combined_data[key].append(data[key])
            else:
                combined_data[key].extend(data[key])

    for key in combined_data:
        if isinstance(combined_data[key], dict):
            for mk, mv in combined_data[key].items():
                print(f"{key}[{mk}] total size: {len(mv)}")
        else:
            print(f"{key} total size: {len(combined_data[key])}")

    output_path = os.path.join(root_dir, "combined_features.pth")
    torch.save(combined_data, output_path)
    print(f"✅ Combined features saved to: {output_path}")

    if delete:
        for path in pth_files:
            os.remove(path)
        print("🗑️  All original .pth files deleted after combining.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine .pth feature files.")
    parser.add_argument("features_dir", type=str, help="Directory containing .pth files to combine")
    parser.add_argument("--delete", default=False, action="store_true", help="Delete original .pth files after combining")

    args = parser.parse_args()
    combine_pth_files(args.features_dir, delete=args.delete)
