# combine_features.py

import torch
import os
import sys
import argparse

def combine_pth_files(root_dir, delete=False):
    pth_files = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".pth")])

    if not pth_files:
        print(f"No .pth files found in {root_dir}")
        return

    print(f"Found {len(pth_files)} .pth files.")

    combined_data = torch.load(pth_files[0], map_location="cpu")
    for key in combined_data:
        combined_data[key] = list(combined_data[key])

    for path in pth_files[1:]:
        print(f"Loading {path}...")
        data = torch.load(path, map_location="cpu")
        for key in data:
            combined_data[key].extend(data[key])

    for key in combined_data:
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
