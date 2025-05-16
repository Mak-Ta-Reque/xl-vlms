# combine_features.py

import torch
import os
import sys

def combine_pth_files(root_dir):
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python combine_features.py <features_dir>")
        sys.exit(1)

    combine_pth_files(sys.argv[1])
