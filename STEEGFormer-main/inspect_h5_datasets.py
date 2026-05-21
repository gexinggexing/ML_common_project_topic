import argparse
from pathlib import Path

import h5py
import numpy as np


def summarize_dataset(path):
    with h5py.File(path, "r") as f:
        x_shape = tuple(f["X"].shape)
        x_dtype = str(f["X"].dtype)
        y_info = None
        if "y" in f:
            y = f["y"][()]
            vals, counts = np.unique(y, return_counts=True)
            y_info = list(zip(vals.tolist(), counts.tolist()))
    return x_shape, x_dtype, y_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/dataset3/panxy/course/project1_data/course project/course project")
    parser.add_argument("--patch-size", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.root)
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        print("=" * 100)
        print(ds_dir.name)
        for name in ["train.h5", "val.h5", "test_x_only.h5"]:
            path = ds_dir / name
            if not path.exists():
                print(f"  {name}: MISSING")
                continue
            x_shape, x_dtype, y_info = summarize_dataset(path)
            n, c, t = x_shape
            ok_patch = (t % args.patch_size == 0)
            print(f"  {name}: X={x_shape} dtype={x_dtype} patch_ok={ok_patch}")
            if y_info is not None:
                print(f"    y={y_info}")


if __name__ == "__main__":
    main()
