from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from downstream_tasks.dataloader_course_h5 import CourseH5Dataset  # noqa: E402


RESOURCE_CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "our_tasks" / "resource_paths.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test CourseH5Dataset and print one batch summary.")
    parser.add_argument("--resource", type=str, default="5080", help="Resource key in resource_paths.yaml.")
    parser.add_argument("--data-root", type=str, default=None, help="Override data root directly.")
    parser.add_argument("--ckpt-root", type=str, default=None, help="Override checkpoint root directly.")
    parser.add_argument("--dataset", type=str, default="SEED", help="Dataset name, e.g. SEED.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the smoke test.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "test"], help="Split mode.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable train z-score normalization (default: true).",
    )
    parser.add_argument(
        "--cache-in-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache the full split in memory (default: false).",
    )
    return parser.parse_args()


def load_resource_configs() -> dict[str, dict[str, object]]:
    with RESOURCE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return {str(key): value for key, value in raw.items()}


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    resource_configs = load_resource_configs()
    if args.resource not in resource_configs:
        available = ", ".join(sorted(resource_configs))
        raise KeyError(f"Unknown resource '{args.resource}'. Available resources: {available}")

    resource_cfg = resource_configs[args.resource]
    data_root_str = args.data_root or str(resource_cfg["data_root"])
    ckpt_root_str = args.ckpt_root or str(resource_cfg["ckpt_root"])
    return Path(data_root_str), Path(ckpt_root_str)


def compute_tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    detached = tensor.detach().float()
    finite = detached[torch.isfinite(detached)]
    if finite.numel() == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
    }


def main() -> None:
    args = parse_args()
    data_root, ckpt_root = resolve_paths(args)
    dataset_path = data_root / args.dataset

    dataset = CourseH5Dataset(
        path=dataset_path,
        dataset_name=args.dataset,
        mode=args.mode,
        ckpt_root=ckpt_root,
        normalize=args.normalize,
        cache_in_memory=args.cache_in_memory,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate,
    )

    batch = next(iter(loader))
    sample_stats = compute_tensor_stats(batch["sample"])

    print(f"dataset name: {dataset.dataset_name}")
    print(f"mode: {dataset.mode}")
    print(f"data path: {dataset.dataset_dir}")
    print(f"h5 file path: {dataset.h5_path}")
    print(f"x key: {dataset.x_key}")
    print(f"y key: {dataset.y_key}")
    print(f"sample.shape: {tuple(batch['sample'].shape)}")
    print(f"sample.dtype: {batch['sample'].dtype}")
    if "label" in batch:
        print(f"label.shape: {tuple(batch['label'].shape)}")
        print(f"label.dtype: {batch['label'].dtype}")
        print(f"label unique values: {torch.unique(batch['label']).tolist()}")
    else:
        print("label.shape: <missing>")
        print("label.dtype: <missing>")
        print("label unique values: <missing>")
    print(f"pos.shape: {tuple(batch['pos'].shape)}")
    print(f"pos.dtype: {batch['pos'].dtype}")
    print(f"sample_id.shape: {tuple(batch['sample_id'].shape)}")
    print(
        "sample min/max/mean/std: "
        f"{sample_stats['min']:.6f} / {sample_stats['max']:.6f} / "
        f"{sample_stats['mean']:.6f} / {sample_stats['std']:.6f}"
    )
    print(f"normalization stats path: {dataset.train_stats_path}")
    print(f"channels not found: {json.dumps(dataset.channels_not_found)}")
    print(f"fallback channels: {json.dumps(dataset.fallback_channels)}")


if __name__ == "__main__":
    main()
