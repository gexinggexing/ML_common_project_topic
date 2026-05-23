from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import h5py
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "our_tasks" / "resource_paths.yaml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "course_project"
DEFAULT_DATASETS = ["BCIC2A", "CHINESE", "MDD", "SEED", "SLEEP", "BCI_Speech"]
SPLIT_FILES = {
    "train": "train.h5",
    "val": "val.h5",
    "test": "test_x_only.h5",
}
INFO_FILES = ("dataset_info_fixed.json", "dataset_info.json")
WINDOWS_RESOURCE_KEY = "5080"
LINUX_RESOURCE_KEY = "4090"
HS_RESOURCE_KEY = "hs"
H5_CHUNK_SIZE = 64


@dataclass
class DatasetKeyInfo:
    path: str
    shape: tuple[int, ...]
    dtype: str
    ndim: int
    kind: str


@dataclass
class SplitStats:
    split: str
    file_path: Path
    data_key: str | None
    label_key: str | None
    data_shape: tuple[int, ...] | None
    data_dtype: str | None
    label_shape: tuple[int, ...] | None
    label_dtype: str | None
    n_samples: int | None
    n_channels: int | None
    time_length: int | None
    mean: float | None
    std: float | None
    min_value: float | None
    max_value: float | None
    label_distribution: dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan course-project EEG H5 datasets and export summaries.")
    parser.add_argument(
        "--resource",
        type=str,
        default=None,
        help="Resource key in src/configs/our_tasks/resource_paths.yaml, e.g. 5080 / l40 / 4090 / hs.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directly. If relative, it is resolved from the repo root.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR.relative_to(PROJECT_ROOT)),
        help="Output directory for markdown and CSV artifacts. Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to scan. Defaults to BCIC2A CHINESE MDD SEED SLEEP BCI_Speech.",
    )
    return parser.parse_args()


def load_resource_configs() -> dict[str, dict[str, Any]]:
    with RESOURCE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw_configs = yaml.safe_load(handle)

    configs: dict[str, dict[str, Any]] = {}
    for key, value in raw_configs.items():
        configs[str(key)] = value
    return configs


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_data_root(args: argparse.Namespace, resource_configs: dict[str, dict[str, Any]]) -> tuple[str | None, Path]:
    if args.data_root is not None:
        selected_resource = args.resource
        data_root = resolve_path(args.data_root)
    elif args.resource is not None:
        if args.resource not in resource_configs:
            available = ", ".join(sorted(resource_configs))
            raise KeyError(f"Unknown resource '{args.resource}'. Available resources: {available}")
        selected_resource = args.resource
        data_root = Path(resource_configs[selected_resource]["data_root"])
    else:
        raise ValueError("Either --resource or --data-root must be provided.")

    if selected_resource == HS_RESOURCE_KEY and not data_root.exists():
        raise FileNotFoundError(
            "hs data_root does not exist: "
            f"{data_root}. Please copy the NAS data to /vePFS-0x0d/nzh/data/eeg/ML_project before scanning."
        )
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    return selected_resource, data_root


def format_number(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{digits}f}"


def format_shape(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "-"
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def format_dict_compact(data: dict[Any, Any]) -> str:
    if not data:
        return "{}"
    items = [f"{key}: {value}" for key, value in data.items()]
    return "{ " + ", ".join(items) + " }"


def format_windows_path(root: str, dataset_name: str) -> str:
    return str(PureWindowsPath(root) / dataset_name)


def format_posix_path(root: str, dataset_name: str) -> str:
    return str(PurePosixPath(root) / dataset_name)


def pick_existing_file(dataset_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate
    return None


def list_extra_files(dataset_dir: Path) -> list[str]:
    standard_paths = {dataset_dir / file_name for file_name in SPLIT_FILES.values()}
    standard_paths.update(dataset_dir / file_name for file_name in INFO_FILES)
    extras: list[str] = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file():
            continue
        if path in standard_paths:
            continue
        extras.append(str(path.relative_to(dataset_dir)))
    return extras


def load_dataset_info(info_path: Path | None) -> dict[str, Any]:
    if info_path is None:
        return {}
    with info_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_h5_items(handle: h5py.File) -> list[DatasetKeyInfo]:
    items: list[DatasetKeyInfo] = []

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            items.append(
                DatasetKeyInfo(
                    path=name,
                    shape=tuple(int(dim) for dim in obj.shape),
                    dtype=str(obj.dtype),
                    ndim=obj.ndim,
                    kind="dataset",
                )
            )

    handle.visititems(visitor)
    return items


def score_data_key(item: DatasetKeyInfo) -> int:
    score = 0
    name = item.path.lower().split("/")[-1]
    if name in {"x", "data", "eeg", "signal", "signals", "sample", "samples"}:
        score += 10
    if item.ndim >= 2:
        score += 5
    if item.ndim >= 3:
        score += 3
    if np.dtype(item.dtype).kind == "f":
        score += 3
    return score


def score_label_key(item: DatasetKeyInfo, data_shape: tuple[int, ...] | None) -> int:
    score = 0
    name = item.path.lower().split("/")[-1]
    if name in {"y", "label", "labels", "target", "targets"}:
        score += 10
    if item.ndim == 1:
        score += 4
    if np.dtype(item.dtype).kind in {"i", "u", "b"}:
        score += 4
    if data_shape is not None and item.shape and item.shape[0] == data_shape[0]:
        score += 4
    return score


def detect_data_key(items: list[DatasetKeyInfo]) -> DatasetKeyInfo | None:
    candidates = [item for item in items if item.kind == "dataset" and item.ndim >= 2]
    if not candidates:
        return None
    return max(candidates, key=score_data_key)


def detect_label_key(items: list[DatasetKeyInfo], data_item: DatasetKeyInfo | None) -> DatasetKeyInfo | None:
    data_shape = data_item.shape if data_item is not None else None
    candidates = [item for item in items if item.kind == "dataset" and item.ndim >= 1]
    scored = []
    for item in candidates:
        name = item.path.lower().split("/")[-1]
        dtype_kind = np.dtype(item.dtype).kind
        if name in {"y", "label", "labels", "target", "targets"} or dtype_kind in {"i", "u", "b"}:
            if score_label_key(item, data_shape) > 0:
                scored.append(item)
    if not scored:
        return None
    return max(scored, key=lambda item: score_label_key(item, data_shape))


def iter_dataset_chunks(dataset: h5py.Dataset, chunk_size: int = H5_CHUNK_SIZE):
    if dataset.ndim == 0:
        yield dataset[()]
        return
    total = dataset.shape[0]
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        yield dataset[start:stop]


def compute_array_stats(dataset: h5py.Dataset) -> tuple[float, float, float, float]:
    total_count = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    min_value = math.inf
    max_value = -math.inf

    for chunk in iter_dataset_chunks(dataset):
        array = np.asarray(chunk, dtype=np.float64)
        if array.size == 0:
            continue
        total_count += int(array.size)
        total_sum += float(array.sum())
        total_sq_sum += float(np.square(array).sum())
        min_value = min(min_value, float(array.min()))
        max_value = max(max_value, float(array.max()))

    if total_count == 0:
        return math.nan, math.nan, math.nan, math.nan

    mean = total_sum / total_count
    variance = max(total_sq_sum / total_count - mean * mean, 0.0)
    std = math.sqrt(variance)
    return mean, std, min_value, max_value


def compute_label_distribution(dataset: h5py.Dataset) -> dict[int, int]:
    counter: Counter[int] = Counter()
    for chunk in iter_dataset_chunks(dataset):
        values = np.asarray(chunk).reshape(-1)
        counter.update(int(value) for value in values.tolist())
    return dict(sorted(counter.items()))


def scan_split(split: str, file_path: Path) -> tuple[SplitStats, list[DatasetKeyInfo]]:
    with h5py.File(file_path, "r") as handle:
        items = collect_h5_items(handle)
        data_item = detect_data_key(items)
        label_item = detect_label_key(items, data_item)

        data_shape = data_item.shape if data_item is not None else None
        data_dtype = data_item.dtype if data_item is not None else None
        label_shape = label_item.shape if label_item is not None else None
        label_dtype = label_item.dtype if label_item is not None else None

        mean = std = min_value = max_value = None
        n_samples = n_channels = time_length = None
        if data_item is not None:
            dataset = handle[data_item.path]
            mean, std, min_value, max_value = compute_array_stats(dataset)
            if data_item.ndim >= 1:
                n_samples = int(data_item.shape[0])
            if data_item.ndim >= 2:
                n_channels = int(data_item.shape[1])
            if data_item.ndim >= 3:
                time_length = int(data_item.shape[2])

        label_distribution: dict[int, int] = {}
        if label_item is not None:
            label_distribution = compute_label_distribution(handle[label_item.path])

    stats = SplitStats(
        split=split,
        file_path=file_path,
        data_key=data_item.path if data_item is not None else None,
        label_key=label_item.path if label_item is not None else None,
        data_shape=data_shape,
        data_dtype=data_dtype,
        label_shape=label_shape,
        label_dtype=label_dtype,
        n_samples=n_samples,
        n_channels=n_channels,
        time_length=time_length,
        mean=mean,
        std=std,
        min_value=min_value,
        max_value=max_value,
        label_distribution=label_distribution,
    )
    return stats, items


def get_sampling_rate(dataset_info: dict[str, Any]) -> tuple[Any, Any]:
    processing = dataset_info.get("processing", {})
    dataset_section = dataset_info.get("dataset", {})
    return processing.get("target_sampling_rate"), dataset_section.get("original_sampling_rate")


def get_channel_names(dataset_info: dict[str, Any]) -> list[str]:
    channels = dataset_info.get("dataset", {}).get("channels", [])
    if isinstance(channels, list):
        return [str(channel) for channel in channels]
    return []


def get_category_names(dataset_info: dict[str, Any]) -> list[str]:
    categories = dataset_info.get("dataset", {}).get("category_list", [])
    if isinstance(categories, list):
        return [str(item) for item in categories]
    return []


def detect_notes(
    dataset_name: str,
    dataset_dir: Path,
    split_stats: dict[str, SplitStats],
    dataset_info_path: Path | None,
    dataset_info: dict[str, Any],
    extra_files: list[str],
) -> list[str]:
    notes: list[str] = []
    if dataset_info_path is None:
        notes.append("Missing dataset_info.json / dataset_info_fixed.json; metadata should be checked manually.")
    elif dataset_info_path.name == "dataset_info_fixed.json":
        notes.append("Uses dataset_info_fixed.json; keep later training configs aligned with the fixed version.")

    if extra_files:
        notes.append(f"Extra files or nested content detected: {', '.join(extra_files)}.")

    split_shapes = [stats.data_shape for stats in split_stats.values() if stats.data_shape is not None]
    if split_shapes:
        base_shape = split_shapes[0][1:]
        mismatch = [stats.split for stats in split_stats.values() if stats.data_shape is not None and stats.data_shape[1:] != base_shape]
        if mismatch:
            notes.append(f"Non-batch dimensions differ across splits: {', '.join(mismatch)}.")

    train_stats = split_stats.get("train")
    val_stats = split_stats.get("val")
    if train_stats is not None and val_stats is not None:
        train_labels = set(train_stats.label_distribution)
        val_labels = set(val_stats.label_distribution)
        if train_labels and val_labels and train_labels != val_labels:
            notes.append("Train and val do not share the exact same label set; handle carefully in dataloader and eval.")

    expected_classes = dataset_info.get("dataset", {}).get("num_labels")
    observed_classes = len(train_stats.label_distribution) if train_stats is not None else 0
    if expected_classes is not None and observed_classes and int(expected_classes) != observed_classes:
        notes.append(f"dataset_info declares num_labels={expected_classes}, but train observes {observed_classes} classes.")

    for split_name in ("train", "val"):
        stats = split_stats.get(split_name)
        if stats is None or not stats.label_distribution:
            continue
        label_min = min(stats.label_distribution)
        if label_min != 0:
            notes.append(f"{split_name} labels start at {label_min}, so they are not 0-based.")
            break

    for split_name, stats in split_stats.items():
        if split_name == "test" and stats.label_key is None:
            notes.append("test_x_only.h5 has no label key, which is expected; only prediction export is possible later.")
            break

    if dataset_name == "SEED":
        nested_seed_dir = dataset_dir / "SEED"
        if nested_seed_dir.exists():
            notes.append("SEED contains a nested SEED subdirectory with extra h5 content; avoid reading the wrong path.")

    if dataset_name == "SLEEP":
        stats = split_stats.get("train")
        if stats is not None and stats.time_length is not None and stats.time_length >= 6000:
            notes.append("SLEEP has long sequences (time_length=6000); watch memory usage and batch size later.")

    if dataset_name == "CHINESE":
        notes.append("CHINESE was originally sampled at 1000Hz, but processed data uses target_sampling_rate=200Hz.")

    if dataset_name == "BCIC2A":
        notes.append("BCIC2A shape [N, 22, 800] corresponds to a 4-second window at 200Hz.")

    if dataset_name == "BCI_Speech":
        notes.append("BCI_Speech shape [N, 64, 600] corresponds to a 3-second window at 200Hz.")
        notes.append("No dataset_info.json was found beside the H5 files; channel metadata must be supplied manually.")

    return notes


def build_dataset_markdown(
    dataset_name: str,
    dataset_dir: Path,
    split_stats: dict[str, SplitStats],
    file_structures: dict[str, list[DatasetKeyInfo]],
    dataset_info_path: Path | None,
    dataset_info: dict[str, Any],
    notes: list[str],
    resource_configs: dict[str, dict[str, Any]],
) -> str:
    windows_root = resource_configs[WINDOWS_RESOURCE_KEY]["data_root"]
    linux_root = resource_configs[LINUX_RESOURCE_KEY]["data_root"]
    hs_root = resource_configs[HS_RESOURCE_KEY]["data_root"]
    target_sr, original_sr = get_sampling_rate(dataset_info)
    channels = get_channel_names(dataset_info)
    categories = get_category_names(dataset_info)

    lines: list[str] = []
    lines.append(f"# {dataset_name}")
    lines.append("")
    lines.append("## Paths")
    lines.append("")
    lines.append(f"- Windows path: `{format_windows_path(windows_root, dataset_name)}`")
    lines.append(f"- Linux mnt path: `{format_posix_path(linux_root, dataset_name)}`")
    lines.append(f"- hs path: `{format_posix_path(hs_root, dataset_name)}`")
    lines.append(f"- Scanned path: `{dataset_dir}`")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for split_name, file_name in SPLIT_FILES.items():
        split_path = dataset_dir / file_name
        status = "present" if split_path.exists() else "missing"
        lines.append(f"- {file_name}: {status}")
    if dataset_info_path is None:
        lines.append("- dataset_info: missing")
    else:
        lines.append(f"- dataset_info: `{dataset_info_path.name}`")
    lines.append("")
    lines.append("## H5 Structure")
    lines.append("")
    for split_name in ("train", "val", "test"):
        lines.append(f"### {split_name}")
        lines.append("")
        items = file_structures.get(split_name, [])
        if not items:
            lines.append("- No datasets found")
            lines.append("")
            continue
        lines.append("| key | shape | dtype |")
        lines.append("| --- | --- | --- |")
        for item in items:
            lines.append(f"| `{item.path}` | `{format_shape(item.shape)}` | `{item.dtype}` |")
        lines.append("")
    lines.append("## Detected Keys")
    lines.append("")
    lines.append("| split | data_key | label_key | data_shape | label_shape |")
    lines.append("| --- | --- | --- | --- | --- |")
    for split_name in ("train", "val", "test"):
        stats = split_stats[split_name]
        lines.append(
            f"| {split_name} | `{stats.data_key or '-'}` | `{stats.label_key or '-'}` | "
            f"`{format_shape(stats.data_shape)}` | `{format_shape(stats.label_shape)}` |"
        )
    lines.append("")
    lines.append("## Data Stats")
    lines.append("")
    lines.append("| split | n_samples | n_channels | time_length | mean | std | min | max |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for split_name in ("train", "val", "test"):
        stats = split_stats[split_name]
        lines.append(
            f"| {split_name} | {format_number(stats.n_samples)} | {format_number(stats.n_channels)} | "
            f"{format_number(stats.time_length)} | {format_number(stats.mean)} | {format_number(stats.std)} | "
            f"{format_number(stats.min_value)} | {format_number(stats.max_value)} |"
        )
    lines.append("")
    lines.append("## Label Stats")
    lines.append("")
    lines.append("| split | label_key | n_classes | label_distribution |")
    lines.append("| --- | --- | --- | --- |")
    for split_name in ("train", "val", "test"):
        stats = split_stats[split_name]
        label_count = len(stats.label_distribution) if stats.label_distribution else 0
        label_dist = format_dict_compact(stats.label_distribution) if stats.label_distribution else "-"
        lines.append(f"| {split_name} | `{stats.label_key or '-'}` | {label_count} | `{label_dist}` |")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- Target sampling rate: `{target_sr if target_sr is not None else '-'}`")
    lines.append(f"- Original sampling rate: `{original_sr if original_sr is not None else '-'}`")
    lines.append(f"- Channel count in dataset_info: `{len(channels) if channels else '-'}`")
    lines.append(f"- Channel names: `{', '.join(channels) if channels else '-'}`")
    lines.append(f"- Category names: `{', '.join(categories) if categories else '-'}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No additional risks detected.")
    lines.append("")
    return "\n".join(lines)


def write_label_stats_csv(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = out_dir / "label_stats.csv"
    fieldnames = ["dataset", "split", "label", "count", "label_name", "label_key"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_markdown(
    dataset_results: list[dict[str, Any]],
    resource_configs: dict[str, dict[str, Any]],
    selected_resource: str | None,
    selected_data_root: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Course Project Dataset Summary")
    lines.append("")
    lines.append(f"- Generated by: `scripts/our_tasks/scan_h5_datasets.py`")
    lines.append(f"- Selected resource: `{selected_resource or 'custom'}`")
    lines.append(f"- Scanned data_root: `{selected_data_root}`")
    lines.append("")
    lines.append("## Resource Paths")
    lines.append("")
    lines.append("| resource | system | data_root | repo_root | ckpt_root | batch_size | num_workers |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for resource_key in (WINDOWS_RESOURCE_KEY, "l40", "4090", HS_RESOURCE_KEY):
        cfg = resource_configs[resource_key]
        lines.append(
            f"| {resource_key} | {cfg['system']} | `{cfg['data_root']}` | `{cfg['repo_root']}` | "
            f"`{cfg['ckpt_root']}` | {cfg['default_batch_size']} | {cfg['num_workers']} |"
        )
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(
        "| dataset | info file | data_key | label_key | train shape | val shape | test shape | "
        "target sr | classes(train) | notes |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for result in dataset_results:
        train_stats = result["split_stats"]["train"]
        val_stats = result["split_stats"]["val"]
        test_stats = result["split_stats"]["test"]
        info_name = result["dataset_info_path"].name if result["dataset_info_path"] is not None else "-"
        note_preview = " / ".join(result["notes"][:2]) if result["notes"] else "-"
        target_sr, _ = get_sampling_rate(result["dataset_info"])
        lines.append(
            f"| {result['dataset_name']} | `{info_name}` | `{train_stats.data_key or '-'}` | `{train_stats.label_key or '-'}` | "
            f"`{format_shape(train_stats.data_shape)}` | `{format_shape(val_stats.data_shape)}` | "
            f"`{format_shape(test_stats.data_shape)}` | `{target_sr if target_sr is not None else '-'}` | "
            f"{len(train_stats.label_distribution)} | {note_preview} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    resource_configs = load_resource_configs()
    selected_resource, data_root = resolve_data_root(args, resource_configs)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_results: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    for dataset_name in args.datasets:
        dataset_dir = data_root / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

        dataset_info_path = pick_existing_file(dataset_dir, INFO_FILES)
        dataset_info = load_dataset_info(dataset_info_path)
        categories = get_category_names(dataset_info)

        split_stats: dict[str, SplitStats] = {}
        file_structures: dict[str, list[DatasetKeyInfo]] = {}
        for split_name, file_name in SPLIT_FILES.items():
            split_path = dataset_dir / file_name
            if not split_path.exists():
                raise FileNotFoundError(f"Missing expected file: {split_path}")
            stats, items = scan_split(split_name, split_path)
            split_stats[split_name] = stats
            file_structures[split_name] = items

            for label_value, count in stats.label_distribution.items():
                label_name = categories[label_value] if 0 <= label_value < len(categories) else ""
                label_rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split_name,
                        "label": label_value,
                        "count": count,
                        "label_name": label_name,
                        "label_key": stats.label_key or "",
                    }
                )

        extra_files = list_extra_files(dataset_dir)
        notes = detect_notes(dataset_name, dataset_dir, split_stats, dataset_info_path, dataset_info, extra_files)

        dataset_markdown = build_dataset_markdown(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            split_stats=split_stats,
            file_structures=file_structures,
            dataset_info_path=dataset_info_path,
            dataset_info=dataset_info,
            notes=notes,
            resource_configs=resource_configs,
        )
        (out_dir / f"{dataset_name}.md").write_text(dataset_markdown, encoding="utf-8")

        dataset_results.append(
            {
                "dataset_name": dataset_name,
                "dataset_dir": dataset_dir,
                "dataset_info_path": dataset_info_path,
                "dataset_info": dataset_info,
                "split_stats": split_stats,
                "notes": notes,
            }
        )

    write_label_stats_csv(out_dir, label_rows)
    summary_markdown = build_summary_markdown(dataset_results, resource_configs, selected_resource, data_root)
    (out_dir / "dataset_summary.md").write_text(summary_markdown, encoding="utf-8")

    print(f"Scanned {len(dataset_results)} datasets.")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
