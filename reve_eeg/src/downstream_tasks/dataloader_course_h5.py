"""H5-backed course-project dataset loader for REVE downstream experiments."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_ROOT = PROJECT_ROOT / "checkpoints"
DEFAULT_STATS_DIR = PROJECT_ROOT / "data" / "course_project" / "stats"
MODE_TO_FILE = {
    "train": "train.h5",
    "val": "val.h5",
    "test": "test_x_only.h5",
}
INFO_FILE_CANDIDATES = ("dataset_info_fixed.json", "dataset_info.json")
X_KEY_CANDIDATES = ("X", "x", "eeg", "data", "sample", "signals")
Y_KEY_CANDIDATES = ("y", "label", "labels", "target", "targets")
CHANNEL_ALIASES = {
    "CB1": "M1",
    "CB2": "M2",
}
COURSE_DATASET_SPECS = {
    "BCIC2A": {"n_chans": 22, "duration_sec": 4.0, "n_classes": 4},
    "CHINESE": {"n_chans": 22, "duration_sec": 1.0, "n_classes": 2},
    "MDD": {"n_chans": 20, "duration_sec": 1.0, "n_classes": 2},
    "SEED": {"n_chans": 62, "duration_sec": 2.0, "n_classes": 3},
    "SLEEP": {"n_chans": 6, "duration_sec": 30.0, "n_classes": 5},
}


@dataclass
class PositionResolution:
    positions: torch.Tensor
    missing_channels: list[str]
    fallback_channels: list[str]


def _ensure_absolute(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _canonical_name(name: str) -> str:
    return name.strip().replace(" ", "").upper()


def _find_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=8)
def _load_position_bank(position_json_path: str) -> dict[str, np.ndarray]:
    path = Path(position_json_path)
    raw_positions = json.loads(path.read_text(encoding="utf-8"))
    return {
        _canonical_name(name): np.asarray(coords, dtype=np.float32)
        for name, coords in raw_positions.items()
    }


def _resolve_position_json_path(ckpt_root: str | Path | None) -> Path:
    root = DEFAULT_CKPT_ROOT if ckpt_root is None else _ensure_absolute(ckpt_root)
    candidates = [
        root / "reve-positions" / "positions.json",
        root / "positions.json",
    ]
    position_json_path = _find_existing_path(candidates)
    if position_json_path is None:
        joined = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Could not find positions.json. Checked: {joined}")
    return position_json_path


def _stable_fallback_position(channel_name: str) -> np.ndarray:
    digest = hashlib.sha256(channel_name.encode("utf-8")).digest()
    values = [int.from_bytes(digest[offset : offset + 4], "little") / (2**32 - 1) for offset in (0, 4, 8)]
    radius = 0.085
    x = (values[0] * 2.0 - 1.0) * radius
    y = (values[1] * 2.0 - 1.0) * radius
    z = 0.01 + values[2] * 0.08
    return np.asarray([x, y, z], dtype=np.float32)


def _lookup_single_channel(channel_name: str, position_bank: dict[str, np.ndarray]) -> np.ndarray | None:
    canonical = _canonical_name(channel_name)
    direct = position_bank.get(canonical)
    if direct is not None:
        return direct

    alias = CHANNEL_ALIASES.get(canonical)
    if alias is None:
        return None
    return position_bank.get(_canonical_name(alias))


def _resolve_channel_position(channel_name: str, position_bank: dict[str, np.ndarray]) -> tuple[np.ndarray, bool]:
    direct = _lookup_single_channel(channel_name, position_bank)
    if direct is not None:
        return direct, False

    separator = "-" if "-" in channel_name else "_" if "_" in channel_name else None
    if separator is not None:
        parts = [part for part in channel_name.split(separator) if part]
        resolved_parts = [_lookup_single_channel(part, position_bank) for part in parts]
        if parts and all(part is not None for part in resolved_parts):
            stacked = np.stack([part for part in resolved_parts if part is not None], axis=0)
            return stacked.mean(axis=0, dtype=np.float32), False

    return _stable_fallback_position(channel_name), True


def resolve_positions_for_channels(
    channel_names: list[str],
    ckpt_root: str | Path | None = None,
    positions_path: str | Path | None = None,
) -> PositionResolution:
    if positions_path is not None:
        loaded = np.load(_ensure_absolute(positions_path), allow_pickle=True)
        positions = torch.as_tensor(loaded, dtype=torch.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"Positions loaded from {positions_path} must have shape [C, 3], got {positions.shape}")
        if channel_names and positions.shape[0] != len(channel_names):
            raise ValueError(
                f"Positions file channel count mismatch: expected {len(channel_names)}, got {positions.shape[0]}"
            )
        return PositionResolution(positions=positions, missing_channels=[], fallback_channels=[])

    position_json_path = _resolve_position_json_path(ckpt_root)
    position_bank = _load_position_bank(str(position_json_path))

    if not channel_names:
        return PositionResolution(positions=torch.empty(0, 3, dtype=torch.float32), missing_channels=[], fallback_channels=[])

    resolved_positions: list[np.ndarray] = []
    fallback_channels: list[str] = []
    for channel_name in channel_names:
        position, used_fallback = _resolve_channel_position(channel_name, position_bank)
        resolved_positions.append(position)
        if used_fallback:
            fallback_channels.append(channel_name)

    if fallback_channels:
        warnings.warn(
            "Using deterministic fallback positions for channels not found in the REVE position bank: "
            + ", ".join(fallback_channels),
            stacklevel=2,
        )

    return PositionResolution(
        positions=torch.as_tensor(np.stack(resolved_positions, axis=0), dtype=torch.float32),
        missing_channels=fallback_channels.copy(),
        fallback_channels=fallback_channels,
    )


def _collect_h5_dataset_paths(handle: h5py.File) -> list[str]:
    dataset_paths: list[str] = []

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            dataset_paths.append(name)

    handle.visititems(visitor)
    return dataset_paths


def _select_h5_key(
    handle: h5py.File,
    candidates: tuple[str, ...],
    *,
    min_ndim: int = 1,
    integer_only: bool = False,
) -> str | None:
    dataset_paths = _collect_h5_dataset_paths(handle)
    for candidate in candidates:
        for dataset_path in dataset_paths:
            if dataset_path.split("/")[-1].lower() != candidate.lower():
                continue
            dataset = handle[dataset_path]
            if dataset.ndim < min_ndim:
                continue
            if integer_only and np.dtype(dataset.dtype).kind not in {"i", "u", "b"}:
                continue
            return dataset_path

    for dataset_path in dataset_paths:
        dataset = handle[dataset_path]
        if dataset.ndim < min_ndim:
            continue
        if integer_only and np.dtype(dataset.dtype).kind not in {"i", "u", "b"}:
            continue
        return dataset_path

    return None


def _compute_streaming_stats(dataset: h5py.Dataset, chunk_size: int = 64) -> dict[str, float | int]:
    total_count = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    finite_count = 0
    non_finite_count = 0
    min_value = math.inf
    max_value = -math.inf

    n_samples = int(dataset.shape[0])
    for start in range(0, n_samples, chunk_size):
        stop = min(n_samples, start + chunk_size)
        chunk = np.asarray(dataset[start:stop], dtype=np.float64)
        finite = np.isfinite(chunk)
        finite_values = chunk[finite]
        finite_count += int(finite_values.size)
        non_finite_count += int(chunk.size - finite_values.size)
        if finite_values.size == 0:
            continue
        total_count += int(finite_values.size)
        total_sum += float(finite_values.sum())
        total_sq_sum += float(np.square(finite_values).sum())
        min_value = min(min_value, float(finite_values.min()))
        max_value = max(max_value, float(finite_values.max()))

    if total_count == 0:
        mean = 0.0
        std = 1.0
        min_value = 0.0
        max_value = 0.0
    else:
        mean = total_sum / total_count
        variance = max(total_sq_sum / total_count - mean * mean, 0.0)
        std = math.sqrt(variance)
        if std < 1e-8:
            std = 1.0

    return {
        "mean": mean,
        "std": std,
        "min": min_value,
        "max": max_value,
        "count": total_count,
        "finite_count": finite_count,
        "non_finite_count": non_finite_count,
    }


class CourseH5Dataset(Dataset):
    """Read one course-project EEG dataset split from H5 without touching nested directories."""

    def __init__(
        self,
        path: str | Path,
        mode: str,
        *,
        dataset_name: str | None = None,
        ckpt_root: str | Path | None = None,
        positions_path: str | Path | None = None,
        electrodes: list[str] | None = None,
        x_key: str | None = "X",
        y_key: str | None = "y",
        normalize: bool = True,
        cache_in_memory: bool = False,
        stats_dir: str | Path | None = None,
        resample_to_200hz: bool = False,
        scale_factor: float = 1.0,
        stats_chunk_size: int = 64,
    ) -> None:
        super().__init__()

        if mode not in MODE_TO_FILE:
            raise ValueError(f"Unsupported mode '{mode}'. Expected one of: {', '.join(sorted(MODE_TO_FILE))}")

        self.mode = mode
        self.dataset_dir = _ensure_absolute(path)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")

        self.dataset_name = dataset_name or self.dataset_dir.name
        self.ckpt_root = DEFAULT_CKPT_ROOT if ckpt_root is None else _ensure_absolute(ckpt_root)
        self.positions_path = None if positions_path is None else _ensure_absolute(positions_path)
        self.normalize = normalize
        self.cache_in_memory = cache_in_memory
        self.resample_to_200hz = resample_to_200hz
        self.scale_factor = float(scale_factor)
        self.stats_dir = DEFAULT_STATS_DIR if stats_dir is None else _ensure_absolute(stats_dir)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self.stats_chunk_size = stats_chunk_size

        self.h5_file_name = MODE_TO_FILE[self.mode]
        self.h5_path = self.dataset_dir / self.h5_file_name
        if not self.h5_path.exists():
            raise FileNotFoundError(f"Expected split file does not exist: {self.h5_path}")

        self.dataset_info_path = _find_existing_path([self.dataset_dir / name for name in INFO_FILE_CANDIDATES])
        self.dataset_info = self._load_dataset_info()
        self.channel_names = electrodes or self._load_channel_names()
        self.dataset_spec = COURSE_DATASET_SPECS.get(self.dataset_name, {})
        self.expected_n_channels = int(self.dataset_spec["n_chans"]) if "n_chans" in self.dataset_spec else None
        self.expected_n_classes = int(self.dataset_spec["n_classes"]) if "n_classes" in self.dataset_spec else None
        self.expected_duration_sec = (
            float(self.dataset_spec["duration_sec"])
            if "duration_sec" in self.dataset_spec
            else self._read_float_from_info("processing", "window_sec")
        )
        self.target_sampling_rate = 200.0
        self.expected_timepoints = (
            int(round(self.expected_duration_sec * self.target_sampling_rate))
            if self.expected_duration_sec is not None
            else None
        )

        self.x_key = None
        self.y_key = None
        self.data_shape: tuple[int, ...] | None = None
        self.data_dtype: str | None = None
        self.label_shape: tuple[int, ...] | None = None
        self.label_dtype: str | None = None
        self.transpose_sample = False
        self.has_label = False
        self.n_samples = 0
        self.n_channels = 0
        self.time_length = 0
        self._x_cache: np.ndarray | None = None
        self._y_cache: np.ndarray | None = None
        self._h5_file: h5py.File | None = None
        self._labels_raw: np.ndarray | None = None
        self._inspect_current_split(preferred_x_key=x_key, preferred_y_key=y_key)
        if not self.channel_names:
            self.channel_names = [f"CH{index:03d}" for index in range(self.n_channels)]

        self.train_stats_path = self.stats_dir / f"{self.dataset_name}_train_stats.json"
        self.label_mapping_path = self.train_stats_path
        self.train_stats = self._load_or_create_train_artifacts()
        self.train_mean = float(self.train_stats["mean"])
        self.train_std = float(self.train_stats["std"])
        label_mapping_raw = self.train_stats.get("label_mapping", {})
        self.label_mapping = {int(key): int(value) for key, value in label_mapping_raw.items()}
        inverse_mapping_raw = self.train_stats.get("inverse_label_mapping", {})
        self.inverse_label_mapping = {int(key): int(value) for key, value in inverse_mapping_raw.items()}
        if self.expected_n_classes is not None and len(self.label_mapping) != self.expected_n_classes:
            raise ValueError(
                f"{self.dataset_name} expected {self.expected_n_classes} classes from the scanned course-project spec, "
                f"got {len(self.label_mapping)}"
            )

        if self.has_label:
            self._labels_raw = self._load_split_labels()
            self._validate_label_mapping(self._labels_raw)

        position_resolution = resolve_positions_for_channels(
            channel_names=self.channel_names,
            ckpt_root=self.ckpt_root,
            positions_path=self.positions_path,
        )
        self.positions = position_resolution.positions.float()
        self.channels_not_found = position_resolution.missing_channels
        self.fallback_channels = position_resolution.fallback_channels
        if self.positions.numel() and self.positions.shape[0] != self.n_channels:
            raise ValueError(
                f"Resolved positions shape {tuple(self.positions.shape)} does not match channel count {self.n_channels}"
            )

        if self.cache_in_memory:
            self._cache_split_arrays()

        if self.expected_timepoints is not None and self.time_length != self.expected_timepoints:
            warnings.warn(
                f"{self.dataset_name}/{self.mode} has time_length={self.time_length}, "
                f"but duration*200Hz suggests {self.expected_timepoints}. Automatic resampling is "
                f"{'enabled' if self.resample_to_200hz else 'disabled'}.",
                stacklevel=2,
            )

    def _load_dataset_info(self) -> dict[str, Any]:
        if self.dataset_info_path is None:
            return {}
        return json.loads(self.dataset_info_path.read_text(encoding="utf-8"))

    def _load_channel_names(self) -> list[str]:
        channels = self.dataset_info.get("dataset", {}).get("channels", [])
        if isinstance(channels, list):
            return [str(channel) for channel in channels]
        return []

    def _read_float_from_info(self, section: str, key: str) -> float | None:
        value = self.dataset_info.get(section, {}).get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _inspect_current_split(self, preferred_x_key: str | None, preferred_y_key: str | None) -> None:
        with h5py.File(self.h5_path, "r") as handle:
            if preferred_x_key is not None and preferred_x_key in handle:
                self.x_key = preferred_x_key
            else:
                self.x_key = _select_h5_key(handle, X_KEY_CANDIDATES, min_ndim=3, integer_only=False)

            if self.x_key is None:
                raise KeyError(f"Could not detect EEG data key in {self.h5_path}")

            x_dataset = handle[self.x_key]
            self.data_shape = tuple(int(dim) for dim in x_dataset.shape)
            self.data_dtype = str(x_dataset.dtype)
            if x_dataset.ndim != 3:
                raise ValueError(
                    f"{self.h5_path}::{self.x_key} must be a 3D array [N, C, T] or [N, T, C], got {self.data_shape}"
                )

            self.n_samples = int(self.data_shape[0])
            metadata_channels = self.expected_n_channels or (len(self.channel_names) if self.channel_names else None)
            second_dim = int(self.data_shape[1])
            third_dim = int(self.data_shape[2])
            if metadata_channels is not None and metadata_channels > 0 and second_dim != metadata_channels and third_dim == metadata_channels:
                self.transpose_sample = True
                self.n_channels = third_dim
                self.time_length = second_dim
            else:
                self.transpose_sample = False
                self.n_channels = second_dim
                self.time_length = third_dim

            if self.expected_n_channels is not None and self.n_channels != self.expected_n_channels:
                raise ValueError(
                    f"{self.dataset_name}/{self.mode} expected {self.expected_n_channels} channels "
                    f"from the scanned course-project spec, got {self.n_channels}"
                )

            if self.mode != "test":
                if preferred_y_key is not None and preferred_y_key in handle:
                    self.y_key = preferred_y_key
                else:
                    self.y_key = _select_h5_key(handle, Y_KEY_CANDIDATES, min_ndim=1, integer_only=True)
                if self.y_key is None:
                    raise KeyError(f"Could not detect label key in {self.h5_path}")
                y_dataset = handle[self.y_key]
                self.label_shape = tuple(int(dim) for dim in y_dataset.shape)
                self.label_dtype = str(y_dataset.dtype)
                self.has_label = True
            else:
                self.y_key = None
                self.label_shape = None
                self.label_dtype = None
                self.has_label = False

    def _load_or_create_train_artifacts(self) -> dict[str, Any]:
        if self.train_stats_path.exists():
            return json.loads(self.train_stats_path.read_text(encoding="utf-8"))

        train_h5_path = self.dataset_dir / MODE_TO_FILE["train"]
        if not train_h5_path.exists():
            raise FileNotFoundError(f"Train split required for artifacts is missing: {train_h5_path}")

        with h5py.File(train_h5_path, "r") as handle:
            train_x_key = _select_h5_key(handle, X_KEY_CANDIDATES, min_ndim=3, integer_only=False)
            train_y_key = _select_h5_key(handle, Y_KEY_CANDIDATES, min_ndim=1, integer_only=True)
            if train_x_key is None or train_y_key is None:
                raise KeyError(f"Could not determine train keys for artifacts in {train_h5_path}")

            x_dataset = handle[train_x_key]
            stats = _compute_streaming_stats(x_dataset, chunk_size=self.stats_chunk_size)

            raw_labels = np.asarray(handle[train_y_key][:]).reshape(-1).astype(np.int64)
            unique_labels = sorted(int(label) for label in np.unique(raw_labels).tolist())
            label_mapping = {label: index for index, label in enumerate(unique_labels)}
            inverse_label_mapping = {index: label for label, index in label_mapping.items()}

        artifacts = {
            "dataset_name": self.dataset_name,
            "train_h5_path": str(train_h5_path),
            "x_key": train_x_key,
            "y_key": train_y_key,
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
            "min": float(stats["min"]),
            "max": float(stats["max"]),
            "count": int(stats["count"]),
            "finite_count": int(stats["finite_count"]),
            "non_finite_count": int(stats["non_finite_count"]),
            "label_mapping": {str(label): mapped for label, mapped in label_mapping.items()},
            "inverse_label_mapping": {str(mapped): label for mapped, label in inverse_label_mapping.items()},
        }
        self.train_stats_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
        return artifacts

    def _validate_label_mapping(self, labels: np.ndarray) -> None:
        unseen_labels = sorted({int(label) for label in np.unique(labels).tolist()} - set(self.label_mapping))
        if unseen_labels:
            raise ValueError(
                f"{self.dataset_name}/{self.mode} contains labels not present in the train mapping: {unseen_labels}"
            )

    def _load_split_labels(self) -> np.ndarray:
        if self._y_cache is not None:
            return self._y_cache
        if self.y_key is None:
            raise ValueError("This split does not have labels.")
        with h5py.File(self.h5_path, "r") as handle:
            labels = np.asarray(handle[self.y_key][:]).reshape(-1).astype(np.int64)
        return labels

    def _cache_split_arrays(self) -> None:
        with h5py.File(self.h5_path, "r") as handle:
            self._x_cache = np.asarray(handle[self.x_key][:], dtype=np.float32)
            if self.has_label and self.y_key is not None:
                self._y_cache = np.asarray(handle[self.y_key][:]).reshape(-1).astype(np.int64)

    def _ensure_h5_is_open(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def _read_raw_sample(self, index: int) -> np.ndarray:
        if self._x_cache is not None:
            return np.asarray(self._x_cache[index], dtype=np.float32)
        handle = self._ensure_h5_is_open()
        return np.asarray(handle[self.x_key][index], dtype=np.float32)

    def _prepare_sample(self, sample: np.ndarray) -> np.ndarray:
        if sample.ndim != 2:
            raise ValueError(f"Expected one sample to have shape [C, T] or [T, C], got {sample.shape}")

        if self.transpose_sample:
            sample = sample.transpose(1, 0)

        if sample.shape != (self.n_channels, self.time_length):
            raise ValueError(
                f"Prepared sample shape mismatch for {self.dataset_name}/{self.mode}: "
                f"expected {(self.n_channels, self.time_length)}, got {sample.shape}"
            )

        if self.scale_factor != 1.0:
            sample = sample / self.scale_factor

        if self.resample_to_200hz and self.expected_timepoints is not None and sample.shape[1] != self.expected_timepoints:
            sample_tensor = torch.as_tensor(sample, dtype=torch.float32).unsqueeze(0)
            sample = (
                F.interpolate(sample_tensor, size=self.expected_timepoints, mode="linear", align_corners=False)
                .squeeze(0)
                .numpy()
            )

        if self.normalize:
            sample = (sample - self.train_mean) / self.train_std

        return np.asarray(sample, dtype=np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._prepare_sample(self._read_raw_sample(index))
        item: dict[str, Any] = {
            "sample": torch.as_tensor(sample, dtype=torch.float32),
            "sample_id": int(index),
        }

        if self.has_label:
            assert self._labels_raw is not None
            raw_label = int(self._labels_raw[index])
            mapped_label = self.label_mapping[raw_label]
            item["label"] = torch.tensor(mapped_label, dtype=torch.long)
            item["raw_label"] = raw_label

        return item

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        samples = torch.stack([item["sample"] for item in batch], dim=0)
        sample_ids = torch.tensor([int(item["sample_id"]) for item in batch], dtype=torch.long)
        positions = self.positions.unsqueeze(0).repeat(len(batch), 1, 1).float()

        collated: dict[str, torch.Tensor] = {
            "sample": samples,
            "pos": positions,
            "sample_id": sample_ids,
        }

        if "label" in batch[0]:
            collated["label"] = torch.stack([item["label"] for item in batch], dim=0).long().view(-1)

        return collated
