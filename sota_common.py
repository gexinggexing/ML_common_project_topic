#!/usr/bin/env python3
"""Shared utilities for SOTA-first EEG course-project sweeps."""

from __future__ import annotations

import csv
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from scipy import signal
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC


DATA_ROOT = Path("/mnt/dataset3/panxy/course/project1_data/course project/course project")
RESULTS_DIR = Path("artifacts/results")
SUBMISSION_DIR = Path("outputs/submission")
EXPECTED_ROWS = {
    "BCIC2A": 360,
    "SEED": 450,
    "CHINESE": 200,
    "MDD": 800,
    "SLEEP": 1945,
}
NUM_CLASSES = {
    "BCIC2A": 4,
    "SEED": 3,
    "CHINESE": 2,
    "MDD": 2,
    "SLEEP": 5,
}


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    num_classes: int
    category_list: list[str]
    channels: list[str]
    n_channels: int
    n_times: int
    fs: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def h5_path(root: Path, name: str, split: str) -> Path:
    filename = "test_x_only.h5" if split == "test" else f"{split}.h5"
    return root / name / filename


def load_meta(root: Path, name: str) -> DatasetMeta:
    info_path = root / name / "dataset_info_fixed.json"
    if not info_path.exists():
        info_path = root / name / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))["dataset"]
    with h5py.File(h5_path(root, name, "train"), "r") as h5:
        _, n_channels, n_times = h5["X"].shape
    return DatasetMeta(
        name=name,
        num_classes=int(info["num_labels"]),
        category_list=list(info["category_list"]),
        channels=list(info.get("channels", [])),
        n_channels=int(n_channels),
        n_times=int(n_times),
        fs=200.0,
    )


def read_split(root: Path, name: str, split: str) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(h5_path(root, name, split), "r") as h5:
        x = np.asarray(h5["X"], dtype=np.float64)
        y = np.asarray(h5["y"], dtype=np.int64) if "y" in h5 else None
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y


def load_arrays(root: Path, name: str) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    return {
        "train": read_split(root, name, "train"),
        "val": read_split(root, name, "val"),
        "test": read_split(root, name, "test"),
    }


def bandpass_epochs(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyquist = fs / 2.0
    low = max(low, 0.5)
    high = min(high, nyquist - 1.0)
    if not low < high:
        raise ValueError(f"Invalid band {low}-{high} for fs={fs}")
    sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float64, copy=False)


def welch_bandpower(
    x: np.ndarray,
    fs: float,
    bands: list[tuple[float, float]],
    *,
    relative: bool = False,
) -> np.ndarray:
    nperseg = min(256, x.shape[-1])
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, axis=-1, scaling="density")
    total = psd.sum(axis=-1, keepdims=True)
    parts: list[np.ndarray] = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            raise ValueError(f"No Welch bins for band {low}-{high}")
        power = psd[..., mask].mean(axis=-1)
        if relative:
            power = power / np.maximum(total[..., 0], 1e-12)
        parts.append(np.log(np.maximum(power, 1e-12)))
    return np.stack(parts, axis=-1)


def flatten_features(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def time_stats_features(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1)
    std = x.std(axis=-1)
    centered = x - mean[..., None]
    rms = np.sqrt(np.mean(x * x, axis=-1))
    ptp = np.ptp(x, axis=-1)
    denom = np.maximum(std, 1e-8)
    skew = np.mean((centered / denom[..., None]) ** 3, axis=-1)
    kurt = np.mean((centered / denom[..., None]) ** 4, axis=-1)
    return np.concatenate([mean, std, rms, ptp, skew, kurt], axis=1)


def logvar_features(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.var(x, axis=-1), 1e-12))


def covariance_features(x: np.ndarray, *, corr: bool = True) -> np.ndarray:
    centered = x - x.mean(axis=-1, keepdims=True)
    cov = np.einsum("nct,ndt->ncd", centered, centered) / max(x.shape[-1] - 1, 1)
    if corr:
        diag = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 1e-12))
        cov = cov / np.maximum(diag[:, :, None] * diag[:, None, :], 1e-12)
    idx = np.triu_indices(x.shape[1])
    return cov[:, idx[0], idx[1]]


def pair_indices(channels: list[str], pairs: Iterable[tuple[str, str]]) -> list[tuple[int, int]]:
    index = {name.upper().replace("-", "_"): i for i, name in enumerate(channels)}
    found: list[tuple[int, int]] = []
    for left, right in pairs:
        lkey = left.upper().replace("-", "_")
        rkey = right.upper().replace("-", "_")
        if lkey in index and rkey in index:
            found.append((index[lkey], index[rkey]))
    return found


def asymmetry_from_band_features(
    band_features: np.ndarray,
    channels: list[str],
    pairs: Iterable[tuple[str, str]],
) -> np.ndarray:
    found = pair_indices(channels, pairs)
    if not found:
        return np.empty((band_features.shape[0], 0), dtype=np.float64)
    parts = [band_features[:, left, :] - band_features[:, right, :] for left, right in found]
    return np.concatenate(parts, axis=1)


def classifiers(seed: int, *, fast: bool = False) -> dict[str, Pipeline]:
    models: dict[str, Any] = {
        "ridge": RidgeClassifier(alpha=1.0, class_weight="balanced"),
        "logreg": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            random_state=seed,
        ),
        "linear_svm": LinearSVC(C=0.5, class_weight="balanced", max_iter=8000, random_state=seed),
        "svm_rbf": SVC(C=3.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed),
        "extratrees": ExtraTreesClassifier(
            n_estimators=200 if fast else 600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "randomforest": RandomForestClassifier(
            n_estimators=200 if fast else 600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    if not fast:
        models["mlp"] = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=1e-3,
            learning_rate_init=1e-3,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=25,
            max_iter=400,
            random_state=seed,
        )
    out: dict[str, Pipeline] = {}
    for name, estimator in models.items():
        scaler = RobustScaler() if name in {"extratrees", "randomforest"} else StandardScaler()
        out[name] = Pipeline([("scaler", scaler), ("clf", estimator)])
    return out


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": json.dumps(
            confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
        ),
    }


def evaluate_classifiers(
    feature_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    *,
    seed: int,
    fast: bool,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clf_name, clf in classifiers(seed, fast=fast).items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf.fit(x_train, y_train)
        pred = clf.predict(x_val).astype(np.int64)
        row: dict[str, Any] = {
            "combo": f"{feature_name}_{clf_name}",
            "feature_family": feature_name,
            "classifier": clf_name,
            "feature_dim": int(x_train.shape[1]),
            "seed": seed,
        }
        if extra:
            row.update(extra)
        row.update(metric_row(y_val, pred, num_classes))
        rows.append(row)
    return rows


def train_predict(
    row: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    fast: bool,
) -> np.ndarray:
    clf_name = str(row["classifier"])
    clf = classifiers(seed, fast=fast).get(clf_name)
    if clf is None:
        raise ValueError(f"Unknown classifier {clf_name}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(x_train, y_train)
    return clf.predict(x_test).astype(np.int64)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path: Path, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in pred.astype(int).tolist():
            f.write(f"{value}\n")


def validate_predictions(path: Path, dataset: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != EXPECTED_ROWS[dataset]:
        raise ValueError(f"{path}: expected {EXPECTED_ROWS[dataset]} rows, got {len(lines)}")
    valid = set(range(NUM_CLASSES[dataset]))
    bad = []
    for idx, line in enumerate(lines, start=1):
        if line.strip() != line or not line.strip().isdigit() or int(line) not in valid:
            bad.append((idx, line))
            if len(bad) >= 5:
                break
    if bad:
        raise ValueError(f"{path}: invalid prediction lines {bad}")


def selected_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: (float(row["balanced_accuracy"]), float(row["accuracy"])))


def write_note(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}", ""]
    body.extend(lines)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
