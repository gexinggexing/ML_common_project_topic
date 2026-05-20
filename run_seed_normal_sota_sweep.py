#!/usr/bin/env python3
"""Normal SEED-only sweep inspired by SEED emotion-recognition literature.

This script deliberately excludes row-order templates and any direct use of
validation labels as training labels during model selection. It evaluates
feature/classifier combinations that are compatible with the course release:

- differential-entropy style band features,
- hemispheric asymmetry features,
- channel covariance / correlation features,
- raw-signal PCA baselines,
- graph-inspired channel node features aggregated with ordinary classifiers.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from scipy import linalg, signal, stats
from sklearn.decomposition import PCA
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.svm import SVC


DATA_ROOT = Path("/mnt/dataset3/panxy/course/project1_data/course project/course project")
RESULTS_DIR = Path("artifacts/results")
OUTPUTS_DIR = Path("outputs/seed_normal_sota_sweep")
SUBMISSION_DIR = Path("outputs/submission")
EXPECTED_TEST_ROWS = 450
SEED_BANDS = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]
WINDOWS = [(0, 200), (200, 400), (0, 400)]
PAIR_NAMES = [
    ("FP1", "FP2"), ("AF3", "AF4"), ("F7", "F8"), ("F5", "F6"),
    ("F3", "F4"), ("F1", "F2"), ("FT7", "FT8"), ("FC5", "FC6"),
    ("FC3", "FC4"), ("FC1", "FC2"), ("T7", "T8"), ("C5", "C6"),
    ("C3", "C4"), ("C1", "C2"), ("TP7", "TP8"), ("CP5", "CP6"),
    ("CP3", "CP4"), ("CP1", "CP2"), ("P7", "P8"), ("P5", "P6"),
    ("P3", "P4"), ("P1", "P2"), ("PO7", "PO8"), ("PO5", "PO6"),
    ("PO3", "PO4"), ("O1", "O2"), ("CB1", "CB2"),
]


@dataclass(frozen=True)
class Meta:
    channels: list[str]
    n_channels: int
    n_times: int
    num_classes: int
    fs: float = 200.0


@dataclass(frozen=True)
class Combo:
    feature: str
    classifier: str
    params: dict[str, Any]

    @property
    def name(self) -> str:
        param_bits = []
        for key, value in sorted(self.params.items()):
            if isinstance(value, float):
                value = f"{value:g}"
            param_bits.append(f"{key}-{value}")
        suffix = "" if not param_bits else "_" + "_".join(param_bits)
        return f"{self.feature}_{self.classifier}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--submission-dir", type=Path, default=SUBMISSION_DIR)
    parser.add_argument("--update-submission", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def h5_path(root: Path, split: str) -> Path:
    filename = "test_x_only.h5" if split == "test" else f"{split}.h5"
    return root / "SEED" / filename


def load_meta(root: Path) -> Meta:
    info_path = root / "SEED" / "dataset_info_fixed.json"
    if not info_path.exists():
        info_path = root / "SEED" / "dataset_info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)["dataset"]
    with h5py.File(h5_path(root, "train"), "r") as h5:
        _, n_channels, n_times = h5["X"].shape
    return Meta(
        channels=list(info["channels"]),
        n_channels=int(n_channels),
        n_times=int(n_times),
        num_classes=int(info["num_labels"]),
    )


def read_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(h5_path(root, split), "r") as h5:
        x = np.asarray(h5["X"], dtype=np.float64)
        y = np.asarray(h5["y"], dtype=np.int64) if "y" in h5 else None
    return x, y


def bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    high = min(high, fs / 2.0 - 1.0)
    sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def pair_indices(channels: list[str]) -> list[tuple[int, int]]:
    idx = {name.upper(): i for i, name in enumerate(channels)}
    return [(idx[left], idx[right]) for left, right in PAIR_NAMES if left in idx and right in idx]


def logvar_features(x: np.ndarray, meta: Meta, include_windows: bool) -> np.ndarray:
    windows = WINDOWS if include_windows else [(0, meta.n_times)]
    parts: list[np.ndarray] = []
    for start, stop in windows:
        segment = x[:, :, start:stop]
        freqs, psd = signal.welch(
            segment,
            fs=meta.fs,
            nperseg=min(128, segment.shape[-1]),
            noverlap=min(64, max(0, segment.shape[-1] // 2)),
            axis=-1,
            scaling="density",
        )
        for low, high in SEED_BANDS:
            mask = (freqs >= low) & (freqs < high)
            power = psd[..., mask].sum(axis=-1)
            parts.append(0.5 * np.log(2.0 * np.pi * np.e * np.maximum(power, 1e-12)))
    return np.concatenate(parts, axis=1)


def welch_features(x: np.ndarray, meta: Meta, relative: bool) -> np.ndarray:
    freqs, psd = signal.welch(
        x,
        fs=meta.fs,
        nperseg=min(200, meta.n_times),
        noverlap=100,
        axis=-1,
        scaling="density",
    )
    band_parts: list[np.ndarray] = []
    total = psd[..., (freqs >= 1) & (freqs < 50)].sum(axis=-1)
    for low, high in SEED_BANDS:
        mask = (freqs >= low) & (freqs < high)
        power = psd[..., mask].sum(axis=-1)
        if relative:
            power = power / np.maximum(total, 1e-12)
        band_parts.append(np.log(np.maximum(power, 1e-12)))
    return np.concatenate(band_parts, axis=1)


def add_asymmetry(base: np.ndarray, meta: Meta) -> np.ndarray:
    # base is shaped as channels * feature_blocks.
    blocks = base.reshape(base.shape[0], -1, meta.n_channels).transpose(0, 2, 1)
    pairs = pair_indices(meta.channels)
    diffs = []
    ratios = []
    for left, right in pairs:
        left_values = blocks[:, left, :]
        right_values = blocks[:, right, :]
        diffs.append(left_values - right_values)
        ratios.append(left_values / np.maximum(np.abs(right_values), 1e-6))
    if not diffs:
        return base
    return np.concatenate([base, np.concatenate(diffs, axis=1), np.concatenate(ratios, axis=1)], axis=1)


def hjorth_features(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=-1, keepdims=True)
    dx = np.diff(centered, axis=-1)
    ddx = np.diff(dx, axis=-1)
    var0 = np.var(centered, axis=-1)
    var1 = np.var(dx, axis=-1)
    var2 = np.var(ddx, axis=-1)
    mobility = np.sqrt(var1 / np.maximum(var0, 1e-12))
    complexity = np.sqrt(var2 / np.maximum(var1, 1e-12)) / np.maximum(mobility, 1e-12)
    return np.concatenate([np.log(np.maximum(var0, 1e-12)), mobility, complexity], axis=1)


def stats_features(x: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            x.mean(axis=-1),
            x.std(axis=-1),
            stats.skew(x, axis=-1, bias=False, nan_policy="omit"),
            stats.kurtosis(x, axis=-1, bias=False, nan_policy="omit"),
            np.percentile(x, 10, axis=-1),
            np.percentile(x, 90, axis=-1),
        ],
        axis=1,
    )


def corr_features(x: np.ndarray, meta: Meta) -> np.ndarray:
    rows = []
    tri = np.triu_indices(meta.n_channels, k=1)
    for trial in x:
        corr = np.corrcoef(trial)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        rows.append(corr[tri])
    return np.asarray(rows, dtype=np.float64)


def cov_log_features(x: np.ndarray, meta: Meta) -> np.ndarray:
    rows = []
    tri = np.triu_indices(meta.n_channels)
    eye = np.eye(meta.n_channels)
    for trial in x:
        centered = trial - trial.mean(axis=-1, keepdims=True)
        cov = centered @ centered.T / max(1, centered.shape[-1] - 1)
        cov = cov / max(float(np.trace(cov)), 1e-12)
        log_cov = linalg.logm(cov + 1e-4 * eye)
        rows.append(np.real(log_cov)[tri])
    return np.asarray(rows, dtype=np.float64)


def raw_pca_input(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        arr = x
    elif mode == "zscore":
        arr = (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)
    elif mode == "diff":
        arr = np.diff(x, axis=-1)
    else:
        raise ValueError(mode)
    return arr.reshape(arr.shape[0], -1)


def extract_feature(x: np.ndarray, meta: Meta, name: str) -> np.ndarray:
    if name == "de_full":
        return logvar_features(x, meta, include_windows=False)
    if name == "de_windows":
        return logvar_features(x, meta, include_windows=True)
    if name == "de_windows_asym":
        return add_asymmetry(logvar_features(x, meta, include_windows=True), meta)
    if name == "welch_abs":
        return welch_features(x, meta, relative=False)
    if name == "welch_rel":
        return welch_features(x, meta, relative=True)
    if name == "welch_rel_asym":
        return add_asymmetry(welch_features(x, meta, relative=True), meta)
    if name == "de_hjorth_stats":
        return np.concatenate(
            [
                logvar_features(x, meta, include_windows=True),
                hjorth_features(x),
                stats_features(x),
            ],
            axis=1,
        )
    if name == "corr":
        return corr_features(x, meta)
    if name == "cov_log":
        return cov_log_features(x, meta)
    if name == "raw_zscore":
        return raw_pca_input(x, "zscore")
    if name == "raw_diff":
        return raw_pca_input(x, "diff")
    raise ValueError(f"Unknown feature: {name}")


def identity_transformer() -> FunctionTransformer:
    return FunctionTransformer(lambda x: x, validate=False)


def classifier_pipeline(name: str, seed: int, params: dict[str, Any]) -> Pipeline:
    pca = params.get("pca")
    scaler_steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if pca:
        scaler_steps.append(("pca", PCA(n_components=pca, svd_solver="randomized", random_state=seed)))
    if name == "svm_rbf":
        clf = SVC(
            C=float(params.get("C", 10.0)),
            gamma=params.get("gamma", "scale"),
            kernel="rbf",
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline(scaler_steps + [("clf", clf)])
    if name == "svm_linear":
        clf = SVC(C=float(params.get("C", 1.0)), kernel="linear", class_weight="balanced", random_state=seed)
        return Pipeline(scaler_steps + [("clf", clf)])
    if name == "logreg":
        clf = LogisticRegression(
            C=float(params.get("C", 1.0)),
            class_weight="balanced",
            max_iter=8000,
            random_state=seed,
        )
        return Pipeline(scaler_steps + [("clf", clf)])
    if name == "ridge":
        return Pipeline(scaler_steps + [("clf", RidgeClassifier(alpha=float(params.get("alpha", 1.0)), class_weight="balanced"))])
    if name == "knn":
        steps = [("scaler", StandardScaler())]
        if pca:
            steps.append(("pca", PCA(n_components=pca, svd_solver="randomized", random_state=seed)))
        if params.get("normalize", False):
            steps.append(("norm", Normalizer()))
        clf = KNeighborsClassifier(
            n_neighbors=int(params.get("k", 5)),
            weights=str(params.get("weights", "distance")),
            metric=str(params.get("metric", "minkowski")),
        )
        return Pipeline(steps + [("clf", clf)])
    if name == "extratrees":
        clf = ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 800)),
            max_features=params.get("max_features", "sqrt"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([("identity", identity_transformer()), ("clf", clf)])
    if name == "randomforest":
        clf = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 800)),
            max_features=params.get("max_features", "sqrt"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([("identity", identity_transformer()), ("clf", clf)])
    if name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(params.get("hidden", (256, 64))),
            activation="relu",
            alpha=float(params.get("alpha", 1e-3)),
            learning_rate_init=float(params.get("lr", 1e-3)),
            batch_size=64,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=40,
            max_iter=800,
            random_state=seed,
        )
        return Pipeline(scaler_steps + [("clf", clf)])
    raise ValueError(f"Unknown classifier: {name}")


def build_combos() -> list[Combo]:
    combos: list[Combo] = []
    for feature in ["de_full", "de_windows", "de_windows_asym", "welch_abs", "welch_rel", "welch_rel_asym", "de_hjorth_stats"]:
        combos.extend(
            [
                Combo(feature, "logreg", {"C": 0.1}),
                Combo(feature, "logreg", {"C": 1.0}),
                Combo(feature, "svm_rbf", {"C": 3.0}),
                Combo(feature, "svm_rbf", {"C": 10.0}),
                Combo(feature, "knn", {"k": 3, "metric": "cosine", "normalize": True}),
                Combo(feature, "extratrees", {"min_samples_leaf": 1}),
            ]
        )
    for feature in ["corr", "cov_log"]:
        combos.extend(
            [
                Combo(feature, "svm_rbf", {"C": 3.0, "pca": 64}),
                Combo(feature, "svm_rbf", {"C": 10.0, "pca": 128}),
                Combo(feature, "logreg", {"C": 0.3, "pca": 128}),
                Combo(feature, "ridge", {"alpha": 1.0, "pca": 128}),
                Combo(feature, "knn", {"k": 1, "metric": "cosine", "normalize": True, "pca": 128}),
                Combo(feature, "knn", {"k": 5, "metric": "cosine", "normalize": True, "pca": 128}),
            ]
        )
    combos.extend(
        [
            Combo("de_windows_asym", "mlp", {"hidden": (256, 128), "alpha": 0.001}),
            Combo("de_hjorth_stats", "mlp", {"hidden": (512, 128), "alpha": 0.001, "pca": 256}),
            Combo("welch_rel_asym", "randomforest", {"min_samples_leaf": 1}),
            Combo("cov_log", "extratrees", {"min_samples_leaf": 1}),
        ]
    )
    return combos


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": json.dumps(confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def validate_predictions(path: Path) -> None:
    values = [int(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(values) != EXPECTED_TEST_ROWS:
        raise ValueError(f"{path}: expected {EXPECTED_TEST_ROWS} rows, got {len(values)}")
    invalid = sorted(set(values) - {0, 1, 2})
    if invalid:
        raise ValueError(f"{path}: invalid labels {invalid}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    meta = load_meta(args.data_root)
    train_x, train_y = read_split(args.data_root, "train")
    val_x, val_y = read_split(args.data_root, "val")
    test_x, _ = read_split(args.data_root, "test")
    if train_y is None or val_y is None:
        raise ValueError("SEED train/val labels are required")

    combos = build_combos()
    feature_names = sorted({combo.feature for combo in combos})
    feature_cache: dict[tuple[str, str], np.ndarray] = {}
    for feature in feature_names:
        print(f"extract {feature}")
        feature_cache[(feature, "train")] = extract_feature(train_x, meta, feature)
        feature_cache[(feature, "val")] = extract_feature(val_x, meta, feature)
        feature_cache[(feature, "test")] = extract_feature(test_x, meta, feature)
        print(
            f"  train={feature_cache[(feature, 'train')].shape} "
            f"val={feature_cache[(feature, 'val')].shape}"
        )

    rows: list[dict[str, Any]] = []
    for idx, combo in enumerate(combos, start=1):
        print(f"[{idx:02d}/{len(combos)}] {combo.name}")
        clf = classifier_pipeline(combo.classifier, args.seed, combo.params)
        x_train = feature_cache[(combo.feature, "train")]
        x_val = feature_cache[(combo.feature, "val")]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf.fit(x_train, train_y)
        val_pred = clf.predict(x_val).astype(np.int64)
        row = {
            "dataset": "SEED",
            "combo": combo.name,
            "feature": combo.feature,
            "classifier": combo.classifier,
            "params": json.dumps(combo.params, sort_keys=True),
            "feature_dim": int(x_train.shape[1]),
        }
        row.update(metric_row(val_y, val_pred, meta.num_classes))
        rows.append(row)
        print(
            f"  acc={row['accuracy']:.4f} bacc={row['balanced_accuracy']:.4f} "
            f"f1={row['macro_f1']:.4f}"
        )

    best = max(rows, key=lambda row: (float(row["balanced_accuracy"]), float(row["accuracy"])))
    best_combo = next(combo for combo in combos if combo.name == best["combo"])
    trainval_x = np.concatenate([feature_cache[(best_combo.feature, "train")], feature_cache[(best_combo.feature, "val")]], axis=0)
    trainval_y = np.concatenate([train_y, val_y], axis=0)
    test_features = feature_cache[(best_combo.feature, "test")]
    final_clf = classifier_pipeline(best_combo.classifier, args.seed, best_combo.params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        final_clf.fit(trainval_x, trainval_y)
    test_pred = final_clf.predict(test_features).astype(np.int64)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / "seed_normal_sota_sweep_metrics.csv"
    selection_path = args.results_dir / "seed_normal_sota_sweep_selection_summary.csv"
    output_path = args.outputs_dir / "SEED.txt"
    combo_output_path = args.outputs_dir / f"SEED_{best_combo.name}.txt"
    write_csv(metrics_path, rows)
    metrics_path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    selection = dict(best)
    selection["selected_output"] = str(output_path)
    selection["selected_combo_output"] = str(combo_output_path)
    selection["updated_submission"] = str(args.submission_dir / "SEED.txt") if args.update_submission else ""
    write_csv(selection_path, [selection])
    selection_path.with_suffix(".json").write_text(json.dumps([selection], indent=2), encoding="utf-8")
    write_predictions(output_path, test_pred)
    write_predictions(combo_output_path, test_pred)
    validate_predictions(output_path)
    if args.update_submission:
        submission_path = args.submission_dir / "SEED.txt"
        write_predictions(submission_path, test_pred)
        validate_predictions(submission_path)
    print(json.dumps({"num_combos": len(combos), "best": selection}, indent=2))


if __name__ == "__main__":
    main()
