#!/usr/bin/env python3
"""Targeted validation sweep for BCIC2A and SEED.

The sweep focuses on literature-backed EEG feature families that are cheap to
run on the course splits:

- BCIC2A: one-vs-rest CSP / filter-bank CSP with several classifier heads.
- SEED: band log-power / differential-entropy style features, plus hemispheric
  asymmetry and Hjorth statistics.

It writes validation metrics for every combination, then refits the selected
best combination per dataset on train+val and updates the two submission files.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import linalg, signal
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


DATA_ROOT = Path("/mnt/dataset3/panxy/course/project1_data/course project/course project")
DATASETS = ("BCIC2A", "SEED")
EXPECTED_TEST_ROWS = {"BCIC2A": 360, "SEED": 450}
RESULTS_DIR = Path("artifacts/results")
OUTPUTS_DIR = Path("outputs/targeted_bcic_seed_sweep")
SUBMISSION_DIR = Path("outputs/submission")

BCIC_NARROW_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
BCIC_WIDE_BANDS = [(4, 12), (8, 16), (12, 20), (16, 24), (20, 30), (30, 40)]
BCIC_DENSE_BANDS = [(low, low + 4) for low in range(4, 40, 4)]
BCIC_MU_BETA_BANDS = [(8, 30)]
BCIC_LOGPOWER_BANDS = [(4, 8), (8, 13), (13, 20), (20, 30), (30, 40)]
SEED_BANDS = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]

SEED_PAIR_NAMES = [
    ("FP1", "FP2"),
    ("AF3", "AF4"),
    ("F7", "F8"),
    ("F5", "F6"),
    ("F3", "F4"),
    ("F1", "F2"),
    ("FT7", "FT8"),
    ("FC5", "FC6"),
    ("FC3", "FC4"),
    ("FC1", "FC2"),
    ("T7", "T8"),
    ("C5", "C6"),
    ("C3", "C4"),
    ("C1", "C2"),
    ("TP7", "TP8"),
    ("CP5", "CP6"),
    ("CP3", "CP4"),
    ("CP1", "CP2"),
    ("P7", "P8"),
    ("P5", "P6"),
    ("P3", "P4"),
    ("P1", "P2"),
    ("PO7", "PO8"),
    ("PO5", "PO6"),
    ("PO3", "PO4"),
    ("O1", "O2"),
    ("CB1", "CB2"),
]


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    num_classes: int
    category_list: list[str]
    channels: list[str]
    n_channels: int
    n_times: int
    fs: float


@dataclass(frozen=True)
class Combo:
    dataset: str
    name: str
    feature_family: str
    classifier: str
    params: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--submission-dir", type=Path, default=SUBMISSION_DIR)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--update-submission", action="store_true")
    return parser.parse_args()


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
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)["dataset"]
    with h5py.File(h5_path(root, name, "train"), "r") as h5:
        _, n_channels, n_times = h5["X"].shape
    # The course release uses 200-sample CBraMod patches; BCIC2A has 4 patches
    # and SEED has 2 patches, so 200 Hz is the consistent downstream rate.
    fs = 200.0
    return DatasetMeta(
        name=name,
        num_classes=int(info["num_labels"]),
        category_list=list(info["category_list"]),
        channels=list(info.get("channels", [])),
        n_channels=int(n_channels),
        n_times=int(n_times),
        fs=fs,
    )


def read_split(root: Path, name: str, split: str) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(h5_path(root, name, split), "r") as h5:
        x = np.asarray(h5["X"], dtype=np.float64)
        y = np.asarray(h5["y"], dtype=np.int64) if "y" in h5 else None
    return x, y


def load_arrays(root: Path, name: str) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    return {
        "train": read_split(root, name, "train"),
        "val": read_split(root, name, "val"),
        "test": read_split(root, name, "test"),
    }


def bandpass_epochs(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyquist = fs / 2.0
    high = min(high, nyquist - 1.0)
    low = max(low, 0.5)
    if not low < high:
        raise ValueError(f"Invalid band {low}-{high} for fs={fs}")
    sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float64, copy=False)


def trial_covariances(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=-1, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", centered, centered)
    traces = np.trace(covs, axis1=1, axis2=2)
    traces = np.where(np.abs(traces) < 1e-12, 1.0, traces)
    covs = covs / traces[:, None, None]
    return covs


def fit_ovr_csp_filters(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    components_per_side: int,
    reg: float = 1e-5,
) -> list[np.ndarray]:
    covs = trial_covariances(x)
    filters: list[np.ndarray] = []
    eye = np.eye(x.shape[1], dtype=np.float64)
    for cls in range(num_classes):
        cls_cov = covs[y == cls].mean(axis=0)
        rest_cov = covs[y != cls].mean(axis=0)
        composite = cls_cov + rest_cov
        scale = np.trace(composite) / composite.shape[0]
        ridge = eye * max(reg * scale, reg)
        evals, evecs = linalg.eigh(cls_cov + ridge, composite + 2.0 * ridge)
        order = np.argsort(evals)
        selected = np.r_[order[:components_per_side], order[-components_per_side:]]
        filters.append(evecs[:, selected].T)
    return filters


def apply_csp_filters(x: np.ndarray, filters: list[np.ndarray]) -> np.ndarray:
    features: list[np.ndarray] = []
    centered = x - x.mean(axis=-1, keepdims=True)
    for filt in filters:
        projected = np.einsum("fc,nct->nft", filt, centered)
        variances = np.var(projected, axis=-1)
        normalized = variances / np.maximum(variances.sum(axis=1, keepdims=True), 1e-12)
        features.append(np.log(np.maximum(normalized, 1e-12)))
    return np.concatenate(features, axis=1)


def fbcsp_features(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_xs: list[np.ndarray],
    meta: DatasetMeta,
    bands: list[tuple[float, float]],
    components_per_side: int,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    train_parts: list[np.ndarray] = []
    eval_parts: list[list[np.ndarray]] = [[] for _ in eval_xs]
    for low, high in bands:
        train_band = bandpass_epochs(train_x, meta.fs, low, high)
        filters = fit_ovr_csp_filters(
            train_band,
            train_y,
            meta.num_classes,
            components_per_side=components_per_side,
        )
        train_parts.append(apply_csp_filters(train_band, filters))
        for idx, eval_x in enumerate(eval_xs):
            eval_band = bandpass_epochs(eval_x, meta.fs, low, high)
            eval_parts[idx].append(apply_csp_filters(eval_band, filters))
    train_features = np.concatenate(train_parts, axis=1)
    eval_features = [np.concatenate(parts, axis=1) for parts in eval_parts]
    details = {
        "bands": bands,
        "components_per_side": components_per_side,
        "feature_dim": int(train_features.shape[1]),
    }
    return train_features, eval_features, details


def welch_logpower(x: np.ndarray, fs: float, bands: list[tuple[float, float]]) -> np.ndarray:
    nperseg = min(256, x.shape[-1])
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, axis=-1, scaling="density")
    band_features: list[np.ndarray] = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            raise ValueError(f"No Welch bins for band {low}-{high}")
        power = psd[..., mask].mean(axis=-1)
        band_features.append(np.log(np.maximum(power, 1e-12)))
    return np.stack(band_features, axis=-1)


def hjorth_features(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=-1, keepdims=True)
    dx = np.diff(centered, axis=-1)
    ddx = np.diff(dx, axis=-1)
    var0 = np.var(centered, axis=-1)
    var1 = np.var(dx, axis=-1)
    var2 = np.var(ddx, axis=-1)
    mobility = np.sqrt(var1 / np.maximum(var0, 1e-12))
    complexity = np.sqrt(var2 / np.maximum(var1, 1e-12)) / np.maximum(mobility, 1e-12)
    return np.concatenate(
        [
            np.log(np.maximum(var0, 1e-12)),
            mobility,
            complexity,
        ],
        axis=1,
    )


def flatten_band_features(features: np.ndarray) -> np.ndarray:
    return features.reshape(features.shape[0], -1)


def seed_pair_indices(channels: list[str]) -> list[tuple[int, int]]:
    index = {name.upper(): idx for idx, name in enumerate(channels)}
    pairs: list[tuple[int, int]] = []
    for left, right in SEED_PAIR_NAMES:
        if left in index and right in index:
            pairs.append((index[left], index[right]))
    return pairs


def asymmetry_features(band_features: np.ndarray, meta: DatasetMeta) -> np.ndarray:
    pairs = seed_pair_indices(meta.channels)
    if not pairs:
        return np.empty((band_features.shape[0], 0), dtype=np.float64)
    diffs = [band_features[:, left, :] - band_features[:, right, :] for left, right in pairs]
    return np.concatenate(diffs, axis=1)


def seed_spectral_features(
    train_x: np.ndarray,
    eval_xs: list[np.ndarray],
    meta: DatasetMeta,
    include_asymmetry: bool,
    include_hjorth: bool,
    include_stats: bool,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    all_x = [train_x] + eval_xs
    feature_sets: list[np.ndarray] = []
    for x in all_x:
        de = welch_logpower(x, meta.fs, SEED_BANDS)
        parts = [flatten_band_features(de)]
        if include_asymmetry:
            parts.append(asymmetry_features(de, meta))
        if include_hjorth:
            parts.append(hjorth_features(x))
        if include_stats:
            parts.extend(
                [
                    x.mean(axis=-1),
                    x.std(axis=-1),
                    np.percentile(x, 10, axis=-1),
                    np.percentile(x, 90, axis=-1),
                ]
            )
        feature_sets.append(np.concatenate(parts, axis=1))
    details = {
        "bands": SEED_BANDS,
        "include_asymmetry": include_asymmetry,
        "include_hjorth": include_hjorth,
        "include_stats": include_stats,
        "feature_dim": int(feature_sets[0].shape[1]),
    }
    return feature_sets[0], feature_sets[1:], details


def logpower_features(
    train_x: np.ndarray,
    eval_xs: list[np.ndarray],
    meta: DatasetMeta,
    bands: list[tuple[float, float]],
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    all_features = [flatten_band_features(welch_logpower(x, meta.fs, bands)) for x in [train_x] + eval_xs]
    details = {"bands": bands, "feature_dim": int(all_features[0].shape[1])}
    return all_features[0], all_features[1:], details


def identity_transformer() -> FunctionTransformer:
    return FunctionTransformer(lambda x: x, validate=False)


def build_classifier(name: str, seed: int) -> Pipeline:
    if name == "lda_shrinkage":
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    if name == "logreg_c1":
        estimator = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    if name == "logreg_c03":
        estimator = LogisticRegression(
            C=0.3,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    if name == "ridge":
        estimator = RidgeClassifier(alpha=1.0, class_weight="balanced")
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    if name == "svm_linear":
        estimator = SVC(C=1.0, kernel="linear", class_weight="balanced", random_state=seed)
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    if name == "svm_rbf_c3":
        estimator = SVC(C=3.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed)
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    if name == "svm_rbf_c10":
        estimator = SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed)
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    if name == "pca95_svm_rbf_c10":
        estimator = SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, svd_solver="full")),
                ("clf", estimator),
            ]
        )
    if name == "mlp_256_64":
        estimator = MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            alpha=1e-3,
            batch_size=64,
            learning_rate_init=1e-3,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=30,
            max_iter=500,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    if name == "extratrees":
        estimator = ExtraTreesClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([("identity", identity_transformer()), ("clf", estimator)])
    if name == "randomforest":
        estimator = RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([("identity", identity_transformer()), ("clf", estimator)])
    raise ValueError(f"Unknown classifier: {name}")


def build_combos(datasets: list[str]) -> list[Combo]:
    combos = [
        Combo(
            "BCIC2A",
            "bcic_fbcsp_narrow_csp2_lda",
            "fbcsp",
            "lda_shrinkage",
            {"bands": BCIC_NARROW_BANDS, "components_per_side": 2},
        ),
        Combo(
            "BCIC2A",
            "bcic_fbcsp_narrow_csp2_logreg",
            "fbcsp",
            "logreg_c1",
            {"bands": BCIC_NARROW_BANDS, "components_per_side": 2},
        ),
        Combo(
            "BCIC2A",
            "bcic_fbcsp_narrow_csp2_svm_linear",
            "fbcsp",
            "svm_linear",
            {"bands": BCIC_NARROW_BANDS, "components_per_side": 2},
        ),
        Combo(
            "BCIC2A",
            "bcic_fbcsp_narrow_csp2_svm_rbf",
            "fbcsp",
            "svm_rbf_c10",
            {"bands": BCIC_NARROW_BANDS, "components_per_side": 2},
        ),
        Combo(
            "BCIC2A",
            "bcic_fbcsp_wide_csp3_lda",
            "fbcsp",
            "lda_shrinkage",
            {"bands": BCIC_WIDE_BANDS, "components_per_side": 3},
        ),
        Combo(
            "BCIC2A",
            "bcic_csp_mu_beta_csp3_lda",
            "fbcsp",
            "lda_shrinkage",
            {"bands": BCIC_MU_BETA_BANDS, "components_per_side": 3},
        ),
        Combo(
            "BCIC2A",
            "bcic_fbcsp_dense_csp1_ridge",
            "fbcsp",
            "ridge",
            {"bands": BCIC_DENSE_BANDS, "components_per_side": 1},
        ),
        Combo(
            "BCIC2A",
            "bcic_fbcsp_wide_csp2_extratrees",
            "fbcsp",
            "extratrees",
            {"bands": BCIC_WIDE_BANDS, "components_per_side": 2},
        ),
        Combo(
            "BCIC2A",
            "bcic_logpower_svm_rbf",
            "logpower",
            "svm_rbf_c10",
            {"bands": BCIC_LOGPOWER_BANDS},
        ),
        Combo(
            "BCIC2A",
            "bcic_logpower_mlp",
            "logpower",
            "mlp_256_64",
            {"bands": BCIC_LOGPOWER_BANDS},
        ),
        Combo(
            "SEED",
            "seed_de5_logreg",
            "seed_spectral",
            "logreg_c1",
            {"include_asymmetry": False, "include_hjorth": False, "include_stats": False},
        ),
        Combo(
            "SEED",
            "seed_de5_logreg_c03",
            "seed_spectral",
            "logreg_c03",
            {"include_asymmetry": False, "include_hjorth": False, "include_stats": False},
        ),
        Combo(
            "SEED",
            "seed_de5_svm_rbf_c3",
            "seed_spectral",
            "svm_rbf_c3",
            {"include_asymmetry": False, "include_hjorth": False, "include_stats": False},
        ),
        Combo(
            "SEED",
            "seed_de5_svm_rbf_c10",
            "seed_spectral",
            "svm_rbf_c10",
            {"include_asymmetry": False, "include_hjorth": False, "include_stats": False},
        ),
        Combo(
            "SEED",
            "seed_de5_asym_logreg",
            "seed_spectral",
            "logreg_c1",
            {"include_asymmetry": True, "include_hjorth": False, "include_stats": False},
        ),
        Combo(
            "SEED",
            "seed_de5_asym_svm_rbf",
            "seed_spectral",
            "svm_rbf_c10",
            {"include_asymmetry": True, "include_hjorth": False, "include_stats": False},
        ),
        Combo(
            "SEED",
            "seed_de5_hjorth_logreg",
            "seed_spectral",
            "logreg_c1",
            {"include_asymmetry": False, "include_hjorth": True, "include_stats": False},
        ),
        Combo(
            "SEED",
            "seed_de5_hjorth_svm_rbf",
            "seed_spectral",
            "svm_rbf_c10",
            {"include_asymmetry": False, "include_hjorth": True, "include_stats": False},
        ),
        Combo(
            "SEED",
            "seed_de5_asym_hjorth_pca_svm",
            "seed_spectral",
            "pca95_svm_rbf_c10",
            {"include_asymmetry": True, "include_hjorth": True, "include_stats": False},
        ),
        Combo(
            "SEED",
            "seed_de5_asym_hjorth_mlp",
            "seed_spectral",
            "mlp_256_64",
            {"include_asymmetry": True, "include_hjorth": True, "include_stats": False},
        ),
        Combo(
            "SEED",
            "seed_de5_asym_stats_extratrees",
            "seed_spectral",
            "extratrees",
            {"include_asymmetry": True, "include_hjorth": False, "include_stats": True},
        ),
        Combo(
            "SEED",
            "seed_de5_asym_stats_randomforest",
            "seed_spectral",
            "randomforest",
            {"include_asymmetry": True, "include_hjorth": False, "include_stats": True},
        ),
    ]
    return [combo for combo in combos if combo.dataset in set(datasets)]


def make_features(
    combo: Combo,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_xs: list[np.ndarray],
    meta: DatasetMeta,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    if combo.feature_family == "fbcsp":
        return fbcsp_features(
            train_x,
            train_y,
            eval_xs,
            meta,
            bands=combo.params["bands"],
            components_per_side=int(combo.params["components_per_side"]),
        )
    if combo.feature_family == "logpower":
        return logpower_features(train_x, eval_xs, meta, bands=combo.params["bands"])
    if combo.feature_family == "seed_spectral":
        return seed_spectral_features(
            train_x,
            eval_xs,
            meta,
            include_asymmetry=bool(combo.params["include_asymmetry"]),
            include_hjorth=bool(combo.params["include_hjorth"]),
            include_stats=bool(combo.params["include_stats"]),
        )
    raise ValueError(f"Unknown feature family: {combo.feature_family}")


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
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


def evaluate_combo(
    combo: Combo,
    meta: DatasetMeta,
    arrays: dict[str, tuple[np.ndarray, np.ndarray | None]],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    train_x, train_y = arrays["train"]
    val_x, val_y = arrays["val"]
    if train_y is None or val_y is None:
        raise ValueError(f"{combo.dataset}: train/val labels are required")

    feature_train, (feature_val,), feature_details = make_features(
        combo,
        train_x,
        train_y,
        [val_x],
        meta,
    )
    clf = build_classifier(combo.classifier, seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(feature_train, train_y)
    val_pred = clf.predict(feature_val).astype(np.int64)

    row: dict[str, Any] = {
        "dataset": combo.dataset,
        "combo": combo.name,
        "feature_family": combo.feature_family,
        "classifier": combo.classifier,
        "feature_dim": int(feature_train.shape[1]),
        "params": json.dumps(combo.params),
        "feature_details": json.dumps(feature_details),
    }
    row.update(metric_dict(val_y, val_pred, meta.num_classes))
    return row, val_pred


def refit_and_predict_test(
    combo: Combo,
    meta: DatasetMeta,
    arrays: dict[str, tuple[np.ndarray, np.ndarray | None]],
    seed: int,
) -> np.ndarray:
    train_x, train_y = arrays["train"]
    val_x, val_y = arrays["val"]
    test_x, _ = arrays["test"]
    if train_y is None or val_y is None:
        raise ValueError(f"{combo.dataset}: train/val labels are required")
    trainval_x = np.concatenate([train_x, val_x], axis=0)
    trainval_y = np.concatenate([train_y, val_y], axis=0)
    feature_trainval, (feature_test,), _ = make_features(
        combo,
        trainval_x,
        trainval_y,
        [test_x],
        meta,
    )
    clf = build_classifier(combo.classifier, seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(feature_trainval, trainval_y)
    return clf.predict(feature_test).astype(np.int64)


def write_predictions(path: Path, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in pred.astype(int).tolist():
            f.write(f"{value}\n")


def validate_prediction_file(path: Path, expected_rows: int, num_classes: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {len(lines)}")
    values = []
    for line in lines:
        if line.strip() != line or not line.strip().isdigit():
            raise ValueError(f"{path}: invalid line {line!r}")
        values.append(int(line))
    invalid = sorted(set(values) - set(range(num_classes)))
    if invalid:
        raise ValueError(f"{path}: invalid labels {invalid}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
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


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def load_existing_baselines(results_dir: Path, datasets: list[str]) -> dict[str, dict[str, Any]]:
    best = {
        name: {"balanced_accuracy": None, "accuracy": None, "source": None}
        for name in datasets
    }
    if not results_dir.exists():
        return best
    for path in sorted(results_dir.glob("*.csv")):
        if path.name.startswith("targeted_bcic_seed_sweep"):
            continue
        try:
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        except Exception:
            continue
        for row in rows:
            dataset = row.get("dataset")
            if dataset not in best:
                continue
            raw_bacc = row.get("val_balanced_accuracy") or row.get("balanced_accuracy")
            raw_acc = row.get("val_accuracy") or row.get("accuracy")
            if raw_bacc in (None, ""):
                continue
            try:
                bacc = float(raw_bacc)
                acc = float(raw_acc) if raw_acc not in (None, "") else bacc
            except ValueError:
                continue
            current = best[dataset]["balanced_accuracy"]
            if current is None or bacc > float(current):
                best[dataset] = {
                    "balanced_accuracy": bacc,
                    "accuracy": acc,
                    "source": str(path),
                }
    return best


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    combos = build_combos(args.datasets)
    if len(combos) < 10:
        raise ValueError(f"Expected at least 10 combos, got {len(combos)}")

    metas = {name: load_meta(args.data_root, name) for name in args.datasets}
    arrays = {name: load_arrays(args.data_root, name) for name in args.datasets}
    baselines = load_existing_baselines(args.results_dir, args.datasets)

    metrics_path = args.results_dir / "targeted_bcic_seed_sweep_metrics.csv"
    selection_path = args.results_dir / "targeted_bcic_seed_sweep_selection_summary.csv"
    preserved_metric_rows = [
        row for row in read_csv_rows(metrics_path) if row.get("dataset") not in set(args.datasets)
    ]
    preserved_selection_rows = [
        row for row in read_csv_rows(selection_path) if row.get("dataset") not in set(args.datasets)
    ]

    metrics_rows: list[dict[str, Any]] = []
    combo_by_name = {combo.name: combo for combo in combos}
    for index, combo in enumerate(combos, start=1):
        print(f"[{index:02d}/{len(combos)}] {combo.name}")
        row, _ = evaluate_combo(combo, metas[combo.dataset], arrays[combo.dataset], args.seed)
        baseline = baselines.get(combo.dataset, {})
        baseline_bacc = baseline.get("balanced_accuracy")
        row["baseline_best_balanced_accuracy"] = "" if baseline_bacc is None else baseline_bacc
        row["delta_vs_baseline_bacc"] = (
            "" if baseline_bacc is None else float(row["balanced_accuracy"]) - float(baseline_bacc)
        )
        row["baseline_source"] = baseline.get("source") or ""
        metrics_rows.append(row)
        print(
            f"  acc={row['accuracy']:.4f} bacc={row['balanced_accuracy']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} dim={row['feature_dim']}"
        )

    selection_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        dataset_rows = [row for row in metrics_rows if row["dataset"] == dataset]
        best_row = max(dataset_rows, key=lambda row: (float(row["balanced_accuracy"]), float(row["accuracy"])))
        best_combo = combo_by_name[str(best_row["combo"])]
        test_pred = refit_and_predict_test(best_combo, metas[dataset], arrays[dataset], args.seed)
        combo_output = args.outputs_dir / f"{dataset}_{best_combo.name}.txt"
        canonical_output = args.outputs_dir / f"{dataset}.txt"
        write_predictions(combo_output, test_pred)
        write_predictions(canonical_output, test_pred)
        validate_prediction_file(canonical_output, EXPECTED_TEST_ROWS[dataset], metas[dataset].num_classes)
        if args.update_submission:
            submission_path = args.submission_dir / f"{dataset}.txt"
            write_predictions(submission_path, test_pred)
            validate_prediction_file(submission_path, EXPECTED_TEST_ROWS[dataset], metas[dataset].num_classes)
        selection = dict(best_row)
        selection["selected_output"] = str(canonical_output)
        selection["selected_combo_output"] = str(combo_output)
        selection["updated_submission"] = (
            str(args.submission_dir / f"{dataset}.txt") if args.update_submission else ""
        )
        selection_rows.append(selection)
        print(
            f"SELECTED {dataset}: {best_combo.name} "
            f"bacc={best_row['balanced_accuracy']:.4f} output={canonical_output}"
        )

    combined_metric_rows = preserved_metric_rows + metrics_rows
    combined_selection_rows = preserved_selection_rows + selection_rows

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metrics_path, combined_metric_rows)
    write_csv(selection_path, combined_selection_rows)
    (args.results_dir / "targeted_bcic_seed_sweep_metrics.json").write_text(
        json.dumps(combined_metric_rows, indent=2),
        encoding="utf-8",
    )
    (args.results_dir / "targeted_bcic_seed_sweep_selection_summary.json").write_text(
        json.dumps(combined_selection_rows, indent=2),
        encoding="utf-8",
    )
    summary = {
        "num_combos": len(combos),
        "num_preserved_metric_rows": len(preserved_metric_rows),
        "datasets": args.datasets,
        "metrics_path": str(metrics_path),
        "selection_path": str(selection_path),
        "best": selection_rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
