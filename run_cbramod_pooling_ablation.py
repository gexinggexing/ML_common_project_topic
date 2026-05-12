#!/usr/bin/env python3
"""Run CBraMod pooling ablations for all EEG course datasets."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path("/mnt/dataset3/panxy/course/project1_data/course project/course project")
CBRAMOD_ROOT = Path("/mnt/dataset4/yinuo/personlized_FM/CBraMod-main/CBraMod-main")
CBRAMOD_CHECKPOINT = CBRAMOD_ROOT / "pretrained_weights/pretrained_weights.pth"
DATASETS = ["BCIC2A", "CHINESE", "MDD", "SEED", "SLEEP"]
POOLINGS = [
    "global_mean",
    "global_mean_std",
    "channel_flat_pca",
    "channel_mean_std_pca",
]
PCA_DIMS = [64, 128, 256, 512]
EXPECTED_TEST_ROWS = {
    "BCIC2A": 360,
    "CHINESE": 200,
    "MDD": 800,
    "SEED": 450,
    "SLEEP": 1945,
}
EXPECTED_CHANNEL_FLAT_DIMS = {
    "BCIC2A": 4400,
    "CHINESE": 4400,
    "MDD": 4000,
    "SEED": 12400,
    "SLEEP": 1200,
}


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    num_classes: int
    category_list: list[str]
    n_channels: int
    n_times: int
    patch_count: int


class SmallMLP(torch.nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clip-z", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)


def dataset_dir(name: str) -> Path:
    return DATA_ROOT / name


def h5_path(name: str, split: str) -> Path:
    filename = "test_x_only.h5" if split == "test" else f"{split}.h5"
    return dataset_dir(name) / filename


def load_meta(name: str) -> DatasetMeta:
    root = dataset_dir(name)
    info_path = root / "dataset_info_fixed.json"
    if not info_path.exists():
        info_path = root / "dataset_info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    with h5py.File(h5_path(name, "train"), "r") as h5:
        _, n_channels, n_times = h5["X"].shape

    if n_times % 200 != 0:
        raise ValueError(f"{name}: n_times={n_times} is not divisible by 200")

    ds_info = info["dataset"]
    return DatasetMeta(
        name=name,
        num_classes=int(ds_info["num_labels"]),
        category_list=list(ds_info["category_list"]),
        n_channels=int(n_channels),
        n_times=int(n_times),
        patch_count=int(n_times // 200),
    )


def load_cbramod(device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(CBRAMOD_ROOT))
    from models.cbramod import CBraMod  # type: ignore

    model = CBraMod().to(device)
    state = torch.load(CBRAMOD_CHECKPOINT, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"CBraMod weight mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def normalize_and_patch(x: np.ndarray, device: torch.device, clip_z: float) -> torch.Tensor:
    tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    tensor = (tensor - tensor.mean(dim=-1, keepdim=True)) / tensor.std(
        dim=-1, keepdim=True
    ).clamp_min(1e-4)
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=clip_z, neginf=-clip_z)
    if clip_z > 0:
        tensor = tensor.clamp(min=-clip_z, max=clip_z)
    batch, channels, n_times = tensor.shape
    if n_times % 200 != 0:
        raise ValueError(f"n_times={n_times} is not divisible by 200")
    return tensor.reshape(batch, channels, n_times // 200, 200)


def pool_raw(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    mean_cp = raw.mean(dim=(1, 2))
    std_cp = raw.std(dim=(1, 2), unbiased=False)
    mean_p = raw.mean(dim=2)
    std_p = raw.std(dim=2, unbiased=False)
    return {
        "global_mean": mean_cp,
        "global_mean_std": torch.cat([mean_cp, std_cp], dim=1),
        "channel_flat_pca": mean_p.flatten(start_dim=1),
        "channel_mean_std_pca": torch.cat([mean_p, std_p], dim=2).flatten(start_dim=1),
    }


@torch.no_grad()
def smoke_poolings(
    model: torch.nn.Module,
    name: str,
    device: torch.device,
    clip_z: float,
) -> dict[str, tuple[int, int]]:
    with h5py.File(h5_path(name, "train"), "r") as h5:
        batch = normalize_and_patch(np.asarray(h5["X"][:2]), device, clip_z)
    pooled = pool_raw(model(batch))
    shapes: dict[str, tuple[int, int]] = {}
    for pooling, tensor in pooled.items():
        if tensor.ndim != 2 or tensor.shape[0] != 2:
            raise ValueError(f"{name} {pooling}: bad smoke shape {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} {pooling}: non-finite smoke feature")
        shapes[pooling] = (int(tensor.shape[0]), int(tensor.shape[1]))
    return shapes


@torch.no_grad()
def extract_split_features(
    model: torch.nn.Module,
    meta: DatasetMeta,
    split: str,
    device: torch.device,
    batch_size: int,
    clip_z: float,
    out_dir: Path,
    overwrite: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    paths = {pooling: out_dir / f"cbramod_{pooling}_{split}.npz" for pooling in POOLINGS}
    if all(path.exists() for path in paths.values()) and not overwrite:
        features = {
            pooling: np.load(path, allow_pickle=False)["features"].astype(np.float32)
            for pooling, path in paths.items()
        }
        label_path = paths["global_mean"]
        loaded = np.load(label_path, allow_pickle=False)
        labels = loaded["labels"].astype(np.int64) if "labels" in loaded.files else None
        return features, labels

    source = h5_path(meta.name, split)
    chunks: dict[str, list[np.ndarray]] = {pooling: [] for pooling in POOLINGS}
    labels: np.ndarray | None = None

    with h5py.File(source, "r") as h5:
        x_ds = h5["X"]
        if "y" in h5:
            labels = np.asarray(h5["y"], dtype=np.int64)

        for start in range(0, x_ds.shape[0], batch_size):
            stop = min(start + batch_size, x_ds.shape[0])
            batch = normalize_and_patch(np.asarray(x_ds[start:stop]), device, clip_z)
            pooled = pool_raw(model(batch))
            for pooling, tensor in pooled.items():
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"{meta.name} {split} {pooling}: non-finite feature")
                chunks[pooling].append(tensor.detach().cpu().numpy().astype(np.float32))

    features = {pooling: np.concatenate(parts, axis=0) for pooling, parts in chunks.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    for pooling, feature_array in features.items():
        save_kwargs: dict[str, Any] = {
            "features": feature_array,
            "split": np.asarray(split),
            "dataset": np.asarray(meta.name),
            "encoder": np.asarray("cbramod"),
            "pooling": np.asarray(pooling),
            "source_h5": np.asarray(str(source)),
            "checkpoint_path": np.asarray(str(CBRAMOD_CHECKPOINT)),
            "raw_feature_dim": np.asarray(feature_array.shape[1]),
        }
        if labels is not None:
            save_kwargs["labels"] = labels
        np.savez_compressed(paths[pooling], **save_kwargs)
    return features, labels


def build_transform(pca_dim: int | None, seed: int) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if pca_dim is not None:
        steps.append(
            (
                "pca",
                PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed),
            )
        )
    return Pipeline(steps)


def candidate_pca_dims(pooling: str, train_x: np.ndarray) -> list[int | None]:
    if not pooling.endswith("_pca"):
        return [None]
    limit = min(train_x.shape[0] - 1, train_x.shape[1])
    return [dim for dim in PCA_DIMS if dim <= limit]


def metric_row(
    dataset: str,
    pooling: str,
    pca_dim: int | None,
    classifier: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    figure_path: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "pooling": pooling,
        "pca_dim": "" if pca_dim is None else int(pca_dim),
        "classifier": classifier,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": json.dumps(
            confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
        ),
        "figure_path": str(figure_path),
    }
    if extra:
        row.update(extra)
    return row


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def plot_confusion(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(max(4.8, len(labels) * 1.2), max(4.0, len(labels) * 1.0)))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fit_logreg(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    pca_dim: int | None,
    seed: int,
) -> tuple[np.ndarray, Any, Pipeline]:
    transform = build_transform(pca_dim, seed)
    x_train = transform.fit_transform(train_x)
    x_val = transform.transform(val_x)
    clf = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed)
    clf.fit(x_train, train_y)
    return clf.predict(x_val).astype(np.int64), clf, transform


def train_mlp_with_validation(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    num_classes: int,
    pca_dim: int | None,
    device: torch.device,
    seed: int,
    max_epochs: int = 200,
    patience: int = 20,
) -> tuple[np.ndarray, dict[str, Any]]:
    transform = build_transform(pca_dim, seed)
    x_train = transform.fit_transform(train_x).astype(np.float32)
    x_val = transform.transform(val_x).astype(np.float32)

    set_seed(seed)
    model = SmallMLP(x_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    train_label = torch.as_tensor(train_y, dtype=torch.long, device=device)
    val_tensor = torch.as_tensor(x_val, dtype=torch.float32, device=device)

    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(train_tensor.shape[0], device=device)
        for start in range(0, train_tensor.shape[0], 128):
            idx = permutation[start : start + 128]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_tensor[idx]), train_label[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(val_tensor).argmax(dim=1).detach().cpu().numpy()
        score = float(balanced_accuracy_score(val_y, pred))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("MLP failed to produce a validation checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(val_tensor).argmax(dim=1).detach().cpu().numpy().astype(np.int64)
    return pred, {
        "transform": transform,
        "best_epoch": int(best_epoch),
        "best_val_balanced_accuracy": float(best_score),
    }


def fit_mlp_fixed_epochs(
    train_x: np.ndarray,
    train_y: np.ndarray,
    num_classes: int,
    pca_dim: int | None,
    device: torch.device,
    seed: int,
    epochs: int,
) -> tuple[SmallMLP, Pipeline]:
    transform = build_transform(pca_dim, seed)
    x_train = transform.fit_transform(train_x).astype(np.float32)

    set_seed(seed)
    model = SmallMLP(x_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    train_label = torch.as_tensor(train_y, dtype=torch.long, device=device)

    for _ in range(max(1, epochs)):
        model.train()
        permutation = torch.randperm(train_tensor.shape[0], device=device)
        for start in range(0, train_tensor.shape[0], 128):
            idx = permutation[start : start + 128]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_tensor[idx]), train_label[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    return model, transform


@torch.no_grad()
def predict_mlp(model: SmallMLP, transform: Pipeline, x: np.ndarray, device: torch.device) -> np.ndarray:
    x_scaled = transform.transform(x).astype(np.float32)
    tensor = torch.as_tensor(x_scaled, dtype=torch.float32, device=device)
    return model(tensor).argmax(dim=1).detach().cpu().numpy().astype(np.int64)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    artifacts_root = Path("artifacts")
    outputs_root = Path("outputs")
    figures_root = artifacts_root / "figures" / "cbramod_pooling_ablation"
    results_dir = artifacts_root / "results"
    metrics_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    print(f"device={device}")
    model = load_cbramod(device)
    print(f"Loaded CBraMod checkpoint: {CBRAMOD_CHECKPOINT}")

    for name in DATASETS:
        meta = load_meta(name)
        print(f"\n== {name} ==")
        print(
            f"classes={meta.num_classes} labels={meta.category_list} "
            f"shape=({meta.n_channels},{meta.n_times}) patches={meta.patch_count}"
        )

        smoke_shapes = smoke_poolings(model, name, device, args.clip_z)
        print(f"smoke_shapes={smoke_shapes}")
        expected_flat_dim = EXPECTED_CHANNEL_FLAT_DIMS[name]
        if smoke_shapes["channel_flat_pca"][1] != expected_flat_dim:
            raise ValueError(
                f"{name}: channel_flat_pca dim {smoke_shapes['channel_flat_pca'][1]}, "
                f"expected {expected_flat_dim}"
            )

        emb_dir = artifacts_root / "embeddings" / name
        train_features, train_y = extract_split_features(
            model, meta, "train", device, args.batch_size, args.clip_z, emb_dir, args.overwrite
        )
        val_features, val_y = extract_split_features(
            model, meta, "val", device, args.batch_size, args.clip_z, emb_dir, args.overwrite
        )
        test_features, _ = extract_split_features(
            model, meta, "test", device, args.batch_size, args.clip_z, emb_dir, args.overwrite
        )

        if train_y is None or val_y is None:
            raise ValueError(f"{name}: train/val labels are required")

        dataset_rows: list[dict[str, Any]] = []
        for pooling in POOLINGS:
            train_x = train_features[pooling]
            val_x = val_features[pooling]
            pca_dims = candidate_pca_dims(pooling, train_x)
            if not pca_dims:
                raise ValueError(f"{name} {pooling}: no valid PCA dims")
            print(
                f"{pooling}: train={train_x.shape} val={val_x.shape} "
                f"pca_dims={pca_dims}"
            )

            for pca_dim in pca_dims:
                pca_tag = "nopca" if pca_dim is None else f"pca{pca_dim}"

                logreg_val_pred, _, _ = fit_logreg(train_x, train_y, val_x, pca_dim, args.seed)
                logreg_figure = figures_root / (
                    f"{name}_{pooling}_{pca_tag}_logreg_balanced.png"
                )
                plot_confusion(
                    logreg_figure,
                    val_y,
                    logreg_val_pred,
                    meta.category_list,
                    f"{name} {pooling} {pca_tag} logreg",
                )
                logreg_row = metric_row(
                    name,
                    pooling,
                    pca_dim,
                    "logreg_balanced",
                    val_y,
                    logreg_val_pred,
                    meta.num_classes,
                    logreg_figure,
                    {"feature_dim": train_x.shape[1]},
                )
                metrics_rows.append(logreg_row)
                dataset_rows.append(logreg_row)

                mlp_val_pred, mlp_info = train_mlp_with_validation(
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    meta.num_classes,
                    pca_dim,
                    device,
                    args.seed,
                )
                mlp_figure = figures_root / f"{name}_{pooling}_{pca_tag}_mlp_small.png"
                plot_confusion(
                    mlp_figure,
                    val_y,
                    mlp_val_pred,
                    meta.category_list,
                    f"{name} {pooling} {pca_tag} mlp",
                )
                mlp_row = metric_row(
                    name,
                    pooling,
                    pca_dim,
                    "mlp_small",
                    val_y,
                    mlp_val_pred,
                    meta.num_classes,
                    mlp_figure,
                    {
                        "feature_dim": train_x.shape[1],
                        "best_epoch": mlp_info["best_epoch"],
                        "best_val_balanced_accuracy": mlp_info[
                            "best_val_balanced_accuracy"
                        ],
                    },
                )
                metrics_rows.append(mlp_row)
                dataset_rows.append(mlp_row)

                print(
                    f"{pooling} {pca_tag}: "
                    f"logreg={logreg_row['balanced_accuracy']:.4f} "
                    f"mlp={mlp_row['balanced_accuracy']:.4f}"
                )

        best = max(dataset_rows, key=lambda row: row["balanced_accuracy"])
        best_pooling = str(best["pooling"])
        best_pca_dim = None if best["pca_dim"] == "" else int(best["pca_dim"])
        best_classifier = str(best["classifier"])

        trainval_x = np.concatenate(
            [train_features[best_pooling], val_features[best_pooling]], axis=0
        )
        trainval_y = np.concatenate([train_y, val_y], axis=0)
        test_x = test_features[best_pooling]

        if best_classifier == "logreg_balanced":
            transform = build_transform(best_pca_dim, args.seed)
            x_trainval = transform.fit_transform(trainval_x)
            x_test = transform.transform(test_x)
            clf = LogisticRegression(
                class_weight="balanced", max_iter=5000, random_state=args.seed
            )
            clf.fit(x_trainval, trainval_y)
            test_pred = clf.predict(x_test).astype(np.int64)
        else:
            epochs = int(best.get("best_epoch") or 1)
            mlp_model, mlp_transform = fit_mlp_fixed_epochs(
                trainval_x,
                trainval_y,
                meta.num_classes,
                best_pca_dim,
                device,
                args.seed,
                epochs,
            )
            test_pred = predict_mlp(mlp_model, mlp_transform, test_x, device)

        prediction_path = outputs_root / "cbramod_pooling" / f"{name}.txt"
        write_predictions(prediction_path, test_pred)
        validate_prediction_file(prediction_path, EXPECTED_TEST_ROWS[name], meta.num_classes)

        selection_rows.append(
            {
                "dataset": name,
                "best_pooling": best_pooling,
                "best_pca_dim": "" if best_pca_dim is None else best_pca_dim,
                "best_classifier": best_classifier,
                "val_balanced_accuracy": best["balanced_accuracy"],
                "val_accuracy": best["accuracy"],
                "val_macro_f1": best["macro_f1"],
                "baseline_global_mean_mlp_bacc": 0.34444444444444444
                if name == "BCIC2A"
                else 0.615
                if name == "CHINESE"
                else 0.9328125
                if name == "MDD"
                else 0.35333333333333333
                if name == "SEED"
                else 0.7057063761724995,
                "num_classes": meta.num_classes,
                "category_list": json.dumps(meta.category_list),
                "prediction_path": str(prediction_path),
                "figure_path": best["figure_path"],
            }
        )

    write_csv(results_dir / "cbramod_pooling_ablation_metrics.csv", metrics_rows)
    write_csv(results_dir / "cbramod_pooling_selection_summary.csv", selection_rows)
    (results_dir / "cbramod_pooling_ablation_metrics.json").write_text(
        json.dumps(metrics_rows, indent=2), encoding="utf-8"
    )
    (results_dir / "cbramod_pooling_selection_summary.json").write_text(
        json.dumps(selection_rows, indent=2), encoding="utf-8"
    )

    print("\n== pooling selection ==")
    for row in selection_rows:
        print(
            f"{row['dataset']}: {row['best_pooling']} "
            f"pca={row['best_pca_dim'] or 'none'} "
            f"{row['best_classifier']} "
            f"bacc={float(row['val_balanced_accuracy']):.4f} "
            f"prediction={row['prediction_path']}"
        )


if __name__ == "__main__":
    main()
