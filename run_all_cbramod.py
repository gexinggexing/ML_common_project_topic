#!/usr/bin/env python3
"""Extract frozen CBraMod embeddings for all course datasets and train probes."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path("/mnt/dataset3/panxy/course/project1_data/course project/course project")
CBRAMOD_ROOT = Path("/mnt/dataset4/yinuo/personlized_FM/CBraMod-main/CBraMod-main")
CBRAMOD_CHECKPOINT = CBRAMOD_ROOT / "pretrained_weights/pretrained_weights.pth"
DATASETS = ["BCIC2A", "CHINESE", "MDD", "SEED", "SLEEP"]
EXPECTED_TEST_ROWS = {
    "BCIC2A": 360,
    "CHINESE": 200,
    "MDD": 800,
    "SEED": 450,
    "SLEEP": 1945,
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


def normalize_and_patch(x: np.ndarray, device: torch.device, clip_z: float = 10.0) -> torch.Tensor:
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


@torch.no_grad()
def smoke_embedding(
    model: torch.nn.Module,
    name: str,
    device: torch.device,
    clip_z: float,
) -> np.ndarray:
    with h5py.File(h5_path(name, "train"), "r") as h5:
        batch = normalize_and_patch(np.asarray(h5["X"][:2]), device, clip_z)
    raw = model(batch)
    emb = raw.mean(dim=(1, 2))
    if not torch.isfinite(emb).all():
        raise ValueError(f"{name}: non-finite smoke embedding")
    return emb.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def extract_split_embeddings(
    model: torch.nn.Module,
    name: str,
    split: str,
    device: torch.device,
    batch_size: int,
    clip_z: float,
    out_dir: Path,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    out_path = out_dir / f"cbramod_{split}.npz"
    if out_path.exists() and not overwrite:
        loaded = np.load(out_path, allow_pickle=False)
        features = loaded["features"].astype(np.float32)
        labels = loaded["labels"].astype(np.int64) if "labels" in loaded.files else None
        return features, labels

    source = h5_path(name, split)
    features: list[np.ndarray] = []
    labels: np.ndarray | None = None

    with h5py.File(source, "r") as h5:
        x_ds = h5["X"]
        if "y" in h5:
            labels = np.asarray(h5["y"], dtype=np.int64)

        for start in range(0, x_ds.shape[0], batch_size):
            stop = min(start + batch_size, x_ds.shape[0])
            batch = normalize_and_patch(np.asarray(x_ds[start:stop]), device, clip_z)
            raw = model(batch)
            emb = raw.mean(dim=(1, 2))
            if not torch.isfinite(emb).all():
                raise ValueError(f"{name} {split}: non-finite CBraMod embedding")
            features.append(emb.detach().cpu().numpy().astype(np.float32))

    feature_array = np.concatenate(features, axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {
        "features": feature_array,
        "split": np.asarray(split),
        "dataset": np.asarray(name),
        "encoder": np.asarray("cbramod"),
        "source_h5": np.asarray(str(source)),
        "checkpoint_path": np.asarray(str(CBRAMOD_CHECKPOINT)),
    }
    if labels is not None:
        save_kwargs["labels"] = labels
    np.savez_compressed(out_path, **save_kwargs)
    return feature_array, labels


def metric_row(
    dataset: str,
    classifier: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "classifier": classifier,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": json.dumps(
            confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
        ),
    }
    if extra:
        row.update(extra)
    return row


def fit_logreg(
    train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray
) -> tuple[np.ndarray, Any]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, random_state=0),
    )
    clf.fit(train_x, train_y)
    return clf.predict(val_x).astype(np.int64), clf


def train_mlp_with_validation(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    num_classes: int,
    device: torch.device,
    seed: int,
    max_epochs: int = 200,
    patience: int = 20,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_x).astype(np.float32)
    x_val = scaler.transform(val_x).astype(np.float32)

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
        "scaler": scaler,
        "best_epoch": int(best_epoch),
        "best_val_balanced_accuracy": float(best_score),
    }


def fit_mlp_fixed_epochs(
    train_x: np.ndarray,
    train_y: np.ndarray,
    num_classes: int,
    device: torch.device,
    seed: int,
    epochs: int,
) -> tuple[SmallMLP, StandardScaler]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_x).astype(np.float32)

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
    return model, scaler


@torch.no_grad()
def predict_mlp(model: SmallMLP, scaler: StandardScaler, x: np.ndarray, device: torch.device) -> np.ndarray:
    x_scaled = scaler.transform(x).astype(np.float32)
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

        emb_dir = artifacts_root / "embeddings" / name
        smoke_x = smoke_embedding(model, name, device, args.clip_z)
        if smoke_x.shape != (2, 200):
            raise ValueError(f"{name}: smoke embedding shape {smoke_x.shape}, expected (2, 200)")

        train_x, train_y = extract_split_embeddings(
            model, name, "train", device, args.batch_size, args.clip_z, emb_dir, args.overwrite
        )
        val_x, val_y = extract_split_embeddings(
            model, name, "val", device, args.batch_size, args.clip_z, emb_dir, args.overwrite
        )
        test_x, _ = extract_split_embeddings(
            model, name, "test", device, args.batch_size, args.clip_z, emb_dir, args.overwrite
        )

        if train_y is None or val_y is None:
            raise ValueError(f"{name}: train/val labels are required")
        expected_shapes = {
            "BCIC2A": ((720, 200), (360, 200), (360, 200)),
            "CHINESE": ((400, 200), (200, 200), (200, 200)),
            "MDD": ((960, 200), (640, 200), (800, 200)),
            "SEED": ((900, 200), (450, 200), (450, 200)),
            "SLEEP": ((3921, 200), (1941, 200), (1945, 200)),
        }[name]
        actual_shapes = (train_x.shape, val_x.shape, test_x.shape)
        if actual_shapes != expected_shapes:
            raise ValueError(f"{name}: shapes {actual_shapes}, expected {expected_shapes}")

        print(f"embeddings train={train_x.shape} val={val_x.shape} test={test_x.shape}")

        logreg_val_pred, logreg_model = fit_logreg(train_x, train_y, val_x)
        logreg_row = metric_row(
            name, "logreg_balanced", val_y, logreg_val_pred, meta.num_classes
        )
        metrics_rows.append(logreg_row)

        mlp_val_pred, mlp_info = train_mlp_with_validation(
            train_x, train_y, val_x, val_y, meta.num_classes, device, args.seed
        )
        mlp_row = metric_row(
            name,
            "mlp_small",
            val_y,
            mlp_val_pred,
            meta.num_classes,
            {
                "best_epoch": mlp_info["best_epoch"],
                "best_val_balanced_accuracy": mlp_info["best_val_balanced_accuracy"],
            },
        )
        metrics_rows.append(mlp_row)

        print(
            "val balanced_accuracy:",
            f"logreg={logreg_row['balanced_accuracy']:.4f}",
            f"mlp={mlp_row['balanced_accuracy']:.4f}",
        )

        trainval_x = np.concatenate([train_x, val_x], axis=0)
        trainval_y = np.concatenate([train_y, val_y], axis=0)

        final_logreg = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=5000, random_state=0),
        )
        final_logreg.fit(trainval_x, trainval_y)
        logreg_test_pred = final_logreg.predict(test_x).astype(np.int64)
        write_predictions(outputs_root / "cbramod" / f"logreg_balanced_{name}.txt", logreg_test_pred)

        final_mlp, final_mlp_scaler = fit_mlp_fixed_epochs(
            trainval_x,
            trainval_y,
            meta.num_classes,
            device,
            args.seed,
            int(mlp_info["best_epoch"]),
        )
        mlp_test_pred = predict_mlp(final_mlp, final_mlp_scaler, test_x, device)
        write_predictions(outputs_root / "cbramod" / f"mlp_small_{name}.txt", mlp_test_pred)

        candidate_rows = [logreg_row, mlp_row]
        best = max(candidate_rows, key=lambda row: row["balanced_accuracy"])
        best_classifier = str(best["classifier"])
        best_pred = logreg_test_pred if best_classifier == "logreg_balanced" else mlp_test_pred
        submission_path = outputs_root / "submission" / f"{name}.txt"
        write_predictions(submission_path, best_pred)
        validate_prediction_file(submission_path, EXPECTED_TEST_ROWS[name], meta.num_classes)

        selection_rows.append(
            {
                "dataset": name,
                "best_classifier": best_classifier,
                "val_balanced_accuracy": best["balanced_accuracy"],
                "val_accuracy": best["accuracy"],
                "val_macro_f1": best["macro_f1"],
                "num_classes": meta.num_classes,
                "category_list": json.dumps(meta.category_list),
                "submission_path": str(submission_path),
            }
        )

    results_dir = artifacts_root / "results"
    write_csv(results_dir / "cbramod_metrics.csv", metrics_rows)
    write_csv(results_dir / "cbramod_selection_summary.csv", selection_rows)
    (results_dir / "cbramod_metrics.json").write_text(
        json.dumps(metrics_rows, indent=2), encoding="utf-8"
    )
    (results_dir / "cbramod_selection_summary.json").write_text(
        json.dumps(selection_rows, indent=2), encoding="utf-8"
    )

    print("\n== selection ==")
    for row in selection_rows:
        print(
            f"{row['dataset']}: {row['best_classifier']} "
            f"bacc={float(row['val_balanced_accuracy']):.4f} "
            f"submission={row['submission_path']}"
        )


if __name__ == "__main__":
    main()
