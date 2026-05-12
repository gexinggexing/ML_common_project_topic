#!/usr/bin/env python3
"""Author-style CBraMod downstream probes for BCIC2A and SEED."""

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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


DATA_ROOT = Path("/mnt/dataset3/panxy/course/project1_data/course project/course project")
CBRAMOD_ROOT = Path("/mnt/dataset4/yinuo/personlized_FM/CBraMod-main/CBraMod-main")
CBRAMOD_CHECKPOINT = CBRAMOD_ROOT / "pretrained_weights/pretrained_weights.pth"
DATASETS = ["BCIC2A", "SEED"]
EXPECTED_TEST_ROWS = {"BCIC2A": 360, "SEED": 450}
DEFAULT_PRIORITY_EXPERIMENTS = [
    ("BCIC2A", "zscore_clip10", "all_patch_reps", "frozen"),
    ("BCIC2A", "zscore_clip10", "all_patch_reps", "last2_unfreeze"),
    ("BCIC2A", "divide100", "all_patch_reps", "frozen"),
    ("BCIC2A", "divide100", "all_patch_reps", "last2_unfreeze"),
    ("SEED", "zscore_clip10", "all_patch_reps", "frozen"),
    ("SEED", "zscore_clip10", "all_patch_reps", "last2_unfreeze"),
    ("SEED", "divide100", "all_patch_reps", "frozen"),
]


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    num_classes: int
    category_list: list[str]
    n_channels: int
    n_times: int
    patch_count: int


@dataclass(frozen=True)
class Experiment:
    dataset: str
    normalization: str
    head: str
    finetune_mode: str

    @property
    def name(self) -> str:
        return f"{self.dataset}_{self.normalization}_{self.head}_{self.finetune_mode}"


class EEGH5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray | None,
        normalization: str,
        clip_z: float,
    ) -> None:
        self.x = x
        self.y = y
        self.normalization = normalization
        self.clip_z = clip_z

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        sample = torch.as_tensor(self.x[index], dtype=torch.float32)
        sample = preprocess_sample(sample, self.normalization, self.clip_z)
        if self.y is None:
            return sample
        return sample, torch.tensor(int(self.y[index]), dtype=torch.long)


class AuthorHead(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        patch_count: int,
        num_classes: int,
        head: str,
        dropout: float,
    ) -> None:
        super().__init__()
        first_hidden = max(4, patch_count) * 200
        if head == "all_patch_reps_onelayer":
            self.net = torch.nn.Linear(in_dim, num_classes)
        elif head == "all_patch_reps_twolayer":
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 200),
                torch.nn.ELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(200, num_classes),
            )
        elif head == "all_patch_reps":
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, first_hidden),
                torch.nn.ELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(first_hidden, 200),
                torch.nn.ELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(200, num_classes),
            )
        else:
            raise ValueError(f"Unknown head: {head}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AuthorCBraModModel(torch.nn.Module):
    def __init__(
        self,
        meta: DatasetMeta,
        head: str,
        finetune_mode: str,
        dropout: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        sys.path.insert(0, str(CBRAMOD_ROOT))
        from models.cbramod import CBraMod  # type: ignore

        self.backbone = CBraMod().to(device)
        state = torch.load(CBRAMOD_CHECKPOINT, map_location=device, weights_only=False)
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"CBraMod weight mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        self.backbone.proj_out = torch.nn.Identity()
        self.finetune_mode = finetune_mode
        self.configure_backbone_trainability(finetune_mode)

        in_dim = meta.n_channels * meta.patch_count * 200
        self.classifier = AuthorHead(
            in_dim=in_dim,
            patch_count=meta.patch_count,
            num_classes=meta.num_classes,
            head=head,
            dropout=dropout,
        )

    def configure_backbone_trainability(self, mode: str) -> None:
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        if mode == "frozen":
            return
        if mode == "last2_unfreeze":
            for layer in self.backbone.encoder.layers[-2:]:
                for param in layer.parameters():
                    param.requires_grad_(True)
            return
        if mode == "full_finetune":
            for param in self.backbone.parameters():
                param.requires_grad_(True)
            return
        raise ValueError(f"Unknown finetune mode: {mode}")

    @property
    def frozen_backbone(self) -> bool:
        return not any(param.requires_grad for param in self.backbone.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen_backbone:
            self.backbone.eval()
            with torch.no_grad():
                feats = self.backbone(x).detach()
        else:
            feats = self.backbone(x)
        feats = feats.flatten(start_dim=1)
        return self.classifier(feats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip-z", type=float, default=10.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--preset",
        default="priority",
        choices=["priority", "full_grid"],
        help="priority runs the high-value experiments from the plan.",
    )
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument(
        "--normalizations",
        nargs="*",
        default=["zscore_clip10", "divide100"],
        choices=["zscore_clip10", "zscore_no_clip", "divide100"],
    )
    parser.add_argument(
        "--heads",
        nargs="*",
        default=["all_patch_reps"],
        choices=["all_patch_reps_onelayer", "all_patch_reps_twolayer", "all_patch_reps"],
    )
    parser.add_argument(
        "--finetune-modes",
        nargs="*",
        default=["frozen", "last2_unfreeze"],
        choices=["frozen", "last2_unfreeze", "full_finetune"],
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)


def h5_path(name: str, split: str) -> Path:
    filename = "test_x_only.h5" if split == "test" else f"{split}.h5"
    return DATA_ROOT / name / filename


def load_meta(name: str) -> DatasetMeta:
    info_path = DATA_ROOT / name / "dataset_info_fixed.json"
    if not info_path.exists():
        info_path = DATA_ROOT / name / "dataset_info.json"
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


def read_h5(name: str, split: str) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(h5_path(name, split), "r") as h5:
        x = np.asarray(h5["X"], dtype=np.float32)
        y = np.asarray(h5["y"], dtype=np.int64) if "y" in h5 else None
    return x, y


def preprocess_sample(sample: torch.Tensor, normalization: str, clip_z: float) -> torch.Tensor:
    sample = torch.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
    if normalization == "divide100":
        sample = sample / 100.0
    elif normalization in {"zscore_clip10", "zscore_no_clip"}:
        sample = (sample - sample.mean(dim=-1, keepdim=True)) / sample.std(
            dim=-1, keepdim=True
        ).clamp_min(1e-4)
        sample = torch.nan_to_num(sample, nan=0.0, posinf=clip_z, neginf=-clip_z)
        if normalization == "zscore_clip10":
            sample = sample.clamp(min=-clip_z, max=clip_z)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    channels, n_times = sample.shape
    return sample.reshape(channels, n_times // 200, 200)


def make_loader(
    x: np.ndarray,
    y: np.ndarray | None,
    normalization: str,
    clip_z: float,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    dataset = EEGH5Dataset(x, y, normalization, clip_z)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    if args.preset == "priority":
        return [Experiment(*parts) for parts in DEFAULT_PRIORITY_EXPERIMENTS]
    return [
        Experiment(dataset, normalization, head, mode)
        for dataset in args.datasets
        for normalization in args.normalizations
        for head in args.heads
        for mode in args.finetune_modes
    ]


def build_optimizer(
    model: AuthorCBraModModel,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    head_lr = args.head_lr
    if head_lr is None:
        head_lr = 0.001 * (args.batch_size / 256.0) ** 0.5
    groups: list[dict[str, Any]] = [
        {"params": model.classifier.parameters(), "lr": head_lr},
    ]
    trainable_backbone = [
        param for param in model.backbone.parameters() if param.requires_grad
    ]
    if trainable_backbone:
        groups.insert(0, {"params": trainable_backbone, "lr": args.backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model.eval()
    truths: list[int] = []
    preds: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist()
            preds.extend(pred)
            truths.extend(y.numpy().astype(int).tolist())
    y_true = np.asarray(truths, dtype=np.int64)
    y_pred = np.asarray(preds, dtype=np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": json.dumps(
            confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
        ),
    }
    return y_true, y_pred, metrics


def train_one(
    exp: Experiment,
    meta: DatasetMeta,
    arrays: dict[str, tuple[np.ndarray, np.ndarray | None]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    train_x, train_y = arrays["train"]
    val_x, val_y = arrays["val"]
    test_x, _ = arrays["test"]
    if train_y is None or val_y is None:
        raise ValueError(f"{exp.dataset}: train/val labels are required")

    train_loader = make_loader(
        train_x, train_y, exp.normalization, args.clip_z, args.batch_size, True
    )
    val_loader = make_loader(
        val_x, val_y, exp.normalization, args.clip_z, args.batch_size, False
    )

    set_seed(args.seed)
    model = AuthorCBraModModel(
        meta=meta,
        head=exp.head,
        finetune_mode=exp.finetune_mode,
        dropout=args.dropout,
        device=device,
    ).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * len(train_loader)),
        eta_min=1e-6,
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))

        _, _, val_metrics = evaluate(model, val_loader, device, meta.num_classes)
        score = float(val_metrics["balanced_accuracy"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            best_metrics["train_loss"] = float(np.mean(losses))
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
        print(
            f"{exp.name} epoch={epoch} loss={np.mean(losses):.4f} "
            f"bacc={score:.4f} kappa={val_metrics['kappa']:.4f}"
        )

    if best_state is None or best_metrics is None:
        raise RuntimeError(f"{exp.name}: no checkpoint selected")
    model.load_state_dict(best_state)

    trainval_x = np.concatenate([train_x, val_x], axis=0)
    trainval_y = np.concatenate([train_y, val_y], axis=0)
    trainval_loader = make_loader(
        trainval_x, trainval_y, exp.normalization, args.clip_z, args.batch_size, True
    )
    test_loader = make_loader(
        test_x, None, exp.normalization, args.clip_z, args.batch_size, False
    )

    set_seed(args.seed)
    final_model = AuthorCBraModModel(
        meta=meta,
        head=exp.head,
        finetune_mode=exp.finetune_mode,
        dropout=args.dropout,
        device=device,
    ).to(device)
    final_optimizer = build_optimizer(final_model, args)
    final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        final_optimizer,
        T_max=max(1, best_epoch * len(trainval_loader)),
        eta_min=1e-6,
    )
    for _ in range(max(1, best_epoch)):
        final_model.train()
        for x, y in trainval_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            final_optimizer.zero_grad(set_to_none=True)
            loss = criterion(final_model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            final_optimizer.step()
            final_scheduler.step()

    test_pred = predict(final_model, test_loader, device)
    row = {
        "dataset": exp.dataset,
        "normalization": exp.normalization,
        "head": exp.head,
        "finetune_mode": exp.finetune_mode,
        "best_epoch": int(best_epoch),
        "num_classes": meta.num_classes,
        "category_list": json.dumps(meta.category_list),
        **best_metrics,
    }
    return row, {"model": model, "val_y": val_y}, test_pred


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds: list[int] = []
    for batch in loader:
        x = batch.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1).detach().cpu().numpy().astype(int).tolist()
        preds.extend(pred)
    return np.asarray(preds, dtype=np.int64)


def plot_confusion(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(max(4.8, len(labels) * 1.2), max(4.0, len(labels))))
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


def write_predictions(path: Path, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in pred.astype(int).tolist():
            f.write(f"{value}\n")


def validate_prediction_file(path: Path, expected_rows: int, num_classes: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {len(lines)}")
    values = [int(line) for line in lines]
    invalid = sorted(set(values) - set(range(num_classes)))
    if invalid:
        raise ValueError(f"{path}: invalid labels {invalid}")


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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    print(f"device={device}")

    experiments = build_experiments(args)
    metas = {name: load_meta(name) for name in sorted({exp.dataset for exp in experiments})}
    arrays = {
        name: {
            "train": read_h5(name, "train"),
            "val": read_h5(name, "val"),
            "test": read_h5(name, "test"),
        }
        for name in metas
    }

    results_dir = Path("artifacts/results")
    figures_dir = Path("artifacts/figures/cbramod_author_downstream")
    outputs_dir = Path("outputs/cbramod_author_downstream")
    metrics_rows: list[dict[str, Any]] = []
    prediction_by_exp: dict[str, np.ndarray] = {}

    for exp in experiments:
        meta = metas[exp.dataset]
        print(f"\n== {exp.name} ==")
        row, val_info, test_pred = train_one(exp, meta, arrays[exp.dataset], args, device)

        val_loader = make_loader(
            arrays[exp.dataset]["val"][0],
            arrays[exp.dataset]["val"][1],
            exp.normalization,
            args.clip_z,
            args.batch_size,
            False,
        )
        _, val_pred, _ = evaluate(val_info["model"], val_loader, device, meta.num_classes)
        figure_path = figures_dir / f"{exp.name}.png"
        plot_confusion(figure_path, val_info["val_y"], val_pred, meta.category_list, exp.name)
        row["figure_path"] = str(figure_path)
        metrics_rows.append(row)
        prediction_by_exp[exp.name] = test_pred
        write_predictions(outputs_dir / f"{exp.name}.txt", test_pred)
        validate_prediction_file(outputs_dir / f"{exp.name}.txt", EXPECTED_TEST_ROWS[exp.dataset], meta.num_classes)
        print(
            f"selected epoch={row['best_epoch']} bacc={row['balanced_accuracy']:.4f} "
            f"kappa={row['kappa']:.4f} f1={row['macro_f1']:.4f}"
        )

    selection_rows: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in metrics_rows}):
        candidates = [row for row in metrics_rows if row["dataset"] == dataset]
        best = max(candidates, key=lambda row: row["balanced_accuracy"])
        exp_name = (
            f"{best['dataset']}_{best['normalization']}_{best['head']}_{best['finetune_mode']}"
        )
        final_path = outputs_dir / f"{dataset}.txt"
        write_predictions(final_path, prediction_by_exp[exp_name])
        validate_prediction_file(final_path, EXPECTED_TEST_ROWS[dataset], metas[dataset].num_classes)
        selection_rows.append(
            {
                "dataset": dataset,
                "best_experiment": exp_name,
                "normalization": best["normalization"],
                "head": best["head"],
                "finetune_mode": best["finetune_mode"],
                "best_epoch": best["best_epoch"],
                "val_balanced_accuracy": best["balanced_accuracy"],
                "val_accuracy": best["accuracy"],
                "val_macro_f1": best["macro_f1"],
                "val_kappa": best["kappa"],
                "prediction_path": str(final_path),
                "figure_path": best["figure_path"],
            }
        )

    write_csv(results_dir / "cbramod_author_downstream_metrics.csv", metrics_rows)
    write_csv(results_dir / "cbramod_author_downstream_selection_summary.csv", selection_rows)
    (results_dir / "cbramod_author_downstream_metrics.json").write_text(
        json.dumps(metrics_rows, indent=2), encoding="utf-8"
    )
    (results_dir / "cbramod_author_downstream_selection_summary.json").write_text(
        json.dumps(selection_rows, indent=2), encoding="utf-8"
    )

    print("\n== author downstream selection ==")
    for row in selection_rows:
        print(
            f"{row['dataset']}: {row['best_experiment']} "
            f"bacc={float(row['val_balanced_accuracy']):.4f} "
            f"kappa={float(row['val_kappa']):.4f} "
            f"prediction={row['prediction_path']}"
        )


if __name__ == "__main__":
    main()
