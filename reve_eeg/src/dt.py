"""Main downstream training script for linear probing and fine-tuning."""

import csv
import json
import math
import os
import random
import time
from builtins import print as bprint
from contextlib import nullcontext
from os.path import join as pjoin
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
import idr_torch
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from configs.resolver import register_resolvers
from downstream_tasks.dataloaders import get_data_loaders
from models.classifier import ReveClassifier
from models.encoder import REVE
from utils.model_utils import (
    freeze_model,
    get_flattened_output_dim,
    load_cls_query_token,
    load_encoder_checkpoint,
    unfreeze_model,
)
from utils.optim import get_lr_scheduler, get_optimizer


dtype_map = {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "float32": torch.float32}
FREQ = 200
METRIC_BROADCAST_KEYS = [
    "loss",
    "accuracy",
    "balanced_accuracy",
    "cohen_kappa",
    "macro_f1",
    "weighted_f1",
    "auroc",
    "auc_pr",
]


# Registry must be called to handle custom resolvers like ${env:SCRATCH}
register_resolvers()


def print(*args, **kwargs):
    if idr_torch.is_master or kwargs.pop("force", False):
        bprint(*args, **kwargs)


def _resolve_config_path(path_like: str | os.PathLike[str]) -> Path:
    path_str = str(path_like)
    raw_path = Path(path_str)
    if raw_path.is_absolute() or path_str.startswith("\\\\") or path_str.startswith("//"):
        return raw_path
    return Path(hydra.utils.to_absolute_path(path_str))


def _maybe_load_external_task_config(args: DictConfig) -> DictConfig:
    task_config_path = args.get("task_config_path")
    if task_config_path is None:
        return args

    resolved_path = _resolve_config_path(task_config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Course task config does not exist: {resolved_path}")

    task_cfg = OmegaConf.load(resolved_path)
    with open_dict(args):
        args.task = OmegaConf.merge(args.task, task_cfg)
    OmegaConf.resolve(args)
    print(f"Loaded external task config: {resolved_path}")
    return args


def _apply_runtime_task_overrides(args: DictConfig) -> DictConfig:
    if "task" not in args:
        return args

    with open_dict(args):
        if args.task.get("mlp_probing") is not None and args.task.get("linear_probing") is not None:
            args.task.mlp_probing = OmegaConf.merge(args.task.linear_probing, args.task.mlp_probing)

        classifier_cfg = args.task.get("classifier")
        if classifier_cfg is not None:
            if args.get("head_type") is not None:
                classifier_cfg.head_type = str(args.head_type)
            if args.get("mlp_hidden_dim") is not None:
                classifier_cfg.mlp_hidden_dim = int(args.mlp_hidden_dim)
            if args.get("classifier_dropout") is not None:
                classifier_cfg.dropout = float(args.classifier_dropout)

        fine_tuning_cfg = args.task.get("fine_tuning")
        if fine_tuning_cfg is not None:
            if args.get("encoder_lr") is not None:
                fine_tuning_cfg.encoder_lr = float(args.encoder_lr)
            if args.get("head_lr") is not None:
                fine_tuning_cfg.head_lr = float(args.head_lr)

    OmegaConf.resolve(args)
    return args


def _prepare_device(args: DictConfig) -> str:
    requested = str(args.trainer.device)
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to cpu", force=True)
            device = "cpu"
        elif requested == "cuda":
            device = f"cuda:{idr_torch.local_rank}"
        else:
            device = requested
            if ":" not in device:
                device = f"cuda:{idr_torch.local_rank}"
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
    else:
        device = requested

    args.trainer.device = device
    return device


def _resolve_pretrained_source(pretrained_path: str | None) -> tuple[str | None, bool]:
    if not pretrained_path:
        return None, False

    if pretrained_path.startswith("hf:"):
        return pretrained_path.split("hf:")[-1], True

    resolved_path = _resolve_config_path(pretrained_path)
    if resolved_path.is_dir():
        return str(resolved_path), True
    return str(resolved_path), False


def _maybe_get_summary_writer(log_dir: str | None):
    if not log_dir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard not installed. Please run: pip install tensorboard", force=True)
        return None
    return SummaryWriter(log_dir=log_dir)


def _resolve_output_dir(args: DictConfig) -> Path:
    requested = args.get("output_dir")
    if requested is None:
        output_dir = Path(os.getcwd())
    else:
        output_dir = _resolve_config_path(requested)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_reve_downstream_model(args: DictConfig, device: str) -> ReveClassifier:
    pretrained_source, use_hf_style_loading = _resolve_pretrained_source(args.pretrained_path)
    cls_query_token = None
    if pretrained_source and use_hf_style_loading:
        if pretrained_source.startswith("brain-bzh/"):
            print(f"Loading encoder from Hugging Face: {pretrained_source}")
            local_files_only = False
        else:
            print(f"Loading encoder from local HF directory: {pretrained_source}")
            local_files_only = True
        encoder, cls_query_token = REVE.from_pretrained(
            pretrained_source,
            cache_dir=args.get("cache_dir", ".cache"),
            local_files_only=local_files_only,
        )
    else:
        backbone_args = SimpleNamespace(
            embed_dim=args.encoder.transformer.embed_dim,
            depth=args.encoder.transformer.depth,
            heads=args.encoder.transformer.heads,
            head_dim=args.encoder.transformer.head_dim,
            mlp_dim_ratio=args.encoder.transformer.mlp_dim_ratio,
            use_geglu=args.encoder.transformer.use_geglu,
        )

        encoder = REVE(
            args_backbone=backbone_args,
            freqs=args.encoder.freqs,
            patch_size=args.encoder.patch_size,
            overlap_size=args.encoder.patch_overlap,
            noise_ratio=args.encoder.noise_ratio,
        )

        if pretrained_source and os.path.exists(pretrained_source):
            load_encoder_checkpoint(encoder, pretrained_source)
        else:
            print(f"Warning: Pretrained path {args.pretrained_path} not found or not specified.")

    if "n_chans" not in args.task:
        raise ValueError("n_chans must be specified in the task config")
    if "duration" not in args.task:
        raise ValueError("duration must be specified in the task config")

    n_chans = args.task.n_chans
    n_timepoints = int(args.task.duration * FREQ)
    print(f"Detected input shape: Chans={n_chans}, Timepoints={n_timepoints}")

    out_shape = None
    if args.task.classifier.pooling == "no":
        out_shape = get_flattened_output_dim(args, n_timepoints, n_chans)

    training_mode = args.get("training_mode", "lp")
    classifier_head_type = str(args.task.classifier.get("head_type", "linear"))
    if training_mode == "lp":
        classifier_head_type = "linear"
    elif training_mode == "mlp":
        classifier_head_type = "mlp"

    dropout = args.task.classifier.get("dropout", args.get("dropout", 0.0))

    model = ReveClassifier(
        encoder=encoder,
        n_classes=args.task.classifier.n_classes,
        dropout=dropout,
        pooling=args.task.classifier.pooling,
        head_type=classifier_head_type,
        mlp_hidden_dim=args.task.classifier.get("mlp_hidden_dim", encoder.embed_dim),
        out_shape=out_shape,
    )

    if pretrained_source and use_hf_style_loading:
        if cls_query_token is not None:
            print("Loading cls_query_token from pretrained REVE weights")
            model.cls_query_token.data.copy_(cls_query_token)
        else:
            print("WARNING: cls_query_token not found in pretrained REVE weights")
    elif pretrained_source and os.path.exists(pretrained_source):
        load_cls_query_token(model, pretrained_source)
    else:
        print(
            f"Warning: Pretrained path {args.pretrained_path} not found or not specified. cls_query_token not loaded.",
        )

    model.to(device)
    return model


def _broadcast_metrics(metrics: dict[str, Any], device: str) -> dict[str, Any]:
    if not dist.is_initialized():
        return metrics

    if idr_torch.is_master:
        values = [float(metrics[key]) for key in METRIC_BROADCAST_KEYS]
    else:
        values = [0.0 for _ in METRIC_BROADCAST_KEYS]
    tensor = torch.tensor(values, device=device, dtype=torch.float32)
    dist.broadcast(tensor, src=0)

    broadcasted = dict(metrics)
    for key, value in zip(METRIC_BROADCAST_KEYS, tensor.tolist(), strict=True):
        broadcasted[key] = float(value)
    if not idr_torch.is_master:
        broadcasted["confusion_matrix"] = []
    return broadcasted


def _write_metrics_row(metrics_csv_path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "stage",
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "val_cohen_kappa",
        "val_auroc",
        "val_auc_pr",
        "test_loss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_cohen_kappa",
        "learning_rate",
        "epoch_time_sec",
        "best_val_balanced_accuracy",
        "best_test_balanced_accuracy",
        "val_confusion_matrix",
        "test_confusion_matrix",
    ]
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_csv_path.exists()
    with metrics_csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _build_optimizer_parameters(
    model: ReveClassifier | DDP,
    current_cfg: DictConfig,
    stage_name: str,
):
    if stage_name != "ft":
        return filter(lambda p: p.requires_grad, model.parameters())

    encoder_lr = current_cfg.get("encoder_lr")
    head_lr = current_cfg.get("head_lr")
    if encoder_lr is None or head_lr is None:
        return filter(lambda p: p.requires_grad, model.parameters())

    target_model = model.module if isinstance(model, DDP) else model
    head_params = []
    encoder_params = []
    for name, param in target_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": float(head_lr), "name": "head"})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": float(encoder_lr), "name": "encoder"})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer setup.")

    return param_groups


def _get_lr_values(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    lr_values = {}
    for idx, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("name", f"group{idx}"))
        lr_values[group_name] = float(group["lr"])
    return lr_values


def _compact_console_logging_enabled(config: DictConfig) -> bool:
    return bool(config.get("compact_console_logging", False))


def _format_lr_console_suffix(lr_values: dict[str, float]) -> str:
    if "head" in lr_values and "encoder" in lr_values:
        return f"lr_head={lr_values['head']:.1e} | lr_encoder={lr_values['encoder']:.1e}"
    if "group0" in lr_values:
        return f"lr={lr_values['group0']:.1e}"
    first_key = next(iter(lr_values), None)
    if first_key is None:
        return "lr=nan"
    return f"lr_{first_key}={lr_values[first_key]:.1e}"


def train_stage(  # noqa: C901, PLR0912, PLR0913, PLR0915
    config: DictConfig,
    current_cfg: DictConfig,
    model: ReveClassifier | DDP,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader | None,
    stage_name: str,
    metrics_csv_path: Path,
    writer,
    output_dir: Path,
):
    scaler = torch.amp.GradScaler(
        device=config.trainer.device,
        enabled="cuda" in config.trainer.device and "16" in config.get("torch_dtype", "fp32"),
    )
    criterion = nn.CrossEntropyLoss()
    device = config.trainer.device
    skip_test_eval = bool(config.get("skip_test_eval", False))

    dtype_str = config.trainer.get("torch_dtype", "fp32")
    torch_dtype = dtype_map.get(dtype_str, torch.float32)

    optimizer = get_optimizer(
        _build_optimizer_parameters(model, current_cfg, stage_name),
        current_cfg.optimizer,
    )

    n_iter_per_epoch = len(train_loader)
    scheduler = get_lr_scheduler(optimizer, current_cfg, n_iter_per_epoch)

    warmup_epochs = current_cfg.warmup_epochs
    if warmup_epochs > 0:
        total_steps = len(train_loader) * warmup_epochs

        def exponential_warmup_lambda(step):
            if step < total_steps:
                return (10 ** (step / total_steps) - 1) / 9
            return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exponential_warmup_lambda)
    else:
        warmup_scheduler = None

    model.train()

    best_val = float("-inf")
    best_test = math.nan
    patience = 0

    n_epochs = int(current_cfg.n_epochs)
    patience_limit = int(current_cfg.patience)

    og_model = model.module if isinstance(model, DDP) else model
    compact_console = _compact_console_logging_enabled(config)

    print(f"Starting training stage: {stage_name} for {n_epochs} epochs. Device: {device}")

    for epoch in range(n_epochs):
        if compact_console and idr_torch.is_master:
            print(
                f"[{config.task.name}][{stage_name}][seed{config.seed}] "
                f"epoch {epoch + 1:03d}/{n_epochs:03d} starting | train_steps={len(train_loader)}",
                flush=True,
            )
        epoch_start = time.time()
        warmup = epoch < warmup_epochs
        train_metrics = train_one_epoch(
            model,
            criterion,
            optimizer,
            scaler,
            train_loader,
            warmup_scheduler if warmup else None,
            config,
            current_cfg,
        )

        if idr_torch.is_master:
            val_metrics = evaluate(
                og_model,
                val_loader,
                criterion=criterion,
                device=device,
                binary=bool(config.task.classifier.n_classes == 2),
                amp_dtype=torch_dtype,
                compact_console=compact_console,
            )
        else:
            val_metrics = {key: 0.0 for key in METRIC_BROADCAST_KEYS}
            val_metrics["confusion_matrix"] = []
        val_metrics = _broadcast_metrics(val_metrics, device)

        test_metrics: dict[str, Any] | None = None
        if val_metrics["balanced_accuracy"] > best_val:
            best_val = float(val_metrics["balanced_accuracy"])
            patience = 0

            if not skip_test_eval and test_loader is not None:
                if idr_torch.is_master:
                    test_metrics = evaluate(
                        og_model,
                        test_loader,
                        criterion=criterion,
                        device=device,
                        binary=bool(config.task.classifier.n_classes == 2),
                        amp_dtype=torch_dtype,
                        compact_console=compact_console,
                    )
                else:
                    test_metrics = {key: 0.0 for key in METRIC_BROADCAST_KEYS}
                    test_metrics["confusion_matrix"] = []
                test_metrics = _broadcast_metrics(test_metrics, device)
                best_test = float(test_metrics["balanced_accuracy"])

            if idr_torch.is_master:
                target_dir = output_dir
                torch.save(og_model.state_dict(), target_dir / "model_best.pth")
                (target_dir / "best_val_confusion_matrix.json").write_text(
                    json.dumps(val_metrics["confusion_matrix"], indent=2),
                    encoding="utf-8",
                )
                if test_metrics is not None:
                    (target_dir / "best_test_confusion_matrix.json").write_text(
                        json.dumps(test_metrics["confusion_matrix"], indent=2),
                        encoding="utf-8",
                    )
                if not compact_console:
                    print(f"New best model saved with val_balanced_acc: {best_val:.4f}")
        else:
            patience += 1

        lr_values = _get_lr_values(optimizer)
        lr_curr = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.time() - epoch_start
        lr_summary = _format_lr_console_suffix(lr_values)
        if compact_console:
            log = (
                f"[{config.task.name}][{stage_name}][seed{config.seed}] "
                f"epoch {epoch + 1:03d}/{n_epochs:03d} | "
                f"train_loss={train_metrics['loss']:.3f} | "
                f"train_acc={train_metrics['accuracy']:.3f} | "
                f"val_loss={val_metrics['loss']:.3f} | "
                f"val_acc={val_metrics.get('accuracy', float('nan')):.3f} | "
                f"val_bal_acc={val_metrics['balanced_accuracy']:.3f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.3f} | "
                f"val_weighted_f1={val_metrics.get('weighted_f1', float('nan')):.3f} | "
                f"best={best_val:.3f} | {lr_summary} | time={epoch_time:.0f}s"
            )
        else:
            best_test_str = "nan" if math.isnan(best_test) else f"{best_test:.3f}"
            log = (
                f"Stage {stage_name} | Epoch {epoch:3d} | "
                f"best test: {best_test_str}, best val: {best_val:.3f}, val: {val_metrics['balanced_accuracy']:.3f}, "
                f"LR {lr_curr:.8f}, patience {patience:3d}/{patience_limit + 1:3d}, "
                f"train_loss: {train_metrics['loss']:.4f}, val_loss: {val_metrics['loss']:.4f}, "
                f"epoch_time: {epoch_time:.1f}s"
            )
        print(log)

        if writer is not None and idr_torch.is_master:
            writer.add_scalar("train/loss", float(train_metrics["loss"]), epoch)
            writer.add_scalar("train/accuracy", float(train_metrics["accuracy"]), epoch)
            writer.add_scalar("val/loss", float(val_metrics["loss"]), epoch)
            writer.add_scalar("val/accuracy", float(val_metrics["accuracy"]), epoch)
            writer.add_scalar("val/balanced_accuracy", float(val_metrics["balanced_accuracy"]), epoch)
            writer.add_scalar("val/macro_f1", float(val_metrics["macro_f1"]), epoch)
            writer.add_scalar("val/weighted_f1", float(val_metrics["weighted_f1"]), epoch)
            writer.add_scalar("learning_rate", lr_curr, epoch)
            if "encoder" in lr_values:
                writer.add_scalar("learning_rate/encoder", lr_values["encoder"], epoch)
            if "head" in lr_values:
                writer.add_scalar("learning_rate/head", lr_values["head"], epoch)
            writer.add_scalar("epoch_time", epoch_time, epoch)
            writer.flush()

        if idr_torch.is_master:
            row = {
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_weighted_f1": float(val_metrics["weighted_f1"]),
                "val_cohen_kappa": float(val_metrics["cohen_kappa"]),
                "val_auroc": float(val_metrics["auroc"]),
                "val_auc_pr": float(val_metrics["auc_pr"]),
                "test_loss": "" if test_metrics is None else float(test_metrics["loss"]),
                "test_accuracy": "" if test_metrics is None else float(test_metrics["accuracy"]),
                "test_balanced_accuracy": "" if test_metrics is None else float(test_metrics["balanced_accuracy"]),
                "test_macro_f1": "" if test_metrics is None else float(test_metrics["macro_f1"]),
                "test_weighted_f1": "" if test_metrics is None else float(test_metrics["weighted_f1"]),
                "test_cohen_kappa": "" if test_metrics is None else float(test_metrics["cohen_kappa"]),
                "learning_rate": lr_curr,
                "epoch_time_sec": epoch_time,
                "best_val_balanced_accuracy": best_val,
                "best_test_balanced_accuracy": "" if math.isnan(best_test) else best_test,
                "val_confusion_matrix": json.dumps(val_metrics["confusion_matrix"]),
                "test_confusion_matrix": "" if test_metrics is None else json.dumps(test_metrics["confusion_matrix"]),
            }
            _write_metrics_row(metrics_csv_path, row)

        if epoch >= warmup_epochs:
            scheduler.step(val_metrics["accuracy"])
        if patience > patience_limit:
            print(f"Stage {stage_name} finished due to patience limit.")
            break

    return model


def train_one_epoch(  # noqa: PLR0913
    model: ReveClassifier | DDP,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    train_loader: torch.utils.data.DataLoader,
    warmup_scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    config: DictConfig,
    current_cfg: DictConfig,
) -> dict[str, float]:
    losses: list[float] = []
    total_correct = 0
    total_count = 0

    model.train()

    use_mixup = current_cfg.get("mixup", False)
    device = config.trainer.device

    dtype = config.trainer.get("torch_dtype", "fp32")
    torch_dtype = dtype_map.get(dtype, torch.float32)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch_dtype) if "16" in dtype and "cuda" in device else nullcontext()

    compact_console = _compact_console_logging_enabled(config)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=not idr_torch.is_master or compact_console)
    ema_loss = None
    for batch_idx, batch_data in pbar:
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            if isinstance(batch_data, dict):
                data = batch_data["sample"]
                target = batch_data["label"]
                pos = batch_data["pos"]
            else:
                data, target, pos = batch_data

            data, target, pos = (
                data.to(device, non_blocking=True),
                target.long().to(device, non_blocking=True),
                pos.to(device, non_blocking=True),
            )

            if use_mixup:
                mm = random.random()
                perm = torch.randperm(data.shape[0], device=data.device)
                output = model(mm * data + (1 - mm) * data[perm], pos)
                loss = mm * criterion(output, target) + (1 - mm) * criterion(output, target[perm])
            else:
                output = model(data, pos)
                loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.trainer.clip_grad)
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()

        losses.append(loss.item())

        decisions = torch.argmax(output.detach(), dim=1)
        total_correct += int((decisions == target).sum().item())
        total_count += int(target.shape[0])

        ema_loss = loss.item() if ema_loss is None else 0.95 * ema_loss + 0.05 * loss.item()
        if not compact_console:
            pbar.set_postfix(ema_loss=f"{ema_loss:.4f}")

        skip_lr_sched = scale != scaler.get_scale()
        if not skip_lr_sched and warmup_scheduler is not None:
            warmup_scheduler.step()

    return {
        "loss": sum(losses) / len(losses),
        "accuracy": total_correct / total_count if total_count > 0 else 0.0,
    }


def evaluate(
    model: ReveClassifier,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module | None = None,
    device="cuda",
    binary=False,
    amp_dtype=torch.float16,
    compact_console=False,
) -> dict[str, Any]:
    score = 0
    count = 0
    total_loss = 0.0
    model.eval()
    y_decisions = []
    y_targets = []
    y_probs = []
    ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if "cuda" in str(device) and "16" in str(amp_dtype) else nullcontext()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), disable=not idr_torch.is_master or compact_console)
    for _, batch_data in pbar:
        with torch.no_grad(), ctx:
            if isinstance(batch_data, dict):
                if "label" not in batch_data:
                    raise ValueError("Label is required for evaluation but missing from the batch.")
                data = batch_data["sample"]
                target = batch_data["label"]
                pos = batch_data["pos"]
            else:
                data, target, pos = batch_data

            data, target, pos = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
                pos.to(device, non_blocking=True),
            )
            output = model(data, pos)
            if criterion is not None:
                loss = criterion(output, target.long())
                total_loss += float(loss.item()) * int(target.shape[0])
            decisions = torch.argmax(output, dim=1)
            score += int((decisions == target).sum().item())
            count += int(target.shape[0])
            y_decisions.append(decisions.detach().cpu())
            y_targets.append(target.detach().cpu())
            y_probs.append(output.detach().cpu())

    gt = torch.cat(y_targets).detach().cpu().numpy()
    pr = torch.cat(y_decisions).detach().cpu().numpy()
    pr_probs = torch.cat(y_probs).detach().cpu().numpy()
    acc = score / count
    balanced_acc = balanced_accuracy_score(gt, pr)
    cohen_kappa = cohen_kappa_score(gt, pr)
    macro_f1 = f1_score(gt, pr, average="macro")
    weighted_f1 = f1_score(gt, pr, average="weighted")
    conf_mat = confusion_matrix(gt, pr).tolist()
    if binary and pr_probs.shape[1] == 2:
        auroc = roc_auc_score(gt, pr_probs[:, 1])
        auc_pr = average_precision_score(gt, pr_probs[:, 1])
    else:
        auroc = 0.0
        auc_pr = 0.0

    return {
        "loss": total_loss / count if criterion is not None and count > 0 else 0.0,
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "cohen_kappa": cohen_kappa,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "auroc": auroc,
        "auc_pr": auc_pr,
        "confusion_matrix": conf_mat,
    }


def test(
    model: ReveClassifier,
    test_loader: torch.utils.data.DataLoader,
    device="cuda",
    binary=False,
    amp_dtype=torch.float16,
):
    metrics = evaluate(
        model,
        test_loader,
        criterion=None,
        device=device,
        binary=binary,
        amp_dtype=amp_dtype,
    )
    return (
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["cohen_kappa"],
        metrics["weighted_f1"],
        metrics["auroc"],
        metrics["auc_pr"],
    )


@hydra.main(version_base=None, config_name="config_dt", config_path="configs")
def main(args):  # noqa: C901, PLR0912, PLR0915
    args = _maybe_load_external_task_config(args)
    args = _apply_runtime_task_overrides(args)

    if idr_torch.world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=idr_torch.world_size,
            rank=idr_torch.rank,
        )

    device = _prepare_device(args)
    output_dir = _resolve_output_dir(args)

    init_seed = args.seed + idr_torch.rank
    torch.manual_seed(init_seed)
    random.seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)

    print(f"Running with task: {args.task.name}")
    print(f"Output directory: {output_dir}")

    tensorboard_dir_cfg = Path(str(args.get("tensorboard_dir", "tensorboard")))
    tensorboard_dir = tensorboard_dir_cfg if tensorboard_dir_cfg.is_absolute() else output_dir / tensorboard_dir_cfg
    writer = _maybe_get_summary_writer(str(tensorboard_dir))
    metrics_csv_cfg = Path(str(args.get("metrics_csv_path", "metrics.csv")))
    metrics_csv_path = metrics_csv_cfg if metrics_csv_cfg.is_absolute() else output_dir / metrics_csv_cfg

    og_model = build_reve_downstream_model(args, device)
    training_mode = args.get("training_mode", "lp")

    data_loaders = None
    if args.task.data_loader.batch_size is not None:
        data_loaders = get_data_loaders(args.task.data_loader, args.loader)
    modes = []

    if training_mode == "lp":
        modes.append("lp")
    elif training_mode == "mlp":
        modes.append("mlp")
    elif training_mode == "ft":
        modes.append("ft")
    elif training_mode == "lp+ft":
        modes = ["lp", "ft"]
    else:
        raise ValueError(f"Unknown training_mode {training_mode}")

    for mode in modes:
        if mode == "lp":
            print(">>> Setup Linear Probing (LP)")
            freeze_model(og_model)
            current_cfg = args.task.linear_probing
        elif mode == "mlp":
            print(">>> Setup MLP Probing (MLP)")
            freeze_model(og_model)
            current_cfg = args.task.get("mlp_probing", args.task.linear_probing)
        elif mode == "ft":
            print(">>> Setup Fine-Tuning (FT)")
            unfreeze_model(og_model)
            current_cfg = args.task.fine_tuning
        else:
            raise ValueError(f"Unknown sub-mode {mode}")

        if idr_torch.size > 1:
            find_unused = mode in {"lp", "mlp"}
            model = DDP(og_model, device_ids=[idr_torch.local_rank], find_unused_parameters=find_unused)
        else:
            model = og_model

        if "batch_size" in current_cfg or data_loaders is None:
            new_bs = current_cfg.batch_size
            print(f"Batch size for stage {mode}: {new_bs}")
            args.task.data_loader.batch_size = new_bs
            data_loaders = get_data_loaders(args.task.data_loader, args.loader)

        train_stage(
            args,
            current_cfg,
            model,
            data_loaders["train"],
            data_loaders["val"],
            data_loaders.get("test"),
            stage_name=mode,
            metrics_csv_path=metrics_csv_path,
            writer=writer,
            output_dir=output_dir,
        )

    target_dir = output_dir
    if idr_torch.is_master:
        if isinstance(model, DDP):
            torch.save(model.module.state_dict(), target_dir / "model_final.pth")
        else:
            torch.save(model.state_dict(), target_dir / "model_final.pth")
        print(f"Model saved to {target_dir}")

    if writer is not None:
        writer.close()

    if idr_torch.size > 1 or dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
