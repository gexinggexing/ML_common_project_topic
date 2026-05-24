from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import dt as dt_module  # noqa: E402


RESOURCE_CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "our_tasks" / "resource_paths.yaml"
ENCODER_CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "encoder" / "base.yaml"
COURSE_CONFIGS = {
    "BCIC2A": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_bcic2a.yaml",
    "BCI_Speech": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_bci_speech.yaml",
    "CHINESE": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_chinese.yaml",
    "MDD": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_mdd.yaml",
    "SEED": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_seed.yaml",
    "SLEEP": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_sleep.yaml",
}
DEFAULT_BATCH_SIZES = {
    "BCIC2A": 16,
    "BCI_Speech": 32,
    "CHINESE": 16,
    "MDD": 16,
    "SEED": 8,
    "SLEEP": 2,
}
MODE_HEAD_DEFAULTS = {
    "lp": {"head_type": "linear", "classifier_dropout": 0.1, "mlp_hidden_dim": 512},
    "mlp": {"head_type": "mlp", "classifier_dropout": 0.2, "mlp_hidden_dim": 512},
    "ft": {"head_type": "linear", "classifier_dropout": 0.1, "mlp_hidden_dim": 512},
    "lp+ft": {"head_type": "linear", "classifier_dropout": 0.1, "mlp_hidden_dim": 512},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export test_x_only predictions for one finished course-project run.")
    parser.add_argument("--resource", type=str, default=None, help="Resource key in resource_paths.yaml, or 'manual'.")
    parser.add_argument("--data-root", type=str, default=None, help="Override data root.")
    parser.add_argument("--ckpt-root", type=str, default=None, help="Override checkpoint root.")
    parser.add_argument("--out-root", type=str, default=None, help="Override output root.")
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(COURSE_CONFIGS), help="Dataset name.")
    parser.add_argument("--mode", type=str, required=True, help="Run mode such as lp, mlp, ft, or lp+ft.")
    parser.add_argument("--seed", type=int, required=True, help="Run seed.")
    parser.add_argument("--run-dir", type=str, default=None, help="Existing run directory. Defaults to out_root/dataset/mode/seedX.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Defaults to <run_dir>/model_best.pth.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device.")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader num_workers override.")
    parser.add_argument("--python-exe", type=str, default=None, help="Accepted for interface symmetry; not used internally.")
    return parser.parse_args()


def load_resource_configs() -> dict[str, dict[str, object]]:
    with RESOURCE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return {str(key): value for key, value in raw.items()}


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def to_rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def resolve_runtime(args: argparse.Namespace) -> dict[str, Any]:
    resource_cfg = None
    if args.resource and args.resource != "manual":
        resource_configs = load_resource_configs()
        if args.resource not in resource_configs:
            available = ", ".join(sorted(resource_configs))
            raise KeyError(f"Unknown resource '{args.resource}'. Available resources: {available}")
        resource_cfg = resource_configs[args.resource]

    data_root_value = args.data_root or (str(resource_cfg["data_root"]) if resource_cfg is not None else None)
    ckpt_root_value = args.ckpt_root or (str(resource_cfg["ckpt_root"]) if resource_cfg is not None else None)
    out_root_value = args.out_root or (str(resource_cfg["out_root"]) if resource_cfg is not None else "outputs/course_project")
    if data_root_value is None or ckpt_root_value is None:
        raise ValueError("predict_test.py requires either --resource or explicit --data-root and --ckpt-root")

    data_root = resolve_path(data_root_value)
    ckpt_root = resolve_path(ckpt_root_value)
    out_root = resolve_path(out_root_value)
    batch_size = int(args.batch_size) if args.batch_size is not None else int(DEFAULT_BATCH_SIZES[args.dataset])
    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(resource_cfg["num_workers"]) if resource_cfg is not None else 0
    )

    run_dir = resolve_path(args.run_dir) if args.run_dir else out_root / args.dataset / args.mode / f"seed{args.seed}"
    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else run_dir / "model_best.pth"

    return {
        "data_root": data_root,
        "ckpt_root": ckpt_root,
        "out_root": out_root,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "run_dir": run_dir,
        "checkpoint": checkpoint,
    }


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_command_txt_overrides(command_text: str) -> dict[str, Any]:
    patterns = {
        "head_type": r"\+head_type=(?:\\\")?(linear|mlp)(?:\\\")?",
        "classifier_dropout": r"\+classifier_dropout=([0-9.eE+-]+)",
        "mlp_hidden_dim": r"\+mlp_hidden_dim=([0-9]+)",
    }
    parsed: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, command_text)
        if match is None:
            continue
        raw_value = match.group(1)
        if key == "head_type":
            parsed[key] = str(raw_value)
        elif key == "mlp_hidden_dim":
            parsed[key] = int(raw_value)
        else:
            parsed[key] = float(raw_value)
    return parsed


def resolve_run_overrides(run_dir: Path, mode: str) -> dict[str, Any]:
    overrides = dict(MODE_HEAD_DEFAULTS.get(mode, MODE_HEAD_DEFAULTS["lp"]))
    run_config = load_json_if_exists(run_dir / "run_config.json")
    if run_config is not None:
        mode_cfg = run_config.get("mode_config", {})
        if isinstance(mode_cfg, dict):
            if "head_type" in mode_cfg:
                overrides["head_type"] = str(mode_cfg["head_type"])
            if "classifier_dropout" in mode_cfg:
                overrides["classifier_dropout"] = float(mode_cfg["classifier_dropout"])
            if "mlp_hidden_dim" in mode_cfg:
                overrides["mlp_hidden_dim"] = int(mode_cfg["mlp_hidden_dim"])

    command_path = run_dir / "command.txt"
    if command_path.exists():
        overrides.update(parse_command_txt_overrides(command_path.read_text(encoding="utf-8")))

    return overrides


def build_runtime_config(
    dataset: str,
    mode: str,
    seed: int,
    data_root: Path,
    ckpt_root: Path,
    device: str,
    run_overrides: dict[str, Any],
) -> Any:
    task_config_path = COURSE_CONFIGS[dataset]
    cfg = OmegaConf.create(
        {
            "task": {},
            "task_config_path": str(task_config_path),
            "data_root": str(data_root),
            "ckpt_root": str(ckpt_root),
            "batch_size": DEFAULT_BATCH_SIZES[dataset],
            "seed": int(seed),
            "n_epochs": 2,
            "patience": 2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "encoder_lr": 1e-5,
            "head_lr": 1e-4,
            "head_type": run_overrides["head_type"],
            "mlp_hidden_dim": int(run_overrides["mlp_hidden_dim"]),
            "classifier_dropout": float(run_overrides["classifier_dropout"]),
            "training_mode": mode,
            "pretrained_path": str(ckpt_root / "reve-base"),
            "cache_dir": str(PROJECT_ROOT / ".cache"),
            "trainer": {"device": device, "torch_dtype": "fp16"},
            "encoder": OmegaConf.load(ENCODER_CONFIG_PATH),
        },
    )
    cfg = dt_module._maybe_load_external_task_config(cfg)
    cfg = dt_module._apply_runtime_task_overrides(cfg)
    return cfg


def instantiate_test_loader(runtime_cfg: Any, dataset_name: str, batch_size: int, num_workers: int) -> tuple[Any, DataLoader]:
    dataset = hydra.utils.instantiate(runtime_cfg.task.data_loader.dataset, mode="test")
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "collate_fn": dataset.collate,
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": str(runtime_cfg.trainer.device).startswith("cuda"),
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    print(
        f"Loaded test dataset for {dataset_name}: "
        f"{len(dataset):,} samples from {dataset.h5_path} with batch_size={batch_size}"
    )
    print(f"Normalization stats: {dataset.train_stats_path}")
    print(f"Channels not found: {dataset.channels_not_found}")
    print(f"Fallback channels: {dataset.fallback_channels}")
    return dataset, loader


def sanitize_samples_if_needed(dataset_name: str, sample: torch.Tensor) -> tuple[torch.Tensor, int]:
    if dataset_name != "CHINESE":
        return sample, 0
    non_finite_mask = ~torch.isfinite(sample)
    non_finite_count = int(non_finite_mask.sum().item())
    if non_finite_count > 0:
        warnings.warn(
            f"CHINESE test batch contains {non_finite_count} non-finite values; applying torch.nan_to_num before inference.",
            stacklevel=2,
        )
        sample = torch.nan_to_num(sample, nan=0.0, posinf=1e6, neginf=-1e6)
    return sample, non_finite_count


def save_predictions(
    run_dir: Path,
    sample_ids: np.ndarray,
    logits: np.ndarray,
    inverse_label_mapping: dict[int, int],
) -> tuple[Path, Path]:
    order = np.argsort(sample_ids, kind="stable")
    sample_ids = sample_ids[order]
    logits = logits[order]
    probs = torch.softmax(torch.from_numpy(logits), dim=1).cpu().numpy()
    pred_labels = probs.argmax(axis=1)
    pred_original = [inverse_label_mapping.get(int(label)) for label in pred_labels.tolist()]

    logits_path = run_dir / "test_logits.npy"
    np.save(logits_path, logits.astype(np.float32, copy=False))

    prediction_path = run_dir / "test_predictions.csv"
    fieldnames = ["sample_id", "pred_label", "pred_label_original"] + [f"prob_{index}" for index in range(probs.shape[1])]
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(len(sample_ids)):
            row = {
                "sample_id": int(sample_ids[row_index]),
                "pred_label": int(pred_labels[row_index]),
                "pred_label_original": "" if pred_original[row_index] is None else int(pred_original[row_index]),
            }
            for prob_index in range(probs.shape[1]):
                row[f"prob_{prob_index}"] = float(probs[row_index, prob_index])
            writer.writerow(row)

    return prediction_path, logits_path


def main() -> None:
    args = parse_args()
    runtime = resolve_runtime(args)
    run_dir = Path(runtime["run_dir"])
    checkpoint_path = Path(runtime["checkpoint"])
    run_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    run_overrides = resolve_run_overrides(run_dir, args.mode)
    runtime_cfg = build_runtime_config(
        dataset=args.dataset,
        mode=args.mode,
        seed=args.seed,
        data_root=Path(runtime["data_root"]),
        ckpt_root=Path(runtime["ckpt_root"]),
        device=args.device,
        run_overrides=run_overrides,
    )
    device = dt_module._prepare_device(runtime_cfg)
    model = dt_module.build_reve_downstream_model(runtime_cfg, device)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch while loading model_best.pth: "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )

    model.eval()
    test_dataset, test_loader = instantiate_test_loader(
        runtime_cfg,
        dataset_name=args.dataset,
        batch_size=int(runtime["batch_size"]),
        num_workers=int(runtime["num_workers"]),
    )

    all_sample_ids: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    total_non_finite = 0
    with torch.inference_mode():
        for batch in test_loader:
            sample = batch["sample"].to(device, non_blocking=device.startswith("cuda"))
            pos = batch["pos"].to(device, non_blocking=device.startswith("cuda"))
            sample, non_finite_count = sanitize_samples_if_needed(args.dataset, sample)
            total_non_finite += non_finite_count

            with (
                torch.autocast(
                    device_type="cuda",
                    dtype=dt_module.dtype_map.get(str(runtime_cfg.trainer.torch_dtype), torch.float16),
                )
                if device.startswith("cuda")
                else nullcontext()
            ):
                logits = model(sample, pos)

            if not torch.isfinite(logits).all():
                warnings.warn(
                    f"{args.dataset} inference produced non-finite logits; applying torch.nan_to_num before export.",
                    stacklevel=2,
                )
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

            all_sample_ids.append(batch["sample_id"].cpu().numpy())
            all_logits.append(logits.float().cpu().numpy())

    sample_ids = np.concatenate(all_sample_ids, axis=0) if all_sample_ids else np.empty(0, dtype=np.int64)
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 0), dtype=np.float32)
    expected_shape = (len(test_dataset), int(runtime_cfg.task.classifier.n_classes))
    if tuple(logits.shape) != expected_shape:
        raise ValueError(f"Expected logits shape {expected_shape}, got {tuple(logits.shape)}")

    prediction_path, logits_path = save_predictions(
        run_dir=run_dir,
        sample_ids=sample_ids,
        logits=logits,
        inverse_label_mapping=test_dataset.inverse_label_mapping,
    )

    summary = {
        "dataset": args.dataset,
        "mode": args.mode,
        "seed": args.seed,
        "run_dir": to_rel_or_abs(run_dir),
        "checkpoint": to_rel_or_abs(checkpoint_path),
        "prediction_csv": to_rel_or_abs(prediction_path),
        "logits_npy": to_rel_or_abs(logits_path),
        "logits_shape": list(logits.shape),
        "x_key": test_dataset.x_key,
        "channels_not_found": test_dataset.channels_not_found,
        "fallback_channels": test_dataset.fallback_channels,
        "normalization_stats_path": to_rel_or_abs(test_dataset.train_stats_path),
        "non_finite_values_handled": int(total_non_finite),
    }
    (run_dir / "test_prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Saved predictions to: {prediction_path}")
    print(f"Saved logits to: {logits_path}")
    print(f"Logits shape: {tuple(logits.shape)}")
    if args.dataset == "CHINESE":
        print(f"CHINESE non-finite values handled: {total_non_finite}")


if __name__ == "__main__":
    main()
