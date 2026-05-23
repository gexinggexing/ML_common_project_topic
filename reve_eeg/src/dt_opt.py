"""Optuna-driven downstream training script for course-project hyperparameter search."""

from __future__ import annotations

import copy
import csv
import gc
import json
import random
from builtins import print as bprint
from contextlib import suppress
from pathlib import Path

import hydra
import idr_torch
import optuna
import torch
import torch.distributed as dist
import torch.multiprocessing
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

import dt as dt_module
from configs.resolver import register_resolvers
from downstream_tasks.dataloaders import get_data_loaders
from utils.model_utils import freeze_model, parse_optuna_config, unfreeze_model


def cleanup_loaders(data_loaders):
    if data_loaders is not None:
        for split in ["train", "val", "test"]:
            if split not in data_loaders:
                continue
            loader = data_loaders[split]
            ds = getattr(loader, "dataset", None)
            if hasattr(ds, "close"):
                with suppress(Exception):
                    ds.close()
            if hasattr(loader, "_iterator") and loader._iterator is not None:
                shutdown = getattr(loader._iterator, "_shutdown_workers", None)
                if callable(shutdown):
                    with suppress(Exception):
                        shutdown()
                with suppress(Exception):
                    loader._iterator = None


torch.multiprocessing.set_sharing_strategy("file_system")
register_resolvers()


def print(*args, **kwargs):
    if idr_torch.is_master or kwargs.pop("force", False):
        bprint(*args, **kwargs)


def _collect_relevant_optuna_params(cfg: DictConfig) -> dict[str, dict]:
    params: dict[str, dict] = {}
    training_mode = str(cfg.get("training_mode", "lp"))

    classifier_cfg = cfg.task.get("classifier")
    if classifier_cfg is not None:
        classifier_params = parse_optuna_config(classifier_cfg, current_path="task.classifier")
        if training_mode != "mlp":
            classifier_params = {
                path: spec for path, spec in classifier_params.items() if not path.endswith("mlp_hidden_dim")
            }
        params.update(classifier_params)

    if training_mode == "lp":
        params.update(parse_optuna_config(cfg.task.linear_probing, current_path="task.linear_probing"))
    elif training_mode == "mlp":
        params.update(parse_optuna_config(cfg.task.mlp_probing, current_path="task.mlp_probing"))
    elif training_mode == "ft":
        params.update(parse_optuna_config(cfg.task.fine_tuning, current_path="task.fine_tuning"))
    elif training_mode == "lp+ft":
        params.update(parse_optuna_config(cfg.task.linear_probing, current_path="task.linear_probing"))
        params.update(parse_optuna_config(cfg.task.fine_tuning, current_path="task.fine_tuning"))
    else:
        raise ValueError(f"Unknown training_mode {training_mode}")

    return params


def _read_best_val_bal_acc(metrics_csv_path: Path) -> float | None:
    if not metrics_csv_path.exists():
        return None

    best_value: float | None = None
    with metrics_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_value = row.get("best_val_balanced_accuracy")
            if raw_value in {None, ""}:
                raw_value = row.get("val_balanced_accuracy")
            if raw_value in {None, ""}:
                continue
            value = float(raw_value)
            if best_value is None or value > best_value:
                best_value = value
    return best_value


def _write_trials_csv(study: optuna.Study, output_path: Path) -> None:
    all_param_keys = sorted({key for trial in study.trials for key in trial.params})
    fieldnames = ["trial_number", "value", "state"] + all_param_keys
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            row = {
                "trial_number": trial.number,
                "value": "" if trial.value is None else float(trial.value),
                "state": str(trial.state),
            }
            for key in all_param_keys:
                row[key] = trial.params.get(key, "")
            writer.writerow(row)


def _trial_dir(base_output_dir: Path, trial_number: int) -> Path:
    return base_output_dir / f"trial_{trial_number}"


def evaluate_trial(
    args: DictConfig,
    trial_number: int,
    data_loaders,
    base_output_dir: Path,
) -> float:  # noqa: C901, PLR0912
    init_seed = int(args.seed) + int(idr_torch.rank)
    torch.manual_seed(init_seed)
    random.seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)

    trial_dir = _trial_dir(base_output_dir, trial_number)
    trial_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = trial_dir / "metrics.csv"
    for stale_file in [
        metrics_csv_path,
        trial_dir / "model_best.pth",
        trial_dir / "model_final.pth",
        trial_dir / "trial_summary.json",
        trial_dir / "best_val_confusion_matrix.json",
        trial_dir / "best_test_confusion_matrix.json",
    ]:
        if stale_file.exists():
            stale_file.unlink()

    writer = dt_module._maybe_get_summary_writer(str(trial_dir / "tensorboard"))
    device = str(args.trainer.device)

    print(
        f"========== TRIAL_START dataset={args.task.name} mode={args.training_mode} "
        f"seed={args.seed} trial={trial_number} device={device} ==========",
    )

    og_model = dt_module.build_reve_downstream_model(args, device)
    training_mode = str(args.get("training_mode", "lp"))

    if training_mode == "lp":
        modes = ["lp"]
    elif training_mode == "mlp":
        modes = ["mlp"]
    elif training_mode == "ft":
        modes = ["ft"]
    elif training_mode == "lp+ft":
        modes = ["lp", "ft"]
    else:
        raise ValueError(f"Unknown training_mode {training_mode}")

    local_data_loaders = data_loaders
    for mode in modes:
        if mode == "lp":
            print(f">>> Setup Linear Probing (LP) for trial {trial_number}")
            freeze_model(og_model)
            current_cfg = args.task.linear_probing
        elif mode == "mlp":
            print(f">>> Setup MLP Probing (MLP) for trial {trial_number}")
            freeze_model(og_model)
            current_cfg = args.task.get("mlp_probing", args.task.linear_probing)
        elif mode == "ft":
            print(f">>> Setup Fine-Tuning (FT) for trial {trial_number}")
            unfreeze_model(og_model)
            current_cfg = args.task.fine_tuning
        else:
            raise ValueError(f"Unknown sub-mode {mode}")

        if idr_torch.size > 1:
            find_unused = mode in {"lp", "mlp"}
            model = DDP(og_model, device_ids=[idr_torch.local_rank], find_unused_parameters=find_unused)
        else:
            model = og_model

        if local_data_loaders is None:
            local_data_loaders = get_data_loaders(args.task.data_loader, args.loader)

        dt_module.train_stage(
            args,
            current_cfg,
            model,
            local_data_loaders["train"],
            local_data_loaders["val"],
            local_data_loaders.get("test"),
            stage_name=mode,
            metrics_csv_path=metrics_csv_path,
            writer=writer,
            output_dir=trial_dir,
        )

    if idr_torch.is_master:
        if isinstance(model, DDP):
            torch.save(model.module.state_dict(), trial_dir / "model_final.pth")
        else:
            torch.save(model.state_dict(), trial_dir / "model_final.pth")

    if writer is not None:
        writer.close()

    best_val = _read_best_val_bal_acc(metrics_csv_path)
    summary = {
        "trial_number": int(trial_number),
        "dataset": str(args.task.name),
        "training_mode": training_mode,
        "seed": int(args.seed),
        "best_val_balanced_accuracy": None if best_val is None else float(best_val),
        "trial_dir": str(trial_dir),
        "metrics_csv": str(metrics_csv_path),
        "best_checkpoint": str(trial_dir / "model_best.pth"),
        "final_checkpoint": str(trial_dir / "model_final.pth"),
    }
    if idr_torch.is_master:
        (trial_dir / "trial_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_val_str = "nan" if best_val is None else f"{best_val:.3f}"
    print(
        f"========== TRIAL_DONE dataset={args.task.name} mode={args.training_mode} "
        f"seed={args.seed} trial={trial_number} best_val_bal_acc={best_val_str} ==========",
    )

    del og_model
    del model
    gc.collect()

    return float("-inf") if best_val is None else float(best_val)


@hydra.main(version_base=None, config_name="config_dt", config_path="configs")
def main(args):  # noqa: C901, PLR0912, PLR0915
    args = dt_module._maybe_load_external_task_config(args)
    args = dt_module._apply_runtime_task_overrides(args)

    if idr_torch.world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=idr_torch.world_size,
            rank=idr_torch.rank,
        )

    device = dt_module._prepare_device(args)
    output_dir = dt_module._resolve_output_dir(args)

    cfg_copy = copy.deepcopy(args)
    OmegaConf.resolve(cfg_copy)
    OmegaConf.set_struct(cfg_copy, False)
    cfg_copy.trainer.device = device

    optuna_params = _collect_relevant_optuna_params(cfg_copy)
    n_trials = int(args.n_runs)

    print(f"Running Optuna with task: {cfg_copy.task.name}")
    print(f"Training mode: {cfg_copy.training_mode}")
    print(f"Output directory: {output_dir}")
    print(f"Starting optuna search for {n_trials} trials.")

    global_data_loaders = None
    if cfg_copy.task.data_loader.batch_size is not None:
        global_data_loaders = get_data_loaders(cfg_copy.task.data_loader, cfg_copy.loader)

    if idr_torch.is_master:
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{cfg_copy.task.name}_{cfg_copy.training_mode}_seed{cfg_copy.seed}",
        )
        if optuna_params:
            init_params = {path: space["init"] for path, space in optuna_params.items()}
            print(f"Enqueueing initial trial with params: {init_params}")
            study.enqueue_trial(init_params)

        def objective(trial):
            params = {}
            for path, space in optuna_params.items():
                if space["type"] == "float":
                    val = trial.suggest_float(path, space["low"], space["high"], log=space["log"])
                elif space["type"] == "int":
                    val = trial.suggest_int(path, space["low"], space["high"], log=space["log"])
                elif space["type"] == "categorical":
                    val = trial.suggest_categorical(path, space["choices"])
                else:
                    raise ValueError(f"Unknown parameter type: {space['type']}")
                params[path] = val

            if idr_torch.world_size > 1:
                dist.broadcast_object_list([True, params, trial.number], src=0)

            trial_cfg = copy.deepcopy(cfg_copy)
            for path, val in params.items():
                OmegaConf.update(trial_cfg, path, val)

            print(f"Trial {trial.number} starting with params: {params}")
            return evaluate_trial(trial_cfg, trial.number, global_data_loaders, output_dir)

        try:
            study.optimize(objective, n_trials=n_trials)
        finally:
            cleanup_loaders(global_data_loaders)

        if idr_torch.world_size > 1:
            dist.broadcast_object_list([False, None, -1], src=0)

        _write_trials_csv(study, output_dir / "optuna_trials.csv")
        best_trial_dir = _trial_dir(output_dir, study.best_trial.number)
        best_summary = {
            "study_name": study.study_name,
            "dataset": str(cfg_copy.task.name),
            "training_mode": str(cfg_copy.training_mode),
            "seed": int(cfg_copy.seed),
            "best_trial_number": int(study.best_trial.number),
            "best_value": float(study.best_trial.value),
            "best_params": study.best_trial.params,
            "best_trial_dir": str(best_trial_dir),
            "best_checkpoint": str(best_trial_dir / "model_best.pth"),
            "metrics_csv": str(best_trial_dir / "metrics.csv"),
            "trials_csv": str(output_dir / "optuna_trials.csv"),
        }
        (output_dir / "best_trial_summary.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")

        print("=== Best trial ===")
        print(f"  Number: {study.best_trial.number}")
        print(f"  Value: {study.best_trial.value}")
        print("  Params:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

    else:
        try:
            while True:
                signal_list = [None, None, None]
                dist.broadcast_object_list(signal_list, src=0)
                continue_flag, params, trial_number = signal_list

                if not continue_flag:
                    break

                trial_cfg = copy.deepcopy(cfg_copy)
                assert params is not None
                for path, val in params.items():
                    OmegaConf.update(trial_cfg, path, val)

                evaluate_trial(trial_cfg, int(trial_number), global_data_loaders, output_dir)
        finally:
            cleanup_loaders(global_data_loaders)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
