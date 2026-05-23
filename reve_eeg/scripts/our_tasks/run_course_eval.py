from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "our_tasks" / "resource_paths.yaml"
DT_SCRIPT_PATH = PROJECT_ROOT / "src" / "dt.py"
PREDICT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "our_tasks" / "predict_test.py"
DEFAULT_WINDOWS_PYTHON = Path(r"D:\app\conda_envs\brain-dl\python.exe")
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "course_project_eval_seed42to46_v1"
COURSE_CONFIGS = {
    "BCIC2A": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_bcic2a.yaml",
    "MDD": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_mdd.yaml",
    "SEED": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_seed.yaml",
    "SLEEP": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_sleep.yaml",
}

# Current formal-eval shortlist after excluding CHINESE.
FORMAL_RUN_CONFIGS: dict[tuple[str, str], dict[str, Any]] = {
    ("BCIC2A", "ft"): {
        "training_mode": "ft",
        "batch_size": 32,
        "predict_batch_size": 64,
        "n_epochs": 30,
        "patience": 8,
        "lr": 2.612533879325072e-4,
        "weight_decay": 1.4941899384581658e-6,
        "head_type": "linear",
        "classifier_dropout": 0.26395978795727476,
        "clip_grad": 1.0,
        "encoder_lr": 1.705827635043124e-5,
        "head_lr": 2.612533879325072e-4,
    },
    ("MDD", "ft"): {
        "training_mode": "ft",
        "batch_size": 64,
        "predict_batch_size": 64,
        "n_epochs": 30,
        "patience": 8,
        "lr": 1.0e-4,
        "weight_decay": 1.0e-4,
        "head_type": "linear",
        "classifier_dropout": 0.1,
        "clip_grad": 1.0,
        "encoder_lr": 1.0e-5,
        "head_lr": 1.0e-4,
    },
    ("SEED", "ft"): {
        "training_mode": "ft",
        "batch_size": 16,
        "predict_batch_size": 16,
        "n_epochs": 24,
        "patience": 6,
        "lr": 5.1634273195154156e-5,
        "weight_decay": 1.1674975458081493e-3,
        "head_type": "linear",
        "classifier_dropout": 0.2074662667109422,
        "clip_grad": 1.0,
        "encoder_lr": 1.7296256273525447e-6,
        "head_lr": 5.1634273195154156e-5,
    },
    ("SEED", "lp"): {
        "training_mode": "lp",
        "batch_size": 32,
        "predict_batch_size": 32,
        "n_epochs": 30,
        "patience": 8,
        "lr": 8.0e-4,
        "weight_decay": 1.0e-4,
        "head_type": "linear",
        "classifier_dropout": 0.1,
        "clip_grad": 2.0,
    },
    ("SLEEP", "ft"): {
        "training_mode": "ft",
        "batch_size": 4,
        "predict_batch_size": 4,
        "n_epochs": 12,
        "patience": 4,
        "lr": 5.066407974570349e-5,
        "weight_decay": 9.52808052852129e-3,
        "head_type": "linear",
        "classifier_dropout": 0.24779275449905408,
        "clip_grad": 1.0,
        "encoder_lr": 2.439555677271592e-6,
        "head_lr": 5.066407974570349e-5,
    },
}
SUPPORTED_DATASETS = sorted(COURSE_CONFIGS)
SUPPORTED_MODES = sorted({mode for _, mode in FORMAL_RUN_CONFIGS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal multi-seed evaluation with selected best configs.")
    parser.add_argument("--resource", type=str, required=True, help="Resource key in resource_paths.yaml.")
    parser.add_argument("--data-root", type=str, default=None, help="Override data root.")
    parser.add_argument("--ckpt-root", type=str, default=None, help="Override checkpoint root.")
    parser.add_argument("--out-root", type=str, default=None, help="Override output root.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Datasets to run.")
    parser.add_argument("--modes", nargs="+", required=True, help="Modes to run.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Random seeds to run.")
    parser.add_argument("--device", type=str, default="cuda", help="Device override passed to dt.py.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional global training batch size override.")
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=None,
        help="Optional global inference batch size override.",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader num_workers override.")
    parser.add_argument("--python-exe", type=str, default=None, help="Python executable used to launch scripts.")
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


def hydra_quote(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def hydra_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def normalize_multi_values(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        normalized.extend(part.strip() for part in str(value).split(",") if part.strip())
    return normalized


def resolve_runtime(args: argparse.Namespace) -> dict[str, object]:
    resource_configs = load_resource_configs()
    if args.resource not in resource_configs:
        available = ", ".join(sorted(resource_configs))
        raise KeyError(f"Unknown resource '{args.resource}'. Available resources: {available}")

    resource_cfg = resource_configs[args.resource]
    data_root = resolve_path(args.data_root or str(resource_cfg["data_root"]))
    ckpt_root = resolve_path(args.ckpt_root or str(resource_cfg["ckpt_root"]))
    out_root = resolve_path(args.out_root or str(DEFAULT_OUT_ROOT))

    if args.python_exe is not None:
        python_exe = Path(args.python_exe)
    elif args.resource == "5080" and DEFAULT_WINDOWS_PYTHON.exists():
        python_exe = DEFAULT_WINDOWS_PYTHON
    else:
        python_exe = Path(sys.executable)

    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable does not exist: {python_exe}")

    return {
        "resource_cfg": resource_cfg,
        "data_root": data_root,
        "ckpt_root": ckpt_root,
        "out_root": out_root,
        "python_exe": python_exe,
        "default_num_workers": args.num_workers if args.num_workers is not None else int(resource_cfg["num_workers"]),
    }


def resolve_training_config(args: argparse.Namespace, dataset: str, mode: str) -> dict[str, Any]:
    key = (dataset, mode)
    if key not in FORMAL_RUN_CONFIGS:
        available = ", ".join(f"{d}/{m}" for d, m in sorted(FORMAL_RUN_CONFIGS))
        raise KeyError(f"Unsupported formal-eval combo '{dataset}/{mode}'. Available: {available}")

    config = dict(FORMAL_RUN_CONFIGS[key])
    if args.batch_size is not None:
        config["batch_size"] = int(args.batch_size)
    if args.predict_batch_size is not None:
        config["predict_batch_size"] = int(args.predict_batch_size)
    return config


def build_training_command(
    python_exe: Path,
    task_config_path: Path,
    data_root: Path,
    ckpt_root: Path,
    output_dir: Path,
    dataset: str,
    seed: int,
    device: str,
    num_workers: int,
    run_config: dict[str, Any],
) -> list[str]:
    pretrained_dir = ckpt_root / "reve-base"
    if not pretrained_dir.exists():
        raise FileNotFoundError(f"Pretrained REVE directory does not exist: {pretrained_dir}")

    command = [
        str(python_exe),
        "-u",
        str(DT_SCRIPT_PATH),
        f"+task.name={hydra_quote(dataset)}",
        f"+task_config_path={hydra_quote(hydra_path(task_config_path))}",
        f"+data_root={hydra_quote(data_root)}",
        f"+ckpt_root={hydra_quote(hydra_path(ckpt_root))}",
        f"+batch_size={hydra_quote(int(run_config['batch_size']))}",
        f"+n_epochs={hydra_quote(int(run_config['n_epochs']))}",
        f"+patience={hydra_quote(int(run_config['patience']))}",
        f"+lr={hydra_quote(float(run_config['lr']))}",
        f"+weight_decay={hydra_quote(float(run_config['weight_decay']))}",
        "+skip_test_eval=true",
        "+compact_console_logging=true",
        f"training_mode={hydra_quote(run_config['training_mode'])}",
        f"seed={seed}",
        f"trainer.device={hydra_quote(device)}",
        f"trainer.clip_grad={hydra_quote(float(run_config['clip_grad']))}",
        f"loader.num_workers={num_workers}",
        f"loader.pin_memory={'true' if device.startswith('cuda') else 'false'}",
        f"loader.persistent_workers={'true' if num_workers > 0 else 'false'}",
        "mode=train",
        f"+head_type={hydra_quote(run_config['head_type'])}",
        f"+classifier_dropout={hydra_quote(float(run_config['classifier_dropout']))}",
        f"pretrained_path={hydra_quote(hydra_path(pretrained_dir))}",
        f"cache_dir={hydra_quote(hydra_path(PROJECT_ROOT / '.cache'))}",
        f"output_dir={hydra_quote(hydra_path(output_dir))}",
        "hydra.job.chdir=false",
        f"hydra.run.dir={hydra_quote(hydra_path(output_dir))}",
        "+metrics_csv_path=metrics.csv",
        "+tensorboard_dir=tensorboard",
    ]
    if "mlp_hidden_dim" in run_config:
        command.append(f"+mlp_hidden_dim={hydra_quote(int(run_config['mlp_hidden_dim']))}")
    if "encoder_lr" in run_config:
        command.append(f"+encoder_lr={hydra_quote(float(run_config['encoder_lr']))}")
    if "head_lr" in run_config:
        command.append(f"+head_lr={hydra_quote(float(run_config['head_lr']))}")
    return command


def build_predict_command(
    python_exe: Path,
    data_root: Path,
    ckpt_root: Path,
    out_root: Path,
    dataset: str,
    mode: str,
    seed: int,
    run_dir: Path,
    device: str,
    predict_batch_size: int,
    num_workers: int,
) -> list[str]:
    return [
        str(python_exe),
        "-u",
        str(PREDICT_SCRIPT_PATH),
        "--resource",
        "manual",
        "--data-root",
        str(data_root),
        "--ckpt-root",
        str(ckpt_root),
        "--out-root",
        str(out_root),
        "--dataset",
        dataset,
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--run-dir",
        str(run_dir),
        "--device",
        device,
        "--batch-size",
        str(int(predict_batch_size)),
        "--num-workers",
        str(num_workers),
        "--python-exe",
        str(python_exe),
    ]


def run_subprocess(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            safe_line = line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
                sys.stdout.encoding or "utf-8",
                errors="replace",
            )
            sys.stdout.write(safe_line)
            sys.stdout.flush()
            log_handle.write(line)
        return process.wait()


def to_rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def write_run_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def read_best_val_bal_acc(metrics_csv_path: Path) -> float | None:
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


def main() -> None:
    args = parse_args()
    args.datasets = normalize_multi_values(args.datasets)
    args.modes = normalize_multi_values(args.modes)
    runtime = resolve_runtime(args)
    out_root = Path(runtime["out_root"])
    had_failure = False

    for dataset in args.datasets:
        if dataset not in COURSE_CONFIGS:
            available = ", ".join(SUPPORTED_DATASETS)
            raise KeyError(f"Dataset '{dataset}' is not supported. Available: {available}")
        task_config_path = COURSE_CONFIGS[dataset]

        for mode in args.modes:
            run_config = resolve_training_config(args, dataset, mode)
            num_workers = int(runtime["default_num_workers"])
            predict_batch_size = int(run_config.get("predict_batch_size", run_config["batch_size"]))

            for seed in args.seeds:
                output_dir = out_root / dataset / mode / f"seed{seed}"
                output_dir.mkdir(parents=True, exist_ok=True)

                command = build_training_command(
                    python_exe=Path(runtime["python_exe"]),
                    task_config_path=task_config_path,
                    data_root=Path(runtime["data_root"]),
                    ckpt_root=Path(runtime["ckpt_root"]),
                    output_dir=output_dir,
                    dataset=dataset,
                    seed=seed,
                    device=args.device,
                    num_workers=num_workers,
                    run_config=run_config,
                )

                metadata = {
                    "resource": args.resource,
                    "dataset": dataset,
                    "mode": mode,
                    "seed": seed,
                    "device": args.device,
                    "batch_size": int(run_config["batch_size"]),
                    "predict_batch_size": int(predict_batch_size),
                    "num_workers": num_workers,
                    "python_exe": str(runtime["python_exe"]),
                    "data_root": str(runtime["data_root"]),
                    "ckpt_root": str(runtime["ckpt_root"]),
                    "out_root": str(out_root),
                    "run_dir": to_rel_or_abs(output_dir),
                    "task_config_path": to_rel_or_abs(task_config_path),
                    "training_command": command,
                    "mode_config": run_config,
                }
                write_run_metadata(output_dir, metadata)
                (output_dir / "command.txt").write_text(subprocess.list2cmdline(command), encoding="utf-8")

                print(
                    f"========== START dataset={dataset} mode={mode} seed={seed} device={args.device} ==========",
                    flush=True,
                )
                exit_code = run_subprocess(command, output_dir / "train.log")
                if exit_code != 0:
                    print(
                        f"========== FAILED dataset={dataset} mode={mode} seed={seed} "
                        f"see {to_rel_or_abs(output_dir / 'train.log')} ==========",
                        flush=True,
                    )
                    had_failure = True
                    continue

                best_checkpoint = output_dir / "model_best.pth"
                test_pred_path = ""
                if best_checkpoint.exists():
                    predict_command = build_predict_command(
                        python_exe=Path(runtime["python_exe"]),
                        data_root=Path(runtime["data_root"]),
                        ckpt_root=Path(runtime["ckpt_root"]),
                        out_root=out_root,
                        dataset=dataset,
                        mode=mode,
                        seed=seed,
                        run_dir=output_dir,
                        device=args.device,
                        predict_batch_size=predict_batch_size,
                        num_workers=num_workers,
                    )
                    (output_dir / "predict_command.txt").write_text(
                        subprocess.list2cmdline(predict_command),
                        encoding="utf-8",
                    )
                    predict_exit_code = run_subprocess(predict_command, output_dir / "predict_test.log")
                    if predict_exit_code != 0:
                        print(
                            f"WARNING: predict_test.py failed for {dataset}/{mode}/seed{seed} "
                            f"with exit code {predict_exit_code}",
                            flush=True,
                        )
                        had_failure = True
                    else:
                        prediction_path = output_dir / "test_predictions.csv"
                        if prediction_path.exists():
                            test_pred_path = to_rel_or_abs(prediction_path)
                else:
                    print(f"WARNING: model_best.pth not found for {dataset}/{mode}/seed{seed}; skip test prediction")
                    had_failure = True

                best_val_bal_acc = read_best_val_bal_acc(output_dir / "metrics.csv")
                best_str = "nan" if best_val_bal_acc is None else f"{best_val_bal_acc:.3f}"
                pred_str = test_pred_path if test_pred_path else "missing"
                print(
                    f"========== DONE dataset={dataset} mode={mode} seed={seed} "
                    f"best_val_bal_acc={best_str} test_pred={pred_str} ==========",
                    flush=True,
                )

    if had_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
