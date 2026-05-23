from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "our_tasks" / "resource_paths.yaml"
DT_OPT_SCRIPT_PATH = PROJECT_ROOT / "src" / "dt_opt.py"
PREDICT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "our_tasks" / "predict_test.py"
DEFAULT_WINDOWS_PYTHON = Path(r"D:\app\conda_envs\brain-dl\python.exe")
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "course_project_optuna"
COURSE_OPTUNA_CONFIGS = {
    "BCIC2A": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_bcic2a_optuna.yaml",
    "CHINESE": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_chinese_optuna.yaml",
    "MDD": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_mdd_optuna.yaml",
    "SEED": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_seed_optuna.yaml",
    "SLEEP": PROJECT_ROOT / "src" / "configs" / "our_tasks" / "course_sleep_optuna.yaml",
}
SUPPORTED_MODES = {"lp", "mlp", "ft"}
DEFAULT_BATCH_SIZES = {
    "BCIC2A": 16,
    "CHINESE": 16,
    "MDD": 16,
    "SEED": 8,
    "SLEEP": 2,
}
DEFAULT_TRIALS = {
    "BCIC2A": {"lp": 16, "mlp": 16, "ft": 12},
    "CHINESE": {"lp": 12, "mlp": 12, "ft": 10},
    "MDD": {"lp": 14, "mlp": 14, "ft": 10},
    "SEED": {"lp": 16, "mlp": 14, "ft": 10},
    "SLEEP": {"lp": 10, "mlp": 8, "ft": 6},
}
MODE_HEAD_DEFAULTS = {
    "lp": {"head_type": "linear"},
    "mlp": {"head_type": "mlp"},
    "ft": {"head_type": "linear"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna-based course-project jobs through REVE dt_opt.py.")
    parser.add_argument("--resource", type=str, required=True, help="Resource key in resource_paths.yaml.")
    parser.add_argument("--data-root", type=str, default=None, help="Override data root.")
    parser.add_argument("--ckpt-root", type=str, default=None, help="Override checkpoint root.")
    parser.add_argument("--out-root", type=str, default=None, help="Override output root.")
    parser.add_argument("--datasets", nargs="+", default=["SEED"], help="Datasets to run.")
    parser.add_argument("--modes", nargs="+", default=["lp"], help="Modes to run.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds to run.")
    parser.add_argument("--trials", type=int, default=None, help="Override Optuna trial count for every study.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device override passed to dt_opt.py.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override.")
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


def resolve_batch_size(args: argparse.Namespace, dataset: str) -> int:
    if args.batch_size is not None:
        return int(args.batch_size)
    return int(DEFAULT_BATCH_SIZES.get(dataset, 8))


def resolve_trial_count(args: argparse.Namespace, dataset: str, mode: str) -> int:
    if args.trials is not None:
        return int(args.trials)
    return int(DEFAULT_TRIALS[dataset][mode])


def build_training_command(
    python_exe: Path,
    task_config_path: Path,
    data_root: Path,
    ckpt_root: Path,
    output_dir: Path,
    dataset: str,
    mode: str,
    seed: int,
    device: str,
    batch_size: int,
    num_workers: int,
    n_trials: int,
) -> list[str]:
    pretrained_dir = ckpt_root / "reve-base"
    if not pretrained_dir.exists():
        raise FileNotFoundError(f"Pretrained REVE directory does not exist: {pretrained_dir}")

    command = [
        str(python_exe),
        "-u",
        str(DT_OPT_SCRIPT_PATH),
        f"+task.name={hydra_quote(dataset)}",
        f"+task_config_path={hydra_quote(hydra_path(task_config_path))}",
        f"+data_root={hydra_quote(data_root)}",
        f"+ckpt_root={hydra_quote(hydra_path(ckpt_root))}",
        f"+batch_size={hydra_quote(batch_size)}",
        f"+skip_test_eval=true",
        f"+compact_console_logging=true",
        f"training_mode={hydra_quote(mode)}",
        f"seed={seed}",
        f"n_runs={n_trials}",
        f"trainer.device={hydra_quote(device)}",
        f"loader.num_workers={num_workers}",
        f"loader.pin_memory={'true' if device.startswith('cuda') else 'false'}",
        f"loader.persistent_workers={'true' if num_workers > 0 else 'false'}",
        "mode=train",
        f"+head_type={hydra_quote(MODE_HEAD_DEFAULTS[mode]['head_type'])}",
        f"pretrained_path={hydra_quote(hydra_path(pretrained_dir))}",
        f"cache_dir={hydra_quote(hydra_path(PROJECT_ROOT / '.cache'))}",
        f"output_dir={hydra_quote(hydra_path(output_dir))}",
        "hydra.job.chdir=false",
        f"hydra.run.dir={hydra_quote(hydra_path(output_dir))}",
    ]
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
    checkpoint: Path,
    device: str,
    batch_size: int,
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
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
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


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.datasets = normalize_multi_values(args.datasets)
    args.modes = normalize_multi_values(args.modes)
    runtime = resolve_runtime(args)
    out_root = Path(runtime["out_root"])
    had_failure = False

    for dataset in args.datasets:
        if dataset not in COURSE_OPTUNA_CONFIGS:
            available = ", ".join(sorted(COURSE_OPTUNA_CONFIGS))
            raise KeyError(f"Dataset '{dataset}' is not supported. Available: {available}")
        task_config_path = COURSE_OPTUNA_CONFIGS[dataset]
        if not task_config_path.exists():
            raise FileNotFoundError(f"Course Optuna task config does not exist: {task_config_path}")

        batch_size = resolve_batch_size(args, dataset)
        num_workers = int(runtime["default_num_workers"])

        for mode in args.modes:
            if mode not in SUPPORTED_MODES:
                allowed = ", ".join(sorted(SUPPORTED_MODES))
                raise ValueError(f"Unsupported mode '{mode}'. Allowed: {allowed}")

            n_trials = resolve_trial_count(args, dataset, mode)
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
                    mode=mode,
                    seed=seed,
                    device=args.device,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    n_trials=n_trials,
                )

                metadata = {
                    "resource": args.resource,
                    "dataset": dataset,
                    "mode": mode,
                    "seed": seed,
                    "n_trials": n_trials,
                    "device": args.device,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "python_exe": str(runtime["python_exe"]),
                    "data_root": str(runtime["data_root"]),
                    "ckpt_root": str(runtime["ckpt_root"]),
                    "out_root": str(out_root),
                    "run_dir": to_rel_or_abs(output_dir),
                    "task_config_path": to_rel_or_abs(task_config_path),
                    "training_command": command,
                }
                write_run_metadata(output_dir, metadata)
                (output_dir / "command.txt").write_text(subprocess.list2cmdline(command), encoding="utf-8")

                print(
                    f"========== START_OPTUNA dataset={dataset} mode={mode} seed={seed} "
                    f"trials={n_trials} device={args.device} ==========",
                    flush=True,
                )
                exit_code = run_subprocess(command, output_dir / "train.log")
                if exit_code != 0:
                    print(
                        f"========== FAILED_OPTUNA dataset={dataset} mode={mode} seed={seed} "
                        f"see {to_rel_or_abs(output_dir / 'train.log')} ==========",
                        flush=True,
                    )
                    had_failure = True
                    continue

                best_summary = load_json_if_exists(output_dir / "best_trial_summary.json")
                test_pred_path = ""
                best_value = None
                if best_summary is not None:
                    best_value = best_summary.get("best_value")
                    checkpoint_path = resolve_path(str(best_summary["best_checkpoint"]))
                    predict_command = build_predict_command(
                        python_exe=Path(runtime["python_exe"]),
                        data_root=Path(runtime["data_root"]),
                        ckpt_root=Path(runtime["ckpt_root"]),
                        out_root=out_root,
                        dataset=dataset,
                        mode=mode,
                        seed=seed,
                        run_dir=output_dir,
                        checkpoint=checkpoint_path,
                        device=args.device,
                        batch_size=batch_size,
                        num_workers=num_workers,
                    )
                    (output_dir / "predict_command.txt").write_text(
                        subprocess.list2cmdline(predict_command),
                        encoding="utf-8",
                    )
                    predict_exit_code = run_subprocess(predict_command, output_dir / "predict_test.log")
                    if predict_exit_code != 0:
                        print(
                            f"WARNING: predict_test.py failed for optuna {dataset}/{mode}/seed{seed} "
                            f"with exit code {predict_exit_code}",
                            flush=True,
                        )
                        had_failure = True
                    else:
                        prediction_path = output_dir / "test_predictions.csv"
                        if prediction_path.exists():
                            test_pred_path = to_rel_or_abs(prediction_path)
                else:
                    print(
                        f"WARNING: best_trial_summary.json not found for {dataset}/{mode}/seed{seed}; skip test prediction",
                        flush=True,
                    )
                    had_failure = True

                best_str = "nan" if best_value is None else f"{float(best_value):.3f}"
                pred_str = test_pred_path if test_pred_path else "missing"
                print(
                    f"========== DONE_OPTUNA dataset={dataset} mode={mode} seed={seed} "
                    f"best_val_bal_acc={best_str} test_pred={pred_str} ==========",
                    flush=True,
                )

    if had_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
