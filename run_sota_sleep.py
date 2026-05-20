#!/usr/bin/env python3
"""SOTA-first ISRUC-SLEEP EEG-only adaptation sweep."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from sota_common import (
    DATA_ROOT,
    RESULTS_DIR,
    SUBMISSION_DIR,
    covariance_features,
    evaluate_classifiers,
    flatten_features,
    load_arrays,
    load_meta,
    logvar_features,
    selected_row,
    set_seed,
    time_stats_features,
    train_predict,
    validate_predictions,
    welch_bandpower,
    write_csv,
    write_note,
    write_predictions,
)


DATASET = "SLEEP"
OUTPUTS_DIR = Path("outputs/sota_sleep")
MIXSLEEP_BANDS = [(0.5, 4), (2, 6), (4, 8), (6, 11), (8, 14), (11, 22), (14, 31), (22, 35)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--submission-dir", type=Path, default=SUBMISSION_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--update-submission", action="store_true")
    parser.add_argument("--fast-smoke", action="store_true")
    return parser.parse_args()


def downsample_stats(x: np.ndarray, bins: int = 30) -> np.ndarray:
    n = x.shape[-1]
    trimmed = x[..., : (n // bins) * bins]
    chunk = trimmed.reshape(x.shape[0], x.shape[1], bins, -1)
    return np.concatenate([chunk.mean(axis=-1), chunk.std(axis=-1)], axis=1).reshape(x.shape[0], -1)


def feature_sets(x: np.ndarray, fs: float, *, fast: bool) -> dict[str, np.ndarray]:
    band = welch_bandpower(x, fs, MIXSLEEP_BANDS, relative=True)
    sets = {
        "sleep_mixsleep_bandpower": flatten_features(band),
        "sleep_mixsleep_band_stats": np.concatenate([flatten_features(band), logvar_features(x)], axis=1),
    }
    if not fast:
        sets["sleep_hass_window_stats"] = np.concatenate(
            [flatten_features(band), time_stats_features(x), downsample_stats(x, bins=30)], axis=1
        )
        sets["sleep_band_cov_stats"] = np.concatenate(
            [flatten_features(band), covariance_features(x), time_stats_features(x)], axis=1
        )
    return sets


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    meta = load_meta(args.data_root, DATASET)
    arrays = load_arrays(args.data_root, DATASET)
    train_x, train_y = arrays["train"]
    val_x, val_y = arrays["val"]
    test_x, _ = arrays["test"]
    if train_y is None or val_y is None:
        raise ValueError("SLEEP train/val labels required")
    train_features = feature_sets(train_x, meta.fs, fast=args.fast_smoke)
    val_features = feature_sets(val_x, meta.fs, fast=args.fast_smoke)
    test_features = feature_sets(test_x, meta.fs, fast=args.fast_smoke)
    rows = []
    for name in train_features:
        rows.extend(
            evaluate_classifiers(
                name,
                train_features[name],
                train_y,
                val_features[name],
                val_y,
                meta.num_classes,
                seed=args.seed,
                fast=args.fast_smoke,
                extra={"dataset": DATASET, "paper_family": "BSTT/HASS/MixSleepNet EEG-only adaptation"},
            )
        )
    best = selected_row(rows)
    best_name = str(best["feature_family"])
    trainval_x = np.concatenate([train_features[best_name], val_features[best_name]], axis=0)
    trainval_y = np.concatenate([train_y, val_y], axis=0)
    pred = train_predict(best, trainval_x, trainval_y, test_features[best_name], seed=args.seed, fast=args.fast_smoke)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / "sota_sleep_metrics.csv"
    selection_path = args.results_dir / "sota_sleep_selection_summary.csv"
    output_path = args.outputs_dir / f"{DATASET}.txt"
    write_predictions(output_path, pred)
    validate_predictions(output_path, DATASET)
    selection = dict(best)
    selection["selected_output"] = str(output_path)
    selection["updated_submission"] = ""
    if args.update_submission:
        dest = args.submission_dir / f"{DATASET}.txt"
        shutil.copy2(output_path, dest)
        validate_predictions(dest, DATASET)
        selection["updated_submission"] = str(dest)
    write_csv(metrics_path, rows)
    metrics_path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(selection_path, [selection])
    selection_path.with_suffix(".json").write_text(json.dumps([selection], indent=2), encoding="utf-8")
    write_note(
        Path("docs/sota_sleep_adaptation_note.md"),
        "SLEEP SOTA-first adaptation note",
        [
            "- Source inspiration: ISRUC BSTT/HASS and EEG-only MixSleepNet-style multiband features.",
            "- Faithful part: 6 EEG channels, 30-second epochs, 5 sleep stages, EEG-only feature extraction.",
            "- Adapted part: temporal-context models are converted to independent-window classifiers.",
            "- Not reproduced: EOG/EMG/ECG modalities and adjacent-epoch sequence context.",
        ],
    )
    print(json.dumps({"best": selection, "metrics_path": str(metrics_path)}, indent=2))


if __name__ == "__main__":
    main()
