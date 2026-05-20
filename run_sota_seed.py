#!/usr/bin/env python3
"""SOTA-first SEED DE/PSD and graph-inspired adaptation sweep."""

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
    asymmetry_from_band_features,
    covariance_features,
    evaluate_classifiers,
    flatten_features,
    load_arrays,
    load_meta,
    selected_row,
    set_seed,
    train_predict,
    validate_predictions,
    welch_bandpower,
    write_csv,
    write_note,
    write_predictions,
)


DATASET = "SEED"
OUTPUTS_DIR = Path("outputs/sota_seed")
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


def one_second_band_features(x: np.ndarray, fs: float) -> np.ndarray:
    if x.shape[-1] != 400:
        return welch_bandpower(x, fs, SEED_BANDS, relative=False)[:, :, None, :]
    parts = []
    for start in (0, 200):
        parts.append(welch_bandpower(x[..., start : start + 200], fs, SEED_BANDS, relative=False))
    return np.stack(parts, axis=2)


def feature_sets(x: np.ndarray, channels: list[str], fs: float, *, fast: bool) -> dict[str, np.ndarray]:
    band_win = one_second_band_features(x, fs)
    band_flat = band_win.reshape(x.shape[0], -1)
    mean_band = band_win.mean(axis=2)
    asym = asymmetry_from_band_features(mean_band, channels, SEED_PAIR_NAMES)
    sets = {
        "seed_de_psd_2x1s": band_flat,
        "seed_de_asym_2x1s": np.concatenate([band_flat, asym], axis=1),
    }
    if not fast:
        node_cov = covariance_features(mean_band.reshape(x.shape[0], x.shape[1], -1), corr=True)
        sets["seed_dgcnn_lite_graph_features"] = np.concatenate([band_flat, asym, node_cov], axis=1)
        beta_gamma = band_win[..., 3:].reshape(x.shape[0], -1)
        sets["seed_beta_gamma_graph_features"] = np.concatenate([beta_gamma, asym], axis=1)
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
        raise ValueError("SEED train/val labels required")
    train_features = feature_sets(train_x, meta.channels, meta.fs, fast=args.fast_smoke)
    val_features = feature_sets(val_x, meta.channels, meta.fs, fast=args.fast_smoke)
    test_features = feature_sets(test_x, meta.channels, meta.fs, fast=args.fast_smoke)
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
                extra={"dataset": DATASET, "paper_family": "DE/PSD DGCNN/RGNN-lite adaptation"},
            )
        )
    best = selected_row(rows)
    best_name = str(best["feature_family"])
    trainval_x = np.concatenate([train_features[best_name], val_features[best_name]], axis=0)
    trainval_y = np.concatenate([train_y, val_y], axis=0)
    pred = train_predict(best, trainval_x, trainval_y, test_features[best_name], seed=args.seed, fast=args.fast_smoke)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / "sota_seed_metrics.csv"
    selection_path = args.results_dir / "sota_seed_selection_summary.csv"
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
        Path("docs/sota_seed_adaptation_note.md"),
        "SEED SOTA-first adaptation note",
        [
            "- Source inspiration: SEED DE/PSD baselines, DGCNN, RGNN, and 4D-aNN-style graph features.",
            "- Faithful part: five-band DE/log-power features over two 1-second subwindows and 62 EEG nodes.",
            "- Adapted part: graph networks are approximated by graph/covariance feature baselines in this first pass.",
            "- Not reproduced: subject/session/trial-aware evaluation, LDS smoothing, domain adaptation, row-order templates.",
        ],
    )
    print(json.dumps({"best": selection, "metrics_path": str(metrics_path)}, indent=2))


if __name__ == "__main__":
    main()
