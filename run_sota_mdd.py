#!/usr/bin/env python3
"""SOTA-first MDD Mumtaz/Deep-Asymmetry adaptation sweep."""

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


DATASET = "MDD"
OUTPUTS_DIR = Path("outputs/sota_mdd")
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
ALPHA_PAIRS = [
    ("FP1", "FP2"),
    ("F3", "F4"),
    ("C3", "C4"),
    ("P3", "P4"),
    ("O1", "O2"),
    ("F7", "F8"),
    ("T3", "T4"),
    ("T5", "T6"),
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


def feature_sets(x: np.ndarray, channels: list[str], fs: float, *, fast: bool) -> dict[str, np.ndarray]:
    x19 = x[:, :19, :]
    channels19 = channels[:19]
    band = welch_bandpower(x19, fs, BANDS, relative=True)
    alpha = band[:, :, 2:3]
    asym = asymmetry_from_band_features(alpha, channels19, ALPHA_PAIRS)
    base = np.concatenate([flatten_features(band), asym, logvar_features(x19)], axis=1)
    sets = {
        "mdd_mumtaz_band_alpha_asym": base,
        "mdd_mumtaz_band_stats": np.concatenate([base, time_stats_features(x19)], axis=1),
    }
    if not fast:
        upper = covariance_features(x19)
        asym_matrix = []
        alpha2 = alpha[:, :, 0]
        for row in alpha2:
            asym_matrix.append((row[:, None] - row[None, :]).reshape(-1))
        sets["mdd_deep_asymmetry_flat"] = np.asarray(asym_matrix, dtype=np.float64)
        sets["mdd_band_cov_stats"] = np.concatenate([base, upper, time_stats_features(x19)], axis=1)
        sets["mdd_20ch_stats_band"] = np.concatenate(
            [flatten_features(welch_bandpower(x, fs, BANDS, relative=True)), time_stats_features(x)], axis=1
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
        raise ValueError("MDD train/val labels required")
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
                extra={"dataset": DATASET, "paper_family": "Mumtaz/Deep-Asymmetry adaptation"},
            )
        )
    best = selected_row(rows)
    best_name = str(best["feature_family"])
    trainval_x = np.concatenate([train_features[best_name], val_features[best_name]], axis=0)
    trainval_y = np.concatenate([train_y, val_y], axis=0)
    pred = train_predict(best, trainval_x, trainval_y, test_features[best_name], seed=args.seed, fast=args.fast_smoke)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / "sota_mdd_metrics.csv"
    selection_path = args.results_dir / "sota_mdd_selection_summary.csv"
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
        Path("docs/sota_mdd_adaptation_note.md"),
        "MDD SOTA-first adaptation note",
        [
            "- Source inspiration: Mumtaz bandpower/asymmetry CAD and Deep-Asymmetry.",
            "- Faithful part: relative bandpower, alpha asymmetry, and asymmetry matrix features.",
            "- Adapted part: 5-minute/5-second subject-aware protocols are reduced to 1-second window classifiers.",
            "- Not reproduced: LOSO split, EC/EO/P300 separation, source subject/session metadata.",
        ],
    )
    print(json.dumps({"best": selection, "metrics_path": str(metrics_path)}, indent=2))


if __name__ == "__main__":
    main()
