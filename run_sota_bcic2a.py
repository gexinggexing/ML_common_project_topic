#!/usr/bin/env python3
"""SOTA-first BCIC2A FBCSP-plus adaptation sweep."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from run_targeted_bcic_seed_sweep import (
    DATA_ROOT,
    DatasetMeta,
    fbcsp_features,
    load_arrays,
    load_meta,
    logpower_features,
)
from sota_common import (
    RESULTS_DIR,
    SUBMISSION_DIR,
    metric_row,
    selected_row,
    set_seed,
    validate_predictions,
    write_csv,
    write_note,
    write_predictions,
)


DATASET = "BCIC2A"
OUTPUTS_DIR = Path("outputs/sota_bcic2a")
BANKS = {
    "fbcsp_4_40_step4": [(low, low + 4) for low in range(4, 40, 4)],
    "fbcsp_8_30_mu_beta": [(8, 30)],
    "fbcsp_wide_overlap": [(4, 12), (8, 16), (12, 20), (16, 24), (20, 30), (30, 40)],
    "fbcsp_narrow_4_32": [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)],
}
LOGPOWER_BANDS = [(4, 8), (8, 13), (13, 20), (20, 30), (30, 40)]


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


def classifiers(seed: int, k: int | None, fast: bool) -> dict[str, Pipeline]:
    estimators = {
        "lda_mibif": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "svm_rbf_mibif": SVC(C=3.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed),
        "extratrees_mibif": ExtraTreesClassifier(
            n_estimators=200 if fast else 700,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    out = {}
    for name, estimator in estimators.items():
        steps = [("scaler", StandardScaler())]
        if k:
            steps.append(("mibif", SelectKBest(mutual_info_classif, k=k)))
        steps.append(("clf", estimator))
        out[name] = Pipeline(steps)
    return out


def window_view(x: np.ndarray, window: str) -> np.ndarray:
    if window == "full4s":
        return x
    if window == "middle2s":
        return x[..., 200:600]
    if window == "first3s":
        return x[..., :600]
    if window == "last3s":
        return x[..., 200:]
    raise ValueError(window)


def feature_cache(
    meta: DatasetMeta,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    *,
    fast: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
    windows = ["full4s"] if fast else ["full4s", "middle2s", "first3s", "last3s"]
    banks = {"fbcsp_wide_overlap": BANKS["fbcsp_wide_overlap"]} if fast else BANKS
    out = {}
    for window in windows:
        w_train = window_view(train_x, window)
        w_val = window_view(val_x, window)
        w_test = window_view(test_x, window)
        local_meta = DatasetMeta(
            name=meta.name,
            num_classes=meta.num_classes,
            category_list=meta.category_list,
            channels=meta.channels,
            n_channels=meta.n_channels,
            n_times=w_train.shape[-1],
            fs=meta.fs,
        )
        for bank_name, bands in banks.items():
            for csp in ([2] if fast else [1, 2, 3]):
                x_train, (x_val, x_test), details = fbcsp_features(
                    w_train,
                    train_y,
                    [w_val, w_test],
                    local_meta,
                    bands=bands,
                    components_per_side=csp,
                )
                key = f"{bank_name}_csp{csp}_{window}"
                out[key] = (x_train, x_val, x_test, details)
        if not fast:
            x_train, (x_val, x_test), details = logpower_features(
                w_train, [w_val, w_test], local_meta, LOGPOWER_BANDS
            )
            out[f"logpower_{window}"] = (x_train, x_val, x_test, details)
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    meta = load_meta(args.data_root, DATASET)
    arrays = load_arrays(args.data_root, DATASET)
    train_x, train_y = arrays["train"]
    val_x, val_y = arrays["val"]
    test_x, _ = arrays["test"]
    if train_y is None or val_y is None:
        raise ValueError("BCIC2A train/val labels required")
    features = feature_cache(meta, train_x, train_y, val_x, test_x, fast=args.fast_smoke)
    rows = []
    for name, (x_train, x_val, _x_test, details) in features.items():
        k_values = [None] if args.fast_smoke else [None, min(24, x_train.shape[1]), min(48, x_train.shape[1])]
        for k in k_values:
            for clf_name, clf in classifiers(args.seed, k, args.fast_smoke).items():
                clf.fit(x_train, train_y)
                pred = clf.predict(x_val).astype(np.int64)
                row = {
                    "dataset": DATASET,
                    "combo": f"{name}_{clf_name}_k{k or 'all'}",
                    "feature_family": name,
                    "classifier": clf_name,
                    "feature_dim": int(x_train.shape[1]),
                    "mibif_k": "" if k is None else int(k),
                    "feature_details": json.dumps(details),
                    "paper_family": "FBCSP-plus/MIBIF adaptation",
                }
                row.update(metric_row(val_y, pred, meta.num_classes))
                rows.append(row)
    best = selected_row(rows)
    x_train, x_val, x_test, _ = features[str(best["feature_family"])]
    trainval_x = np.concatenate([x_train, x_val], axis=0)
    trainval_y = np.concatenate([train_y, val_y], axis=0)
    k_raw = best.get("mibif_k")
    k = None if k_raw in (None, "") else int(k_raw)
    clf = classifiers(args.seed, k, args.fast_smoke)[str(best["classifier"])]
    clf.fit(trainval_x, trainval_y)
    pred = clf.predict(x_test).astype(np.int64)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / "sota_bcic2a_metrics.csv"
    selection_path = args.results_dir / "sota_bcic2a_selection_summary.csv"
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
        Path("docs/sota_bcic2a_adaptation_note.md"),
        "BCIC2A SOTA-first adaptation note",
        [
            "- Source inspiration: original FBCSP + MIBIF BCI Competition IV Dataset 2a line.",
            "- Faithful part: filter-bank CSP, multi-class OVR CSP, mutual-information feature selection.",
            "- Adapted part: official subject/session evaluation is converted to pooled course train/val/test split.",
            "- Not reproduced: EOG channels, official session-specific train/eval protocol, full FBCNet/EEG Conformer training.",
        ],
    )
    print(json.dumps({"best": selection, "metrics_path": str(metrics_path)}, indent=2))


if __name__ == "__main__":
    main()
