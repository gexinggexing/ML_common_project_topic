#!/usr/bin/env python3
"""Finalize SOTA-first course-project runs across all five datasets.

This coordinator is intentionally conservative:
- it reads per-dataset ``sota_*_selection_summary.csv`` files,
- compares each candidate against the current valid baseline,
- validates the selected prediction file shape/range,
- writes a final summary,
- and only updates ``outputs/submission`` when ``--update-submission`` is set.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASETS = ["BCIC2A", "SEED", "CHINESE", "MDD", "SLEEP"]
RESULTS_DIR = Path("artifacts/results")
SUBMISSION_DIR = Path("outputs/submission")
EXPECTED_ROWS = {
    "BCIC2A": 360,
    "SEED": 450,
    "CHINESE": 200,
    "MDD": 800,
    "SLEEP": 1945,
}
NUM_CLASSES = {
    "BCIC2A": 4,
    "SEED": 3,
    "CHINESE": 2,
    "MDD": 2,
    "SLEEP": 5,
}
BASELINES = {
    "BCIC2A": (0.5583333333333333, "FBCSP wide CSP2 + ExtraTrees"),
    "SEED": (0.40444444444444444, "valid normal CBraMod author-style baseline"),
    "CHINESE": (0.625, "CBraMod global_mean_std + MLP"),
    "MDD": (0.9328125, "CBraMod global_mean + MLP"),
    "SLEEP": (0.7252356931281023, "CBraMod channel_flat_pca512 + MLP"),
}


@dataclass(frozen=True)
class Candidate:
    dataset: str
    row: dict[str, Any]
    score: float
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--submission-dir", type=Path, default=SUBMISSION_DIR)
    parser.add_argument("--update-submission", action="store_true")
    parser.add_argument(
        "--allow-equal",
        action="store_true",
        help="Allow replacing submissions when candidate score equals baseline.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_score(row: dict[str, Any]) -> float:
    for key in ("balanced_accuracy", "val_balanced_accuracy", "mean_balanced_accuracy"):
        raw = row.get(key)
        if raw not in (None, ""):
            return float(raw)
    raise ValueError(f"Cannot find balanced accuracy in row keys={sorted(row)}")


def find_output_path(row: dict[str, Any], dataset: str) -> Path:
    for key in ("selected_output", "prediction_path", "output_path"):
        raw = row.get(key)
        if raw:
            return Path(str(raw))
    default = Path(f"outputs/sota_{dataset.lower()}") / f"{dataset}.txt"
    return default


def validate_prediction_file(path: Path, dataset: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    expected = EXPECTED_ROWS[dataset]
    if len(lines) != expected:
        raise ValueError(f"{path}: expected {expected} rows, got {len(lines)}")
    valid = set(range(NUM_CLASSES[dataset]))
    bad: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped != line or not stripped.isdigit() or int(stripped) not in valid:
            bad.append((idx, line))
            if len(bad) >= 5:
                break
    if bad:
        raise ValueError(f"{path}: invalid prediction lines {bad}")


def load_candidate(results_dir: Path, dataset: str) -> Candidate | None:
    slug = dataset.lower()
    path = results_dir / f"sota_{slug}_selection_summary.csv"
    rows = read_rows(path)
    if not rows:
        return None
    scored = sorted(rows, key=find_score, reverse=True)
    row = dict(scored[0])
    score = find_score(row)
    output_path = find_output_path(row, dataset)
    return Candidate(dataset=dataset, row=row, score=score, output_path=output_path)


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
    final_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        baseline_score, baseline_name = BASELINES[dataset]
        candidate = load_candidate(args.results_dir, dataset)
        if candidate is None:
            final_rows.append(
                {
                    "dataset": dataset,
                    "baseline_balanced_accuracy": baseline_score,
                    "baseline_name": baseline_name,
                    "candidate_balanced_accuracy": "",
                    "delta_vs_baseline": "",
                    "candidate_output": "",
                    "candidate_valid": False,
                    "selected_for_submission": False,
                    "submission_updated": False,
                    "reason": "missing selection summary",
                }
            )
            continue

        validate_prediction_file(candidate.output_path, dataset)
        delta = candidate.score - baseline_score
        selected = delta > 0 or (args.allow_equal and delta == 0)
        submission_updated = False
        if selected and args.update_submission:
            destination = args.submission_dir / f"{dataset}.txt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate.output_path, destination)
            validate_prediction_file(destination, dataset)
            submission_updated = True

        final_rows.append(
            {
                "dataset": dataset,
                "baseline_balanced_accuracy": baseline_score,
                "baseline_name": baseline_name,
                "candidate_balanced_accuracy": candidate.score,
                "delta_vs_baseline": delta,
                "candidate_output": str(candidate.output_path),
                "candidate_valid": True,
                "selected_for_submission": selected,
                "submission_updated": submission_updated,
                "reason": "beats baseline" if selected else "does not beat baseline",
                "candidate_combo": candidate.row.get("combo", candidate.row.get("method", "")),
            }
        )

    out_csv = args.results_dir / "sota_final_selection_summary.csv"
    out_json = args.results_dir / "sota_final_selection_summary.json"
    write_csv(out_csv, final_rows)
    out_json.write_text(json.dumps(final_rows, indent=2), encoding="utf-8")
    print(json.dumps({"summary_csv": str(out_csv), "rows": final_rows}, indent=2))


if __name__ == "__main__":
    main()
