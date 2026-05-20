# EEG Course Project

This repository contains runnable code and final prediction files for the shared EEG course project across five datasets:

- BCIC2A
- CHINESE
- MDD
- SEED
- SLEEP

Dataset files and generated intermediate artifacts are not committed because they are large. The code expects the course data root to be available at:

```text
/mnt/dataset3/panxy/course/project1_data/course project/course project
```

On the Lab 212 host, run the project from:

```text
/mnt/dataset4/yichen/ML_practice&homework/course_project_1
```

The recommended Python environment on the Lab 212 host is `timesfm2`.

## Final Submission Files

The final prediction files required by the course are tracked in `submission/`:

```text
submission/BCIC2A.txt
submission/CHINESE.txt
submission/MDD.txt
submission/SEED.txt
submission/SLEEP.txt
```

Each file contains one integer class label per line, without header or filename columns. The row order is the fixed test DataLoader order with `shuffle=False`.

Expected row counts:

| Dataset | Rows | Classes |
| --- | ---: | --- |
| BCIC2A | 360 | 0-3 |
| CHINESE | 200 | 0-1 |
| MDD | 800 | 0-1 |
| SEED | 450 | 0-2 |
| SLEEP | 1945 | 0-4 |

## Main Reproduction Commands

Run the five-dataset SOTA-first pipeline on the Lab 212 host:

```bash
bash run_sota_five_pipeline.sh
```

The script runs the per-dataset pipelines, validates candidate predictions, and updates `outputs/submission/` when a candidate beats the current valid baseline.

Individual scripts:

```bash
python run_sota_bcic2a.py
python run_sota_seed.py
python run_sota_chinese.py
python run_sota_mdd.py
python run_sota_sleep.py
python run_sota_finalize.py --update-submission
```

## Current Final Selection

| Dataset | Validation balanced accuracy | Final method |
| --- | ---: | --- |
| BCIC2A | 0.6028 | FBCSP wide overlap CSP + ExtraTrees + MIBIF |
| SEED | 0.4556 | beta/gamma graph-inspired features + Ridge |
| CHINESE | 0.7100 | covariance and band statistics + RandomForest |
| MDD | 0.9328 | retained strongest CBraMod baseline |
| SLEEP | 0.7594 | band/covariance statistics + ExtraTrees |

## Repository Layout

- `run_sota_*.py`: final per-dataset pipelines.
- `run_sota_five_pipeline.sh`: one-command five-dataset run.
- `run_sota_finalize.py`: conservative final-selection and submission updater.
- `sota_common.py`: shared data loading, feature extraction, validation, and output utilities.
- `run_all_cbramod.py`, `run_cbramod_pooling_ablation.py`, `run_cbramod_author_downstream.py`: CBraMod baseline and ablation pipelines.
- `docs/`: method notes, adaptation notes, and model inventory.
- `submission/`: final five `.txt` files for course submission.

## Files Intentionally Not Tracked

The following are generated or too large for GitHub and are ignored:

- `artifacts/`
- `outputs/`
- `.codex_runs/`
- model weights and data files such as `*.h5`, `*.pt`, `*.pth`, `*.npz`
