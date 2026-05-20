# Targeted BCIC2A and SEED Sweep

## Goal

Find stronger validation methods for `BCIC2A` and `SEED` than the existing
CBraMod validation baselines, then write the selected hidden-test prediction
files.

Current remote baselines before this sweep:

- `BCIC2A`: best balanced validation accuracy `0.4389`
- `SEED`: best balanced validation accuracy `0.4044`

## Method Rationale

- `BCIC2A` is a 4-class motor-imagery task. The sweep prioritizes
  one-vs-rest CSP and filter-bank CSP because FBCSP is a standard strong method
  for BCI Competition IV Dataset 2a.
- `SEED` is an emotion-recognition task. The sweep prioritizes band log-power
  / differential-entropy style features, optional left-right asymmetry, and
  Hjorth statistics because SEED baselines commonly rely on spectral entropy
  features plus classical classifiers.

## Sweep Families

`run_targeted_bcic_seed_sweep.py` evaluates 22 total normal EEG combinations:

- 10 `BCIC2A` combinations:
  - narrow 4 Hz FBCSP + LDA/logistic/linear SVM/RBF SVM
  - wider FBCSP + LDA/ExtraTrees
  - mu-beta CSP + LDA
  - dense filter-bank CSP + ridge
  - log-power + RBF SVM/MLP
- 12 `SEED` combinations:
  - 5-band spectral features + logistic/RBF SVM/MLP
  - spectral features plus left-right asymmetry
  - spectral features plus Hjorth statistics
  - spectral + asymmetry + Hjorth with PCA-SVM/MLP
  - spectral + asymmetry + raw statistics with tree ensembles

All validation metrics use the provided train/val split. Hidden-test files are
written only after the best validation combination for each dataset is selected
and refit on train+val.

Do not use row-order or non-shuffle label templates as a model. Those are split
artifacts, not normal EEG classifiers.

## Outputs

- `artifacts/results/targeted_bcic_seed_sweep_metrics.csv`
- `artifacts/results/targeted_bcic_seed_sweep_selection_summary.csv`
- `outputs/targeted_bcic_seed_sweep/BCIC2A.txt`
- `outputs/targeted_bcic_seed_sweep/SEED.txt`
- optionally updated:
  - `outputs/submission/BCIC2A.txt`
  - `outputs/submission/SEED.txt`
