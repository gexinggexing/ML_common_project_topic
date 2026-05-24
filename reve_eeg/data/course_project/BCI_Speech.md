# BCI_Speech

- dataset path (windows): `\\10.16.93.90\dataset3\nzh\eeg_FM\ML_common_project_topic\BCI_Speech`
- dataset path (linux mnt): `/mnt/dataset3/nzh/eeg_FM/ML_common_project_topic/BCI_Speech`
- train file: `train.h5`
- val file: `val.h5`
- test file: `test_x_only.h5`

## H5 Structure

- train keys: `X`, `y`
- val keys: `X`, `y`
- test keys: `X`

## Shapes

- train: `[4050, 64, 600]`
- val: `[200, 64, 600]`
- test: `[250, 64, 600]`

## Labels

- label key: `y`
- n_classes: `5`
- train label distribution: `0:810, 1:810, 2:810, 3:810, 4:810`
- val label distribution: `0:40, 1:40, 2:40, 3:40, 4:40`

## Working Assumptions

- current H5 tensors are already `[N, C, T]`
- current time length `600` implies `3s` at `200Hz`
- no `dataset_info.json` / `dataset_info_fixed.json` was found beside the H5 files
- because metadata is missing, the course loader uses a standard 64-channel EEG montage as a position fallback for REVE

## Notes

- test set has no labels, so only prediction export is possible
- dataset looks class-balanced in train and val
- this dataset replaces `CHINESE` for the current course-project run
