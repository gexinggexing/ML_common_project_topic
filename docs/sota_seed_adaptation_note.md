# SEED SOTA-first adaptation note

- Source inspiration: SEED DE/PSD baselines, DGCNN, RGNN, and 4D-aNN-style graph features.
- Faithful part: five-band DE/log-power features over two 1-second subwindows and 62 EEG nodes.
- Adapted part: graph networks are approximated by graph/covariance feature baselines in this first pass.
- Not reproduced: subject/session/trial-aware evaluation, LDS smoothing, domain adaptation, row-order templates.
