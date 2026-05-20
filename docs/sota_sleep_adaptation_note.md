# SLEEP SOTA-first adaptation note

- Source inspiration: ISRUC BSTT/HASS and EEG-only MixSleepNet-style multiband features.
- Faithful part: 6 EEG channels, 30-second epochs, 5 sleep stages, EEG-only feature extraction.
- Adapted part: temporal-context models are converted to independent-window classifiers.
- Not reproduced: EOG/EMG/ECG modalities and adjacent-epoch sequence context.
