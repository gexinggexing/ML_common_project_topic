# CHINESE SOTA-first adaptation note

- Source inspiration: ChineseEEG reading corpus and EEG2TEXT-CN-style preprocessing, but not text decoding.
- Faithful part: EEG-only reading-state features from 22-channel, 200 Hz, 1-second windows.
- Adapted part: public semantic-alignment methods are converted into binary reading-detection feature baselines.
- Not reproduced: text embeddings, event markers, 128-channel EGI layout, subject/run-aware alignment.
