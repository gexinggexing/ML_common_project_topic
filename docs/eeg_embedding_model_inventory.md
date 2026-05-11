# EEG Embedding Model Inventory

| Model | NAS weights | NAS code | Load / smoke status | Fits five course datasets? | Recommended role | Notes |
|---|---|---|---|---|---|---|
| SubCLR | `/mnt/dataset0/yinte/direct_eeg_try/subclr/pretrained/subclr_HBN_checkpoint_seed_0.pt`; also HBN/TUAB seeds 0-4 | `/mnt/dataset0/yinte/direct_eeg_try/subclr/models/models.py`; helper code also in `/mnt/dataset1/ws2319/workspace/addiction/models.py` | Weight is a plain encoder state dict; no classifier head found | Yes. Use channel-wise input `(B*C, 1, T)` and mean-pool channel embeddings | Main frozen encoder | Cleanest existing route. Good for MDD, SLEEP, CHINESE; useful as complement for BCIC2A/SEED |
| CBraMod | `/mnt/dataset4/yinuo/personlized_FM/CBraMod-main/CBraMod-main/pretrained_weights/pretrained_weights.pth` | `/mnt/dataset4/yinuo/personlized_FM/CBraMod-main/CBraMod-main/models/cbramod.py`; `/mnt/dataset4/yinuo/personlized_FM/CBraMod-main/CBraMod-main/models/criss_cross_transformer.py` | Verified `CBraMod()` loads with `missing=0`, `unexpected=0` | Yes. Native input is `(B, C, patch_num, 200)`, matching 200 Hz 1-second patches | Add to main frozen encoder set | Strong new candidate. Patch counts: CHINESE/MDD=1, SEED=2, BCIC2A=4, SLEEP=30. Output can be mean-pooled to `(B, 200)` |
| LaBraM clean base | `/mnt/dataset1/ws2319/workspace/addiction/baseline_models_features/pretrained_pth/braindecode_labram_base.pt` | Needs `braindecode.models.Labram`; current `timesfm2` did not have `braindecode` installed when checked | Weight is clean-looking LaBraM base state dict; local non-braindecode code does not match key names directly | Yes after installing/using matching Braindecode implementation | Main frozen encoder, especially for BCIC2A and SEED | Drop mismatched `position_embedding`, `temporal_embedding`, and `final_layer.*` if dimensions differ |
| LaBraM local multi-dataset | `/mnt/dataset4/yizhiliao/ckpt/multi_datasets_labram/checkpoint.pth`; checkpoints 0-19 also present | `/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/modeling_finetune.py` | Verified it can load into local `labram_base_patch200_200` with `num_patches_per_channel_input=16` and output `(B, 200)` for patch counts 1, 2, 4, 16 | Partially. SLEEP has 30 patches, so it needs chunking because checkpoint time embedding has 16 patches | Experimental only | Checkpoint contains `model`, `optimizer`, `epoch`, so it is a training-process checkpoint. Use cautiously because source data may overlap course data |
| CBraMod local multi-dataset | `/mnt/dataset4/yizhiliao/ckpt/multi_datasets_cbramod/checkpoint.pth`; checkpoints 0-19 also present | Same CBraMod code as above | Loads with `missing=2` (`proj_out.0.weight`, `proj_out.0.bias`) and `unexpected=0` | Probably yes, but weaker than clean CBraMod pretrained weight | Experimental only | Also a training-process checkpoint with optimizer/epoch. Prefer clean CBraMod pretrained weight unless training source is verified |
| SignalJEPA | `/mnt/dataset1/ws2319/workspace/addiction/baseline_models_features/pretrained_pth/signal-jepa_16s-60_adeuwv4s.pth` | No matching model class found on the checked NAS paths | Weight found, code not found | Unknown | Do not prioritize | Keep as later option only if matching SignalJEPA implementation is found |
| EEGPT | `/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/EEGPT/model.safetensors` | `/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/EEGPT/models/EEGPT_mcae.py`; `/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/EEGPT/models/EEGPT_mcae_finetune.py` | Code and weight exist; smoke load not yet verified | Likely possible, but integration cost is higher | Optional extension | Useful if SubCLR/CBraMod/LaBraM underperform |
| BIOT | Several scripts reference BIOT; clear pretrained reusable weight not established in the current search | `/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/biot.py`; `/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/EEGPT/downstream/Modules/BIOT/biot.py` | Code exists; clean weight path not confirmed | Unknown | Optional / baseline only | Existing notebooks are mostly fine-tune/embedding analysis workflows |
| LUNA / FEMBA / BrainOmni | `/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/models/checkpoint/FEMBA_tiny.safetensors`; `/mnt/dataset4/yinuo/personlized_FM/personalized_FM_0108/models/checkpoint/BrainOmni_tiny/BrainOmni.pt`; other checkpoint files in same folder | `/mnt/dataset4/yinuo/personlized_FM/BioFoundation-main/BioFoundation-main/models/LUNA.py`; `/mnt/dataset4/yinuo/personlized_FM/BioFoundation-main/BioFoundation-main/models/FEMBA.py` | Code and some weights exist; smoke load not verified | Unknown | Low-priority extension | More engineering cost than CBraMod/SubCLR/LaBraM |
| REVE | No local NAS weight found | No local NAS code found | Not available locally | Would fit if downloaded, but not available on NAS | Not part of local-only route | Previous web-based option; excluded from NAS-only route |
| ST-EEGFormer | No local NAS weight found | No local NAS code found | Not available locally | Would need resampling/chunking even if downloaded | Not part of local-only route | Excluded from NAS-only route |

## Practical Priority

| Priority | Encoder | Why |
|---:|---|---|
| 1 | SubCLR | Weight and code already available; clean frozen encoder route |
| 2 | CBraMod clean pretrained | Weight and code verified; native 200-sample patching matches all datasets |
| 3 | LaBraM clean base | Strong model, but needs matching Braindecode implementation in the environment |
| 4 | EEGPT | Code and weight exist, but needs smoke test and more adapter work |
| 5 | Local multi-dataset LaBraM / CBraMod | Technically runnable but provenance is uncertain, so keep for analysis only |

## Dataset Patch Mapping For CBraMod / Local LaBraM-Style Inputs

| Dataset | Shape `(C, T)` | 200 Hz patch count |
|---|---:|---:|
| BCIC2A | `(22, 800)` | 4 |
| CHINESE | `(22, 200)` | 1 |
| MDD | `(20, 200)` | 1 |
| SEED | `(62, 400)` | 2 |
| SLEEP | `(6, 6000)` | 30 |
