# EEG/BCI 课程项目 — 最终汇总

## 各数据集最佳验证准确率

| 数据集 | 最佳模型 | 验证 Acc | Epoch | 模型参数量 | Test 预测 |
|--------|---------|---------|-------|-----------|----------|
| **CHINESE** | `best_CHINESE.pth` (原版) | **49.5%** | 10 | - | ✅ `results/predictions_CHINESE.json` |
| **BCIC2A** | `best_BCIC2A_v2.pth` (v2) | **53.6%** | 184 | - | ✅ `results/predictions_BCIC2A.json` |
| **MDD** | `best_MDD.pth` (原版) | **94.7%** | 89 | - | ✅ `results/predictions_MDD.json` |
| **SEED** | `best_SEED.pth` (原版) | **34.9%** | 1 | - | ✅ `results/predictions_SEED.json` |
| **SLEEP** | `best_SLEEP_lite_v2.pth` (v2) | **73.5%** | 63 | 79,877 | ✅ `results/predictions_SLEEP.json` |
| **BCI_SPEECH** | `best_BCI_SPEECH_v2.pth` (v2) | **20.0%** | 2 | - | ✅ `results/predictions_BCI_SPEECH.json` |

## 关键说明

### 模型来源
- **原版** = `train_all.py` 训练的模型，使用 `models/multimodel.py`
- **v2** = `train_all_v2.py` 训练的模型，使用 `models/multimodel_v2.py`
- **SLEEP Lite v2** = `train_sleep_lite.py` 改进版（dropout 0.2, wd 0.005），使用 `models/sleep_lite_cnn.py`

### 训练代码位置
| 脚本 | 用途 |
|------|------|
| `train_all.py` | 原版统一训练脚本（CHINESE, BCIC2A, MDD, SEED, SLEEP, BCI_SPEECH） |
| `train_all_v2.py` | v2 轻量版训练脚本（更强的正则化，z-score 归一化） |
| `train_sleep_lite.py` | SLEEP 专用 SleepLiteCNN 训练脚本 |
| `train_bcic2a_bandpower.py` | BCIC2A Bandpower + MLP（尝试，未用） |
| `train_bcic2a_fft_cnn.py` | BCIC2A FFT CNN + 增强（尝试，未用） |
| `train_bcic2a_csp_lda.py` | BCIC2A CSP + LDA（尝试，未用） |
| `train_bcic2a_v3_augment.py` | BCIC2A 时域增强版（未跑完） |
| `train_sleep_v2.py` | SLEEP 频域增强版（未跑完） |

### 模型架构
| 数据集 | 原版模型 | v2 模型 |
|--------|---------|---------|
| CHINESE | ShallowConvNet | ShallowConvNetLite |
| BCIC2A | DeepConvNet | DeepConvNetLite |
| MDD | CNN_SE_BiGRU | CNN_SE_BiGRU_Lite |
| SEED | CNN_ChannelAttention | CNN_ChannelAttn_Lite |
| SLEEP | ResNet1D_BiGRU | ResNet1D_BiGRU_Lite |
| BCI_SPEECH | DeepConvNet | DeepConvNetLite |

### Test 预测生成
运行 `_run_all_tests.py`（或 `test_all.py` / `test.py`）生成。

各 test 集样本数和预测分布：
- CHINESE: 200 样本 | 0:51, 1:149
- BCIC2A: 360 样本 | 0:135, 1:66, 2:90, 3:69
- MDD: 800 样本 | 0:26, 1:774
- SEED: 450 样本 | **0:450**（⚠️ 全部预测为 0）
- SLEEP: 1945 样本 | 0:372, 1:349, 2:362, 3:465, 4:397
- BCI_SPEECH: 250 样本 | **0:250**（⚠️ 全部预测为 0）

### ⚠️ 注意事项
1. **SEED 和 BCI_SPEECH 预测有问题**：checkpoint 保存的 epoch 太早（SEED ep1, BCI_SPEECH ep2），模型尚未充分学习，导致全部预测为单一类别。但因用户要求不再跑新训练，只能使用现有 checkpoint。
2. **MDD 预测高度偏向类别 1**：test 集可能本身类别不平衡，或模型有偏。但 val_acc 94.7% 很高，说明模型在验证集上表现好。
3. **SLEEP 预测最健康**：5 个类别分布相对均匀，73.5% 是第二高的准确率。

### 主要尝试记录（未采用）
| 方法 | 数据集 | 结果 | 说明 |
|------|--------|------|------|
| CSP + LDA | BCIC2A | 42.2% | 传统方法，效果差 |
| Bandpower + MLP | BCIC2A | 39.7% | 频域粗粒度特征丢失信息 |
| FFT CNN + 增强 | BCIC2A | 多次崩溃 | DataLoader collate 问题 |
| v3 时域增强 | BCIC2A | 未跑完 | dropout 0.5, wd 0.01, 增强样本 |
| SleepLiteCNN v2 | SLEEP | 73.5% | ✅ **已采用** |
| 频域 SleepLiteCNNv2 | SLEEP | 未跑完 | 组件测试通过，完整脚本崩溃 |
| SEED v2 轻量版 | SEED | 32.4% | 正则化过强，效果更差 |
| BCI_SPEECH v2 | BCI_SPEECH | 20.0% | 仅 2 个 epoch |

### 文件结构
```
D:\1\course project\course project\
├── BCIC2A/              # 数据
├── BCI_Speech/          # 数据
├── CHINESE/             # 数据
├── MDD/                 # 数据
├── SEED/                # 数据
├── SLEEP/               # 数据
├── checkpoints/           # 所有 checkpoint + history
│   ├── best_*.pth
│   └── history_*.json
├── models/                # 模型定义
│   ├── multimodel.py
│   ├── multimodel_v2.py
│   ├── eegnet.py
│   ├── sleep_lite_cnn.py
│   └── bcic2a_specialist.py
├── results/               # Test 预测
│   ├── predictions_CHINESE.json / .npy
│   ├── predictions_BCIC2A.json / .npy
│   ├── predictions_MDD.json / .npy
│   ├── predictions_SEED.json / .npy
│   ├── predictions_SLEEP.json / .npy
│   └── predictions_BCI_SPEECH.json / .npy
└── *.py                   # 训练脚本
```
