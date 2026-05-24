# EEG 数据集模型推荐报告

> 基于 5 个脑电数据集的数据特性分析

---

## 一、数据集概览

| 数据集 | 下游任务 | 类别数 | 通道数 | 时间步 | 训练样本 | 测试样本 | 特点 |
|--------|----------|--------|--------|--------|----------|----------|------|
| BCIC2A | 运动想象 | 4 | 22 | 800 | 720 | 360 | 小样本，高区分度任务 |
| CHINESE | 阅读检测 | 2 | 22 | 200 | 400 | 200 | 最小数据集，简单二分类 |
| MDD | 抑郁识别 | 2 | 20 | 200 | 960 | 800 | 医学诊断，类不平衡需注意 |
| SEED | 情绪识别 | 3 | 62 | 400 | 900 | 450 | 通道数最多，空间信息丰富 |
| SLEEP | 睡眠分期 | 5 | 6 | 6000 | 3921 | 1945 | 时间步最长，时序依赖强 |

---

## 二、推荐模型

### 首选：EEGNet（统一框架）

**为什么选 EEGNet：**
- **专为EEG设计**：深度可分离卷积同时提取时间和空间特征，完美匹配脑电数据
- **参数极少**：约1K-5K参数，小数据集也不容易过拟合
- **跨任务泛化好**：在运动想象、情绪识别、睡眠分期上都有SOTA或接近SOTA的表现
- **可配置性强**：只需改 `num_channels`、`num_classes`、`num_timepoints` 三个参数即可适配所有数据集
- **论文引用高**：Lawhern et al. 2018，EEG领域最经典的深度学习模型之一

**各数据集配置建议：**

```python
configs = {
    "BCIC2A":  {"num_channels": 22, "num_classes": 4,  "num_timepoints": 800,  "dropout_rate": 0.5},
    "CHINESE": {"num_channels": 22, "num_classes": 2,  "num_timepoints": 200,  "dropout_rate": 0.5},
    "MDD":     {"num_channels": 20, "num_classes": 2,  "num_timepoints": 200,  "dropout_rate": 0.5},
    "SEED":    {"num_channels": 62, "num_classes": 3,  "num_timepoints": 400,  "dropout_rate": 0.5},
    "SLEEP":   {"num_channels": 6,  "num_classes": 5,  "num_timepoints": 6000, "dropout_rate": 0.3},
}
```

> SLEEP 数据集样本量大，dropout 可以稍低；其他数据集样本量小，dropout 用 0.5 防过拟合。

---

### 备选方案

| 模型 | 适用数据集 | 优势 | 劣势 |
|------|-----------|------|------|
| **ShallowConvNet** | BCIC2A, CHINESE | 结构更简单，训练更快 | 通道数多时效果下降 |
| **DeepConvNet** | SEED, BCIC2A | 容量更大，能拟合复杂模式 | 小数据集容易过拟合 |
| **EEG-Conformer** | SEED, SLEEP | Transformer 捕获长程依赖 | 参数量大，小数据集需要预训练 |
| **CNN+LSTM** | SLEEP | 显式建模时序 | 训练慢，调参复杂 |
| **GCN (图卷积)** | SEED | 利用电极空间拓扑关系 | 需要电极坐标，实现复杂 |

**课程项目建议**：先用 EEGNet 跑通全部 5 个数据集作为 baseline，如果时间充裕，再挑 1-2 个数据集尝试 Conformer 或 CNN+LSTM 做对比实验。

---

## 三、训练策略建议

### 1. 数据预处理
- 你的数据已经是 `(N, C, T)` 格式，符合 PyTorch 标准
- EEGNet 期望输入 `(batch, 1, C, T)` —— 已在你提供的模型代码中处理

### 2. 划分策略
- 你已经有 train/val/test 划分，直接用即可
- SLEEP 数据集训练集 3921 样本，可以考虑做 subject-independent 验证（如果知道被试信息的话）

### 3. 训练超参数

```python
# 通用配置
batch_size = 32
learning_rate = 0.001
epochs = 200
optimizer = Adam
scheduler = CosineAnnealingLR  # 或 StepLR
weight_decay = 0.01  # 小数据集建议加

# 早停策略
patience = 30  # val loss 不下降就停
```

### 4. 评价指标
- 分类任务：Accuracy, F1-score（macro，因为类别可能不平衡）
- SLEEP 分期额外关注：Cohen's Kappa（睡眠领域标准）

### 5. 过拟合防范
- BCIC2A、CHINESE、SEED 训练样本 < 1000，过拟合风险高
- 强烈建议：加 dropout、weight decay、早停、数据增强（如果有被试信息可以做 subject-wise augmentation）

---

## 四、快速开始

我已经为你实现了：
1. `models/eegnet.py` —— 可配置 EEGNet 模型
2. `train.py` —— 统一训练脚本，支持所有 5 个数据集
3. `test.py` —— 测试脚本，生成预测结果

使用示例：
```bash
# 训练 BCIC2A
python train.py --dataset BCIC2A --epochs 200 --lr 0.001

# 训练 SEED
python train.py --dataset SEED --epochs 200 --lr 0.001

# 训练 SLEEP
python train.py --dataset SLEEP --epochs 100 --lr 0.001 --batch_size 64

# 测试（生成 submission/predictions）
python test.py --dataset BCIC2A --checkpoint checkpoints/best_BCIC2A.pth
```

---

## 五、各数据集特别提示

### BCIC2A（运动想象）
- 4分类任务，CNN 的空间滤波层很关键
- EEGNet 的 `F1=8, D=2` 配置适合 22 通道
- 论文里常用 F1=8, D=2,  dropout=0.5

### CHINESE（阅读检测）
- 样本最少（400），最容易过拟合
- 建议加更强的正则化（dropout 0.5 + weight decay 0.01）
- 训练 epoch 不用太多，80-100 可能就够

### MDD（抑郁识别）
- 医学数据，准确率不是唯一指标，要关注 sensitivity/specificity
- 二分类可以用 BCE loss 替代 CrossEntropy
- 注意类别是否平衡

### SEED（情绪识别）
- 62 通道，空间信息最丰富
- EEGNet 会自动学习空间滤波，效果通常很好
- 也可以尝试把 62 通道重排成近似 2D 拓扑后接 2D CNN

### SLEEP（睡眠分期）
- 时间步 6000（30秒×200Hz），序列很长
- EEGNet 也能处理，但时间维度卷积核可能需要调大
- 5 类分类，类别不平衡（N2最多，N1/N3/REM较少）
- 建议用 weighted CrossEntropy 或 Focal Loss
- 睡眠领域标准：Accuracy + Cohen's Kappa + per-class F1

---

## 六、参考文献

1. Lawhern et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *JNE*.
2. Schirrmeister et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*.
3. Zhao et al. (2023). EEG Conformer: A Transformer-based Model for EEG Classification. *ICASSP*.
4. Supratak et al. (2017). TinySleepNet: An Efficient Deep Learning Model for Sleep Stage Scoring. * arXiv*.

---

*祝课程项目顺利！有问题随时问我。*
