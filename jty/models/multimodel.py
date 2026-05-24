"""
Multi-Model EEG Classification Framework
为 5 个数据集各匹配一个领域代表性 SOTA 架构

数据集 → 模型映射:
- BCIC2A:   DeepConvNet       (Schirrmeister 2017, 运动想象经典深度CNN)
- CHINESE:  ShallowConvNet    (Schirrmeister 2017, 小样本浅层CNN)
- MDD:      CNN_SE_BiGRU      (GCGRU+SE 变体, 抑郁检测时序+空间注意力)
- SEED:     CNN_ChannelAttn   (SEED 情绪识别常用 CNN+Channel Attention)
- SLEEP:    ResNet1D_BiGRU    (ResNet+SE+LSTM 变体, 睡眠分期标准架构)

所有模型输入: (batch, num_channels, num_timepoints)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0. 通用工具模块
# ============================================================

class SqueezeExcitation(nn.Module):
    """SE 注意力模块 (Hu et al. 2018)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ChannelAttention(nn.Module):
    """通道注意力 (类似 SE 但用 Conv1d 实现)"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.shared_mlp(self.avg_pool(x))
        max_ = self.shared_mlp(self.max_pool(x))
        return x * self.sigmoid(avg + max_)


# ============================================================
# 1. DeepConvNet  (for BCIC2A 运动想象)
#    Schirrmeister et al. 2017, Human Brain Mapping
# ============================================================

class DeepConvNet(nn.Module):
    """
    4层深度CNN + 分类器
    输入: (B, C, T)
    """
    def __init__(self, num_classes, num_channels, num_timepoints, dropout=0.5):
        super().__init__()
        self.num_channels = num_channels
        self.num_timepoints = num_timepoints

        # Block 1: 时间卷积 (大kernel捕获频率特征)
        self.conv1 = nn.Conv2d(1, 25, (1, 10), padding=(0, 5), bias=False)
        self.conv2 = nn.Conv2d(25, 25, (num_channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(25)
        self.elu = nn.ELU()
        self.maxpool1 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop1 = nn.Dropout(dropout)

        # Block 2
        self.conv3 = nn.Conv2d(25, 50, (1, 10), padding=(0, 5), bias=False)
        self.bn2 = nn.BatchNorm2d(50)
        self.maxpool2 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop2 = nn.Dropout(dropout)

        # Block 3
        self.conv4 = nn.Conv2d(50, 100, (1, 10), padding=(0, 5), bias=False)
        self.bn3 = nn.BatchNorm2d(100)
        self.maxpool3 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop3 = nn.Dropout(dropout)

        # Block 4
        self.conv5 = nn.Conv2d(100, 200, (1, 10), padding=(0, 5), bias=False)
        self.bn4 = nn.BatchNorm2d(200)
        self.maxpool4 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop4 = nn.Dropout(dropout)

        # 自动计算 flatten 维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, num_timepoints)
            out = self._forward_feature(dummy)
            self.flatten_dim = out.numel()

        self.classifier = nn.Linear(self.flatten_dim, num_classes)

    def _forward_feature(self, x):
        # Block 1
        x = self.conv1(x)           # (B,25,C,T)
        x = self.conv2(x)           # (B,25,1,T)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.maxpool2(x)
        x = self.drop2(x)

        # Block 3
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.maxpool3(x)
        x = self.drop3(x)

        # Block 4
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.elu(x)
        x = self.maxpool4(x)
        x = self.drop4(x)
        return x

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B,1,C,T)
        x = self._forward_feature(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# 2. ShallowConvNet  (for CHINESE 阅读检测)
#    Schirrmeister et al. 2017, 小样本浅层CNN
# ============================================================

class ShallowConvNet(nn.Module):
    """
    2层浅层CNN — 适合小样本数据集
    输入: (B, C, T)
    """
    def __init__(self, num_classes, num_channels, num_timepoints, dropout=0.5):
        super().__init__()
        # 时间卷积 (捕捉频率信息)
        self.conv1 = nn.Conv2d(1, 40, (1, 25), padding=(0, 12), bias=False)
        # 空间卷积 (跨通道)
        self.conv2 = nn.Conv2d(40, 40, (num_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        # 平均池化保留更多信息
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, num_timepoints)
            out = self._forward_feature(dummy)
            self.flatten_dim = out.numel()

        self.classifier = nn.Linear(self.flatten_dim, num_classes)

    def _forward_feature(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = x ** 2  # 平方激活 (论文原文)
        x = self.pool(x)
        x = torch.log(x + 1e-6)  # 对数
        x = self.dropout(x)
        return x

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._forward_feature(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# 3. CNN_SE_BiGRU  (for MDD 抑郁识别)
#    结合 CNN 空间特征提取 + SE注意力 + BiGRU 时序建模
#    参考: GCGRU (2024) + SE 模块
# ============================================================

class CNN_SE_BiGRU(nn.Module):
    """
    CNN + SE + BiGRU —— 适合需要捕获时序依赖的医学EEG分类
    输入: (B, C, T)
    """
    def __init__(self, num_classes, num_channels, num_timepoints,
                 hidden_size=64, num_layers=2, dropout=0.5):
        super().__init__()
        # CNN 特征提取 (沿时间轴)
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(2)
        self.se = SqueezeExcitation(256, reduction=16)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

        # BiGRU 时序建模
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.elu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.elu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.elu(self.bn3(self.conv3(x)))
        x = self.se(x)
        x = self.pool(x)
        x = self.dropout(x)

        # GRU: (B, 256, T') -> (B, T', 256)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)  # (B, T', hidden*2)

        # 取最后时刻 + 平均池化
        x_last = x[:, -1, :]  # (B, hidden*2)
        x_mean = x.mean(dim=1)  # (B, hidden*2)
        x = x_last + x_mean  # 融合

        return self.classifier(x)


# ============================================================
# 4. CNN_ChannelAttention  (for SEED 情绪识别)
#    通道注意力增强的 CNN —— SEED 62通道空间信息丰富
#    参考: ECLGCNN / DGCNN 思想，用 CNN+ChannelAttn 实现
# ============================================================

class CNN_ChannelAttention(nn.Module):
    """
    多层 CNN + Channel Attention —— 适合多通道EEG情绪识别
    输入: (B, C, T)
    """
    def __init__(self, num_classes, num_channels, num_timepoints, dropout=0.5):
        super().__init__()
        # 每个block: Conv -> BN -> ELU -> ChannelAttn -> Pool
        self.block1 = nn.Sequential(
            nn.Conv1d(num_channels, 64, 7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            ChannelAttention(64, reduction=8),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            ChannelAttention(128, reduction=8),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ELU(),
            ChannelAttention(256, reduction=8),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, num_timepoints)
            out = self._forward_feature(dummy)
            self.flatten_dim = out.numel()

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _forward_feature(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x):
        x = self._forward_feature(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# 5. ResNet1D_BiGRU  (for SLEEP 睡眠分期)
#    1D ResNet + BiGRU —— 睡眠分期标准架构
#    参考: ResNet-SE-LSTM / TinySleepNet 变体
# ============================================================

class ResBlock1D(nn.Module):
    """1D Residual Block"""
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, dropout=0.3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SqueezeExcitation(out_ch, reduction=16)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

        # shortcut
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.elu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x = x + residual
        x = self.elu(x)
        return x


class ResNet1D_BiGRU(nn.Module):
    """
    1D ResNet + BiGRU —— 适合长序列睡眠分期
    输入: (B, C, T)
    """
    def __init__(self, num_classes, num_channels, num_timepoints,
                 hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, 64, 15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )

        # ResBlocks (下采样逐步减小序列长度)
        self.res1 = ResBlock1D(64, 64, stride=2, dropout=dropout)
        self.res2 = ResBlock1D(64, 128, stride=2, dropout=dropout)
        self.res3 = ResBlock1D(128, 256, stride=2, dropout=dropout)
        self.res4 = ResBlock1D(256, 512, stride=2, dropout=dropout)

        # BiGRU 捕获睡眠阶段转换规则
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        # BiGRU: (B, 512, T') -> (B, T', 512)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)

        # 最后时刻 + 平均池化
        x_last = x[:, -1, :]
        x_mean = x.mean(dim=1)
        x = x_last + x_mean

        return self.classifier(x)


# ============================================================
# 模型注册表 & 构建函数
# ============================================================

MODEL_REGISTRY = {
    "BCIC2A":    DeepConvNet,
    "CHINESE":   ShallowConvNet,
    "MDD":       CNN_SE_BiGRU,
    "SEED":      CNN_ChannelAttention,
    "SLEEP":     ResNet1D_BiGRU,
    "BCI_SPEECH":DeepConvNet,
}

DATASET_CONFIGS = {
    "BCIC2A":    {"num_classes": 4,  "num_channels": 22, "num_timepoints": 800,  "dropout": 0.5},
    "CHINESE":   {"num_classes": 2,  "num_channels": 22, "num_timepoints": 200,  "dropout": 0.5},
    "MDD":       {"num_classes": 2,  "num_channels": 20, "num_timepoints": 200,  "dropout": 0.5},
    "SEED":      {"num_classes": 3,  "num_channels": 62, "num_timepoints": 400,  "dropout": 0.5},
    "SLEEP":     {"num_classes": 5,  "num_channels": 6,  "num_timepoints": 6000, "dropout": 0.3},
    "BCI_SPEECH":{"num_classes": 5,  "num_channels": 64, "num_timepoints": 600,  "dropout": 0.5},
}


def build_model(dataset_name: str):
    """根据数据集名称自动构建对应模型"""
    name = dataset_name.upper()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Supported: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[name]
    config = DATASET_CONFIGS[name]
    return model_cls(**config), config


# ========== 测试 ==========
if __name__ == "__main__":
    for name in DATASET_CONFIGS.keys():
        model, cfg = build_model(name)
        dummy = torch.zeros(2, cfg["num_channels"], cfg["num_timepoints"])
        out = model(dummy)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{name:8s}: input {str(dummy.shape):20s} -> output {str(out.shape):15s} | params: {num_params:,}")
