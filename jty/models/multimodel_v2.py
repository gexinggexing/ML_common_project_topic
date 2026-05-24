"""
Multi-Model EEG Classification Framework - v2 (轻量版)
为 5 个数据集匹配轻量模型 + z-score 数据归一化

改进:
1. 所有模型参数量大幅缩减
2. 输入增加 z-score 归一化 (per-sample)
3. 更强的正则化 (dropout 0.7, weight_decay 0.1)
4. 更保守的学习率 (0.0005)

数据集 → 模型映射:
- BCIC2A:   DeepConvNetLite  (砍半滤波器数)
- MDD:      CNN_SE_BiGRU_Lite (保留原版，效果很好)
- SEED:     CNN_ChannelAttn_Lite (砍层+砍通道)
- SLEEP:    ResNet1D_BiGRU_Lite (砍到2残差块)
- BCI_SPEECH: DeepConvNetLite (砍半滤波器数)

输入: (batch, num_channels, num_timepoints)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0. 通用工具模块
# ============================================================

class SqueezeExcitation(nn.Module):
    """SE 注意力模块 (Lite版，reduction更大)"""
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ChannelAttention(nn.Module):
    """通道注意力 (Lite版)"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(in_channels, max(1, in_channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(1, in_channels // reduction), in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.shared_mlp(self.avg_pool(x))
        max_ = self.shared_mlp(self.max_pool(x))
        return x * self.sigmoid(avg + max_)


# ============================================================
# 1. DeepConvNetLite  (轻量版，用于 BCIC2A, BCI_SPEECH)
# ============================================================

class DeepConvNetLite(nn.Module):
    """
    DeepConvNet 轻量版 - 滤波器数砍半，kernel减小
    输入: (B, C, T)
    """
    def __init__(self, num_classes, num_channels, num_timepoints, dropout=0.7):
        super().__init__()
        # Block 1: 12 滤波器 (原版 25)
        self.conv1 = nn.Conv2d(1, 12, (1, 5), padding=(0, 2), bias=False)
        self.conv2 = nn.Conv2d(12, 12, (num_channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(12)
        self.elu = nn.ELU()
        self.maxpool1 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: 24 (原版 50)
        self.conv3 = nn.Conv2d(12, 24, (1, 5), padding=(0, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(24)
        self.maxpool2 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop2 = nn.Dropout(dropout)

        # Block 3: 48 (原版 100)
        self.conv4 = nn.Conv2d(24, 48, (1, 5), padding=(0, 2), bias=False)
        self.bn3 = nn.BatchNorm2d(48)
        self.maxpool3 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop3 = nn.Dropout(dropout)

        # Block 4: 96 (原版 200)
        self.conv5 = nn.Conv2d(48, 96, (1, 5), padding=(0, 2), bias=False)
        self.bn4 = nn.BatchNorm2d(96)
        self.maxpool4 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop4 = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, num_timepoints)
            out = self._forward_feature(dummy)
            self.flatten_dim = out.numel()

        self.classifier = nn.Linear(self.flatten_dim, num_classes)

    def _forward_feature(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool1(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.maxpool2(x)
        x = self.drop2(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.maxpool3(x)
        x = self.drop3(x)

        x = self.conv5(x)
        x = self.bn4(x)
        x = self.elu(x)
        x = self.maxpool4(x)
        x = self.drop4(x)
        return x

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._forward_feature(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# 2. CNN_SE_BiGRU_Lite  (MDD - 保留原版，效果很好)
# ============================================================

class CNN_SE_BiGRU_Lite(nn.Module):
    """
    MDD 效果已经很好 (94.7%)，只做微调：
    - 通道数砍半
    - dropout 加大
    输入: (B, C, T)
    """
    def __init__(self, num_classes, num_channels, num_timepoints,
                 hidden_size=32, num_layers=1, dropout=0.7):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.se = SqueezeExcitation(128, reduction=32)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
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

        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x_last = x[:, -1, :]
        x_mean = x.mean(dim=1)
        x = x_last + x_mean
        return self.classifier(x)


# ============================================================
# 3. CNN_ChannelAttn_Lite  (SEED)
# ============================================================

class CNN_ChannelAttn_Lite(nn.Module):
    """
    SEED 轻量版：2层CNN + 通道注意力
    输入: (B, C, T)
    """
    def __init__(self, num_classes, num_channels, num_timepoints, dropout=0.7):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(num_channels, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            ChannelAttention(32, reduction=8),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            ChannelAttention(64, reduction=8),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, num_timepoints)
            out = self._forward_feature(dummy)
            self.flatten_dim = out.numel()

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _forward_feature(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

    def forward(self, x):
        x = self._forward_feature(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# 4. ResNet1D_BiGRU_Lite  (SLEEP)
# ============================================================

class ResBlock1DLite(nn.Module):
    """1D Residual Block Lite"""
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, dropout=0.5):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SqueezeExcitation(out_ch, reduction=32)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

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


class ResNet1D_BiGRU_Lite(nn.Module):
    """
    1D ResNet Lite + BiGRU Lite
    输入: (B, C, T)
    """
    def __init__(self, num_classes, num_channels, num_timepoints,
                 hidden_size=64, num_layers=1, dropout=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, 32, 15, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
        )

        # 只保留2个残差块，stride=4 快速下采样
        self.res1 = ResBlock1DLite(32, 64, stride=4, dropout=dropout)
        self.res2 = ResBlock1DLite(64, 128, stride=4, dropout=dropout)

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)

        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)

        x_last = x[:, -1, :]
        x_mean = x.mean(dim=1)
        x = x_last + x_mean
        return self.classifier(x)


# ============================================================
# 模型注册表
# ============================================================

MODEL_REGISTRY = {
    "BCIC2A":      DeepConvNetLite,
    "MDD":         CNN_SE_BiGRU_Lite,
    "SEED":        CNN_ChannelAttn_Lite,
    "SLEEP":       ResNet1D_BiGRU_Lite,
    "BCI_SPEECH":  DeepConvNetLite,
}

DATASET_CONFIGS = {
    "BCIC2A":      {"num_classes": 4,  "num_channels": 22, "num_timepoints": 800,  "dropout": 0.7},
    "MDD":         {"num_classes": 2,  "num_channels": 20, "num_timepoints": 200,  "dropout": 0.7},
    "SEED":        {"num_classes": 3,  "num_channels": 62, "num_timepoints": 400,  "dropout": 0.7},
    "SLEEP":       {"num_classes": 5,  "num_channels": 6,  "num_timepoints": 6000, "dropout": 0.5},
    "BCI_SPEECH":  {"num_classes": 5,  "num_channels": 64, "num_timepoints": 600,  "dropout": 0.7},
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
        print(f"{name:12s}: input {str(dummy.shape):20s} -> output {str(out.shape):15s} | params: {num_params:,}")
