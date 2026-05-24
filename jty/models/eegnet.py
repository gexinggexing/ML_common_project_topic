"""
EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces
Lawhern et al. 2018 (Journal of Neural Engineering)

Adapted for multi-dataset support. Input shape: (batch, 1, num_channels, num_timepoints)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        num_timepoints: int,
        dropout_rate: float = 0.5,
        kernel_length: int = 64,
        F1: int = 8,
        D: int = 2,
    ):
        """
        Args:
            num_classes: 输出类别数
            num_channels: EEG 通道数 (C)
            num_timepoints: 每个 trial 的时间点数 (T)
            dropout_rate: Dropout 概率
            kernel_length: 时间卷积核长度（默认 64，约对应 0.3s @ 200Hz）
            F1: 第一层时间滤波器数量
            D: 深度乘数（空间滤波器数量 = F1 * D）
        """
        super(EEGNet, self).__init__()

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_timepoints = num_timepoints

        # ------------------- Block 1 -------------------
        # Conv2D: 时间滤波（F1个滤波器，1xkernel_length）
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise Conv2D: 空间滤波（每个时间滤波器配 D 个空间滤波器）
        self.conv2 = nn.Conv2d(
            F1, F1 * D, (num_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # ------------------- Block 2 -------------------
        # Separable Conv2D: 深度可分离卷积 = Depthwise + Pointwise
        # Depthwise: 每个通道单独做时间卷积
        self.sep_conv1 = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )
        # Pointwise: 1x1 卷积混合通道
        self.sep_conv2 = nn.Conv2d(F1 * D, F1 * D, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F1 * D)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # ------------------- 分类器 -------------------
        # 计算 flatten 后的维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, num_timepoints)
            out = self._forward_feature(dummy)
            self.flatten_dim = out.numel()

        self.classifier = nn.Linear(self.flatten_dim, num_classes)

    def _forward_feature(self, x):
        """前向到 flatten 之前，用于计算维度"""
        # Block 1
        x = self.conv1(x)  # (B, 1, C, T) -> (B, F1, C, T-k+1)
        # 手动做 same padding（Conv2d padding=0 时会在右边多切）
        # 重新调整：如果输入不是标准长度，这里做自适应
        x = F.pad(x, (self.conv1.kernel_size[1] // 2, self.conv1.kernel_size[1] // 2))
        x = self.bn1(x)

        x = self.conv2(x)  # (B, F1, C, T) -> (B, F1*D, 1, T)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)  # (B, F1*D, 1, T) -> (B, F1*D, 1, T/4)
        x = self.dropout1(x)

        # Block 2
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)  # -> (B, F1*D, 1, T/32)
        x = self.dropout2(x)
        return x

    def forward(self, x):
        """
        Args:
            x: (batch, 1, num_channels, num_timepoints) 或 (batch, num_channels, num_timepoints)
        """
        # 如果输入缺少 channel 维度，补上
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, C, T) -> (B, 1, C, T)

        x = self._forward_feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ========== 各数据集推荐配置 ==========
DATASET_CONFIGS = {
    "BCIC2A": {
        "num_channels": 22,
        "num_classes": 4,
        "num_timepoints": 800,
        "dropout_rate": 0.5,
        "kernel_length": 64,
        "F1": 8,
        "D": 2,
    },
    "CHINESE": {
        "num_channels": 22,
        "num_classes": 2,
        "num_timepoints": 200,
        "dropout_rate": 0.5,
        "kernel_length": 32,
        "F1": 8,
        "D": 2,
    },
    "MDD": {
        "num_channels": 20,
        "num_classes": 2,
        "num_timepoints": 200,
        "dropout_rate": 0.5,
        "kernel_length": 32,
        "F1": 8,
        "D": 2,
    },
    "SEED": {
        "num_channels": 62,
        "num_classes": 3,
        "num_timepoints": 400,
        "dropout_rate": 0.5,
        "kernel_length": 64,
        "F1": 8,
        "D": 2,
    },
    "SLEEP": {
        "num_channels": 6,
        "num_classes": 5,
        "num_timepoints": 6000,
        "dropout_rate": 0.3,
        "kernel_length": 128,
        "F1": 8,
        "D": 2,
    },
}


def build_eegnet(dataset_name: str):
    """根据数据集名称自动构建 EEGNet"""
    if dataset_name.upper() not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_CONFIGS.keys())}")
    config = DATASET_CONFIGS[dataset_name.upper()]
    return EEGNet(**config), config


# ========== 简单的测试 ==========
if __name__ == "__main__":
    for name in DATASET_CONFIGS.keys():
        model, cfg = build_eegnet(name)
        dummy = torch.zeros(2, cfg["num_channels"], cfg["num_timepoints"])
        out = model(dummy)
        print(f"{name}: input {dummy.shape} -> output {out.shape} | params: {sum(p.numel() for p in model.parameters()):,}")
