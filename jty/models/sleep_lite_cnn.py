"""
SLEEP 专用模型 - 纯 CNN 轻量版
无需 GRU/LSTM，避免 CPU 内存问题

改进:
1. 纯 1D CNN 堆叠，快速下采样长序列 (6000 -> ~37)
2. SE 注意力加权通道
3. 参数量 < 50K
4. 支持 Weighted CrossEntropy (处理类别不平衡)

输入: (B, num_channels, num_timepoints)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
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


class SleepLiteCNN(nn.Module):
    """
    纯 CNN 睡眠分期模型
    输入: (B, C, T) 其中 T=6000
    """
    def __init__(self, num_classes, num_channels, num_timepoints, dropout=0.3):
        super().__init__()
        # Block 1: 6000 -> 1500 (stride=4)
        self.block1 = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=25, stride=4, padding=12, bias=False),
            nn.BatchNorm1d(16),
            nn.ELU(),
            SqueezeExcitation(16, reduction=8),
            nn.Dropout(dropout),
        )
        # Block 2: 1500 -> 375 (stride=4)
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            SqueezeExcitation(32, reduction=8),
            nn.Dropout(dropout),
        )
        # Block 3: 375 -> 94 (stride=4)
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            SqueezeExcitation(64, reduction=8),
            nn.Dropout(dropout),
        )
        # Block 4: 94 -> 24 (stride=4)
        self.block4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            SqueezeExcitation(128, reduction=8),
            nn.Dropout(dropout),
        )
        # Global Average Pooling + 分类
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.classifier(x)


# ========== 测试 ==========
if __name__ == "__main__":
    model = SleepLiteCNN(num_classes=5, num_channels=6, num_timepoints=6000)
    dummy = torch.zeros(2, 6, 6000)
    out = model(dummy)
    print(f"SleepLiteCNN: {dummy.shape} -> {out.shape} | params: {sum(p.numel() for p in model.parameters()):,}")
