"""
BCIC2A 专用模型 - 带数据增强 + 频带预处理 + EEGNet

针对运动想象任务优化:
1. 8-30Hz 带通滤波 (scipy.signal.butterworth)
2. 数据增强: 高斯噪声 + 时间平移
3. 标准 EEGNet 架构 (Lawhern 2018)
4. 被试推断归一化 (基于 y 的 block 结构推断 group)

输入: (B, num_channels, num_timepoints)
"""
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0. 频带预处理 (8-30Hz 带通滤波)
# ============================================================

def bandpass_filter(data, fs=200, low=8, high=30, order=4):
    """
    对 EEG 数据做 8-30Hz 带通滤波
    data: numpy array (N, C, T) 或 (C, T)
    fs: 采样率 (默认 200Hz，BCIC2A 标准)
    """
    nyq = fs / 2.0
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = scipy.signal.butter(order, [low_norm, high_norm], btype='band')
    if data.ndim == 3:
        filtered = np.stack([scipy.signal.filtfilt(b, a, data[i], axis=-1) for i in range(len(data))])
    else:
        filtered = scipy.signal.filtfilt(b, a, data, axis=-1)
    return filtered.astype(np.float32)


# ============================================================
# 1. 数据增强
# ============================================================

def augment_eeg(x, noise_std=0.01, shift_max=50):
    """
    EEG 数据增强
    x: tensor (C, T)
    """
    # 高斯噪声
    if noise_std > 0:
        noise = torch.randn_like(x) * noise_std
        x = x + noise
    # 时间平移 (circular shift)
    if shift_max > 0:
        shift = np.random.randint(-shift_max, shift_max + 1)
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=-1)
    return x


# ============================================================
# 2. EEGNet (Lawhern 2018, 标准配置)
# ============================================================

class EEGNet(nn.Module):
    """
    标准 EEGNet v4 (Lawhern et al. 2018)
    专为运动想象设计，参数极少，跨被试泛化好

    输入: (B, C, T) 或 (B, 1, C, T)
    """
    def __init__(self, num_classes, num_channels, num_timepoints,
                 dropout_rate=0.5, kernel_length=64, F1=8, D=2):
        super().__init__()
        self.num_channels = num_channels
        self.num_timepoints = num_timepoints

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2 - Separable Conv
        self.sep_conv1 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.sep_conv2 = nn.Conv2d(F1 * D, F1 * D, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F1 * D)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # 自动计算 flatten 维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, num_timepoints)
            dummy = self._pad_input(dummy)
            out = self._forward_feature(dummy)
            self.flatten_dim = out.numel()

        self.classifier = nn.Linear(self.flatten_dim, num_classes)

    def _pad_input(self, x):
        """正确的 same padding: 两侧各补 kernel_length//2"""
        pad = self.conv1.kernel_size[1] // 2
        return F.pad(x, (pad, pad))

    def _forward_feature(self, x):
        # Block 1
        # x 已经在外部 pad 过
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        return x

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._pad_input(x)
        x = self._forward_feature(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# 3. 推断被试分组 (基于 y 的 block 结构)
# ============================================================

def infer_subject_groups(y, expected_trials_per_subject_class=20):
    """
    从 y 的排列推断被试分组
    BCIC2A: 4 类, 每类 180 个 → 假设 9 被试 × 20 trials/类
    返回 group labels: (N,) 每个样本属于哪个被试
    """
    y = np.array(y)
    n = len(y)
    num_classes = len(np.unique(y))
    trials_per_class = n // num_classes  # 180
    num_subjects = trials_per_class // expected_trials_per_subject_class  # 9
    trials_per_block = expected_trials_per_subject_class  # 20

    groups = np.zeros(n, dtype=int)
    # y 是按类分块的: [类3×180, 类2×180, 类0×180, 类1×180]
    # 每个 180 个样本内部可能按被试分 20-trial 块
    # 让我们检测实际块大小

    # 更简单的方法：用 trial 索引对 num_subjects 取模
    # 假设数据是按 [被试1-类3, 被试2-类3, ..., 被试9-类3, 被试1-类2, ...] 排列
    # 但实际排列顺序未知，让我们用保守估计

    # 策略：对每个类内的样本，按顺序每 expected_trials_per_subject_class 个分为一组
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n_c = len(idx)
        n_blocks = max(1, n_c // trials_per_block)
        for b in range(n_blocks):
            start = b * trials_per_block
            end = min((b + 1) * trials_per_block, n_c)
            groups[idx[start:end]] = b % num_subjects

    return groups


# ============================================================
# 4. 被试级别 z-score
# ============================================================

def subject_wise_zscore(X, y, expected_trials_per_subject_class=20):
    """
    对每个被试的数据独立做 z-score
    X: numpy array (N, C, T)
    y: labels (N,)
    返回标准化后的 X
    """
    groups = infer_subject_groups(y, expected_trials_per_subject_class)
    X_norm = X.copy()
    for g in np.unique(groups):
        idx = groups == g
        if idx.sum() > 1:
            mean = X_norm[idx].mean(axis=0, keepdims=True)
            std = X_norm[idx].std(axis=0, keepdims=True) + 1e-6
            X_norm[idx] = (X_norm[idx] - mean) / std
    return X_norm


# ========== 测试 ==========
if __name__ == "__main__":
    model = EEGNet(num_classes=4, num_channels=22, num_timepoints=800)
    dummy = torch.zeros(2, 22, 800)
    out = model(dummy)
    print(f"EEGNet: input {dummy.shape} -> output {out.shape} | params: {sum(p.numel() for p in model.parameters()):,}")
