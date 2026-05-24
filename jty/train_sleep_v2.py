"""
SLEEP 训练脚本 v2 - SleepLiteCNN + 频域 Bandpower 联合特征

睡眠分期的关键特征在频域功率比（Delta/Theta/Alpha/Beta）。
本脚本在 SleepLiteCNN 的时域特征基础上，拼接频域 bandpower 特征，
让模型同时学习时域形态和频域功率信息。

用法：
    python train_sleep_v2.py --epochs 300 --dropout 0.25
"""
import os
import json
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from scipy import signal


def load_data(h5_path):
    with h5py.File(h5_path, "r") as f:
        x = f["X"][()].astype(np.float32)
        y = f["y"][()].astype(np.int64)
    return x, y


def compute_bandpower(x, fs=100, nperseg=256):
    """
    计算每个样本每个通道的各频段功率
    x: (N, C, T)
    返回: (N, C * n_bands)
    """
    N, C, T = x.shape
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 45),
    }
    n_bands = len(bands)
    features = np.zeros((N, C * n_bands), dtype=np.float32)

    for i in range(N):
        for c in range(C):
            freqs, psd = signal.welch(x[i, c], fs=fs, nperseg=min(nperseg, T))
            for b_idx, (name, (low, high)) in enumerate(bands.items()):
                idx = (freqs >= low) & (freqs < high)
                if idx.any():
                    power = np.trapz(psd[idx], freqs[idx])
                else:
                    power = 0.0
                features[i, c * n_bands + b_idx] = power

    # log 变换 + z-score 归一化
    features = np.log(features + 1e-10)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    features = (features - mean) / std
    return features


class SleepDataset(Dataset):
    def __init__(self, time_data, labels, bp_features=None, normalize=True):
        self.x = torch.tensor(time_data, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.bp = torch.tensor(bp_features, dtype=torch.float32) if bp_features is not None else None
        if normalize:
            mean = self.x.mean(dim=-1, keepdim=True)
            std = self.x.std(dim=-1, keepdim=True) + 1e-6
            self.x = (self.x - mean) / std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.bp is not None:
            return self.x[idx], self.bp[idx], self.y[idx]
        return self.x[idx], self.y[idx]


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
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


class SleepLiteCNNv2(nn.Module):
    """
    SleepLiteCNN + 频域 Bandpower 联合分类
    输入: (B, C, T) 时域 + (B, C*n_bands) 频域
    """
    def __init__(self, num_classes, num_channels, num_timepoints, bp_dim, dropout=0.25):
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
        self.gap = nn.AdaptiveAvgPool1d(1)
        # 联合分类: 128 (时域GAP) + bp_dim (频域)
        self.classifier = nn.Sequential(
            nn.Linear(128 + bp_dim, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, time_x, bp_x):
        x = self.block1(time_x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        time_feat = self.gap(x).view(x.size(0), -1)  # (B, 128)
        # 拼接频域特征
        combined = torch.cat([time_feat, bp_x], dim=1)  # (B, 128 + bp_dim)
        return self.classifier(combined)


def compute_class_weights(labels):
    unique, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in dataloader:
        if len(batch) == 3:
            x, bp, y = batch
            x, bp, y = x.to(device), bp.to(device), y.to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            bp = None
        optimizer.zero_grad()
        out = model(x, bp)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(dataloader), acc, f1


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, bp, y = batch
                x, bp, y = x.to(device), bp.to(device), y.to(device)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                bp = None
            out = model(x, bp)
            loss = criterion(out, y)
            total_loss += loss.item()
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(dataloader), acc, f1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.25)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_dir = "D:/1/course project/course project/SLEEP"
    train_x, train_y = load_data(os.path.join(base_dir, "train.h5"))
    val_x, val_y = load_data(os.path.join(base_dir, "val.h5"))

    print(f"\n{'='*60}")
    print(f"[SLEEP LiteCNN v2 + Bandpower]")
    print(f"训练: {train_x.shape} | 验证: {val_x.shape}")
    print(f"时域 CNN + 频域 Bandpower (delta/theta/alpha/beta/gamma)")
    print(f"设备: {args.device}")
    print(f"超参: lr={args.lr}, wd={args.weight_decay}, dropout={args.dropout}")
    print(f"{'='*60}\n")

    # 提取频域特征
    print("Extracting bandpower features...")
    t0 = time.time()
    train_bp = compute_bandpower(train_x, fs=100, nperseg=256)
    val_bp = compute_bandpower(val_x, fs=100, nperseg=256)
    bp_dim = train_bp.shape[1]
    print(f"Bandpower 特征维度: {bp_dim} (6 channels x 5 bands) | 耗时: {time.time()-t0:.1f}s")

    train_dataset = SleepDataset(train_x, train_y, bp_features=train_bp, normalize=True)
    val_dataset = SleepDataset(val_x, val_y, bp_features=val_bp, normalize=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = args.device
    model = SleepLiteCNNv2(num_classes=5, num_channels=6, num_timepoints=6000, bp_dim=bp_dim, dropout=args.dropout)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 加权 CrossEntropy
    class_weights = compute_class_weights(train_y)
    print(f"类别权重: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    best_epoch = 0
    history = []

    os.makedirs("checkpoints", exist_ok=True)
    start_time = time.time()
    print(f"\n开始训练...")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss), "train_acc": float(train_acc), "train_f1": float(train_f1),
            "val_loss": float(val_loss), "val_acc": float(val_acc), "val_f1": float(val_f1),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            }, "checkpoints/best_SLEEP_v2.pth")
        else:
            patience_counter += 1

        if epoch % 20 == 0 or patience_counter == 0:
            print(f"Epoch {epoch:03d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= args.patience:
            print(f"\n早停触发！最佳 epoch: {best_epoch}")
            break

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"训练完成！耗时: {elapsed:.1f}s")
    print(f"最佳验证指标 -> Epoch {best_epoch} | Loss: {best_val_loss:.4f} | Acc: {best_val_acc:.4f} | F1: {best_val_f1:.4f}")
    print(f"最佳模型: checkpoints/best_SLEEP_v2.pth")
    print(f"{'='*60}")

    result = {
        "dataset": "SLEEP",
        "method": "LiteCNN+Bandpower",
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "best_val_f1": float(best_val_f1),
        "args": vars(args),
        "num_params": num_params,
        "training_time_sec": elapsed,
        "history": history,
    }
    with open("checkpoints/history_SLEEP_v2.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
