"""
BCIC2A 训练脚本 v3 - 频域 Bandpower 特征 + MLP

运动想象的关键特征在 mu(8-12Hz) 和 beta(13-30Hz) 波段。
本脚本提取各通道各频段的功率谱密度作为特征，用轻量 MLP 分类。

频带定义:
    Delta: 0.5-4 Hz
    Theta: 4-8 Hz
    Alpha: 8-12 Hz  (mu band, 运动想象关键)
    Beta:  13-30 Hz (beta band, 运动想象关键)
    Gamma: 30-45 Hz
    Total: 0.5-45 Hz

用法:
    python train_bcic2a_bandpower.py
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
        x = f["X"][()].astype(np.float32)  # (N, C, T)
        y = f["y"][()].astype(np.int64)
    return x, y


def compute_bandpower(x, fs=250, nperseg=256):
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
                power = np.trapz(psd[idx], freqs[idx]) if idx.any() else 0.0
                features[i, c * n_bands + b_idx] = power

    # log 变换 + z-score 归一化
    features = np.log(features + 1e-10)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    features = (features - mean) / std
    return features


class BandpowerDataset(Dataset):
    def __init__(self, features, labels):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BandpowerMLP(nn.Module):
    """轻量 MLP，输入: (N, C * n_bands)"""
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
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
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = "D:/1/course project/course project/BCIC2A"
    train_x, train_y = load_data(os.path.join(base_dir, "train.h5"))
    val_x, val_y = load_data(os.path.join(base_dir, "val.h5"))

    print(f"\n{'='*60}")
    print(f"[BCIC2A Bandpower + MLP]")
    print(f"训练: {train_x.shape} | 验证: {val_x.shape}")
    print(f"频域特征提取 (Welch PSD: delta/theta/alpha/beta/gamma)")
    print(f"{'='*60}\n")

    print("Extracting bandpower features...")
    t0 = time.time()
    train_feat = compute_bandpower(train_x, fs=250, nperseg=256)
    val_feat = compute_bandpower(val_x, fs=250, nperseg=256)
    print(f"特征提取耗时: {time.time()-t0:.1f}s")
    print(f"特征维度: {train_feat.shape[1]} (22 channels x 5 bands)")

    train_dataset = BandpowerDataset(train_feat, train_y)
    val_dataset = BandpowerDataset(val_feat, val_y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = args.device
    model = BandpowerMLP(input_dim=train_feat.shape[1], num_classes=4, dropout=args.dropout)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    os.makedirs("checkpoints", exist_ok=True)
    start_time = time.time()

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
            }, "checkpoints/best_BCIC2A_bandpower.pth")
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
    print(f"{'='*60}")

    result = {
        "dataset": "BCIC2A",
        "method": "Bandpower+MLP",
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "best_val_f1": float(best_val_f1),
        "args": vars(args),
        "num_params": num_params,
        "training_time_sec": elapsed,
        "history": history,
    }
    with open("checkpoints/history_BCIC2A_bandpower.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
