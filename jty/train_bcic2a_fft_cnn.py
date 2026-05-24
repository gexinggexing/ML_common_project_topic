"""
BCIC2A 训练脚本 v4 - FFT 频谱输入 + 1D-CNN + 数据增强

运动想象的关键在 mu/beta 频段的 ERD，但 bandpower 太粗糙。
这里直接对每个通道做 FFT，保留完整幅度谱作为输入 (C, F)，
让 1D CNN 自动学习关键频段模式。

同时加入时域数据增强：circular shift + 加噪 + 幅度缩放。

用法：
    python train_bcic2a_fft_cnn.py --epochs 300 --dropout 0.4
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


def compute_fft_spectrum(x, fs=250):
    """
    对每个样本的每个通道做 FFT，保留 0-50Hz 幅度谱
    x: (N, C, T)
    返回: (N, C, F) 其中 F ~ T/2 * (50Hz/fs*2) 约等于 T*0.1
    """
    N, C, T = x.shape
    freqs = np.fft.rfftfreq(T, d=1.0/fs)
    # 保留 0-50Hz
    keep_idx = freqs <= 50
    n_keep = keep_idx.sum()
    
    spectrum = np.zeros((N, C, n_keep), dtype=np.float32)
    for i in range(N):
        for c in range(C):
            fft_vals = np.fft.rfft(x[i, c])
            amp = np.abs(fft_vals)
            spectrum[i, c] = amp[keep_idx]
    
    # 归一化
    mean = spectrum.mean(axis=(0, 1), keepdims=True)
    std = spectrum.std(axis=(0, 1), keepdims=True) + 1e-6
    spectrum = (spectrum - mean) / std
    return spectrum


def augment_time(x, shift_range=50, noise_std=0.01, scale_range=0.1):
    """时域增强: circular shift + 加噪 + 随机缩放"""
    if np.random.rand() < 0.5:
        shift = np.random.randint(-shift_range, shift_range + 1)
        x = np.roll(x, shift, axis=-1)
    if np.random.rand() < 0.5:
        x = x + np.random.normal(0, noise_std, x.shape).astype(np.float32)
    if np.random.rand() < 0.3:
        scale = 1.0 + np.random.uniform(-scale_range, scale_range)
        x = x * scale
    return x


class FFTDataset(Dataset):
    def __init__(self, time_data, labels, augment=False, fs=250):
        self.spectrum = compute_fft_spectrum(time_data, fs=fs)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.time_data = time_data if augment else None
        self.augment = augment
        self.fs = fs
        self.num_augment = 3 if augment else 0  # 每个样本扩增3倍

    def __len__(self):
        return len(self.y) * (self.num_augment + 1) if self.augment else len(self.y)

    def __getitem__(self, idx):
        if self.augment:
            orig_idx = idx // (self.num_augment + 1)
            aug_idx = idx % (self.num_augment + 1)
            if aug_idx == 0:
                spec = self.spectrum[orig_idx]
            else:
                x_aug = augment_time(self.time_data[orig_idx].copy())
                # 只对单个样本做 FFT
                freqs = np.fft.rfftfreq(x_aug.shape[-1], d=1.0/self.fs)
                keep_idx = freqs <= 50
                n_keep = keep_idx.sum()
                spec = np.zeros((x_aug.shape[0], n_keep), dtype=np.float32)
                for c in range(x_aug.shape[0]):
                    fft_vals = np.fft.rfft(x_aug[c])
                    spec[c] = np.abs(fft_vals)[keep_idx]
                # 用训练集全局均值/标准差归一化（近似）
                mean = self.spectrum.mean(axis=(0,1))  # (n_keep,)
                std = self.spectrum.std(axis=(0,1)) + 1e-6  # (n_keep,)
                spec = (spec - mean) / std
            return torch.tensor(spec, dtype=torch.float32), self.y[orig_idx]
        else:
            return torch.tensor(self.spectrum[idx], dtype=torch.float32), self.y[idx]


class FFTCNN(nn.Module):
    """
    输入: (B, C, F) - 频谱特征
    C=22 channels, F~80 frequency bins
    """
    def __init__(self, num_classes, num_channels, num_freqs, dropout=0.4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_augment", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = "D:/1/course project/course project/BCIC2A"
    train_x, train_y = load_data(os.path.join(base_dir, "train.h5"))
    val_x, val_y = load_data(os.path.join(base_dir, "val.h5"))

    print(f"\n{'='*60}")
    print(f"[BCIC2A FFT Spectrum + 1D-CNN + Augmentation]")
    print(f"训练: {train_x.shape} | 验证: {val_x.shape}")
    print(f"FFT 频谱输入 (0-50Hz) + 数据增强 (circular shift / noise / scale)")
    print(f"{'='*60}\n")

    # 预计算验证集频谱（不做增强）
    print("Computing FFT spectra...")
    t0 = time.time()
    train_dataset = FFTDataset(train_x, train_y, augment=not args.no_augment, fs=250)
    val_dataset = FFTDataset(val_x, val_y, augment=False, fs=250)
    
    # 获取一个样本看看维度
    sample_spec, _ = train_dataset[0]
    C, F = sample_spec.shape
    print(f"频谱维度: ({C} channels, {F} freq bins) | 耗时: {time.time()-t0:.1f}s")
    print(f"增强后训练样本: {len(train_dataset)} (含 augment x3)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = args.device
    model = FFTCNN(num_classes=4, num_channels=C, num_freqs=F, dropout=args.dropout)
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
            }, "checkpoints/best_BCIC2A_fft_cnn.pth")
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
        "method": "FFT_CNN+Aug",
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "best_val_f1": float(best_val_f1),
        "args": vars(args),
        "num_params": num_params,
        "training_time_sec": elapsed,
        "history": history,
    }
    with open("checkpoints/history_BCIC2A_fft_cnn.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
