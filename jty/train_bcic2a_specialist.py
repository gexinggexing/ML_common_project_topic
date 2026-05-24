"""
BCIC2A 专用训练脚本
改进策略:
1. 8-30Hz 带通滤波 (运动想象关键频带)
2. 被试推断 + 被试级别 z-score 归一化
3. 数据增强 (高斯噪声 + 时间平移)
4. EEGNet 架构 (参数量仅 3K，跨被试泛化好)
5. 更长的训练 + 早停
"""
import os
import argparse
import json
import time
import h5py
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from models.bcic2a_specialist import EEGNet, infer_subject_groups, bandpass_filter


# ========== 数据集 (带频带滤波 + 被试归一化 + 增强) ==========
class BCIC2ADataset(Dataset):
    def __init__(self, h5_path, augment=False, noise_std=0.01, shift_max=50, fs=200):
        self.h5_path = h5_path
        self.augment = augment
        self.noise_std = noise_std
        self.shift_max = shift_max

        with h5py.File(self.h5_path, "r") as f:
            X_raw = f["X"][()].astype(np.float32)
            self.y = torch.tensor(f["y"][()], dtype=torch.long)

        # 1. 带通滤波 8-30Hz
        print("  Applying 8-30Hz bandpass filter...")
        X_filtered = bandpass_filter(X_raw, fs=fs, low=8, high=30, order=4)

        # 2. 被试推断归一化
        print("  Applying subject-wise z-score normalization...")
        groups = infer_subject_groups(self.y.numpy(), expected_trials_per_subject_class=20)
        X_norm = X_filtered.copy()
        for g in np.unique(groups):
            idx = groups == g
            if idx.sum() > 1:
                mean = X_norm[idx].mean(axis=0, keepdims=True)
                std = X_norm[idx].std(axis=0, keepdims=True) + 1e-6
                X_norm[idx] = (X_norm[idx] - mean) / std

        self.x = torch.tensor(X_norm, dtype=torch.float32)
        print(f"  Data ready: {self.x.shape}, range=[{self.x.min():.2f}, {self.x.max():.2f}]")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.augment:
            # 高斯噪声
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            # 时间平移
            if self.shift_max > 0:
                shift = np.random.randint(-self.shift_max, self.shift_max + 1)
                if shift != 0:
                    x = torch.roll(x, shifts=shift, dims=-1)
        return x, self.y[idx]


# ========== 训练/评估函数 ==========
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


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--shift_max", type=int, default=50)
    parser.add_argument("--fs", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_dir = f"D:/1/course project/course project/BCIC2A"
    train_path = os.path.join(base_dir, "train.h5")
    val_path = os.path.join(base_dir, "val.h5")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据不存在: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"验证数据不存在: {val_path}")

    print("\n加载训练数据...")
    train_dataset = BCIC2ADataset(train_path, augment=True,
                                   noise_std=args.noise_std, shift_max=args.shift_max, fs=args.fs)
    print("\n加载验证数据...")
    val_dataset = BCIC2ADataset(val_path, augment=False, fs=args.fs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\n{'='*60}")
    print(f"[BCIC2A Specialist] EEGNet with bandpass + subject-wise norm + aug")
    print(f"训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)}")
    print(f"输入: {train_dataset.x.shape} | 4-class motor imagery")
    print(f"增强: noise_std={args.noise_std}, shift_max={args.shift_max}")
    print(f"超参: lr={args.lr}, wd={args.weight_decay}")
    print(f"{'='*60}\n")

    model = EEGNet(num_classes=4, num_channels=22, num_timepoints=800,
                    dropout_rate=0.5, kernel_length=64, F1=8, D=2)
    model = model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

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
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, args.device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_f1": float(train_f1),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
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
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }, f"checkpoints/best_BCIC2A_specialist.pth")
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
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
    print(f"最佳模型: checkpoints/best_BCIC2A_specialist.pth")
    print(f"{'='*60}")

    result = {
        "dataset": "BCIC2A",
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "best_val_f1": float(best_val_f1),
        "args": vars(args),
        "num_params": num_params,
        "training_time_sec": elapsed,
        "history": history,
    }
    with open(f"checkpoints/history_BCIC2A_specialist.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
