"""
SLEEP 专用训练脚本 - SleepLiteCNN (纯 CNN, 无 GRU/LSTM)

用法:
    python train_sleep_lite.py --epochs 200 --batch_size 32 --lr 0.001
"""
import os
import argparse
import json
import time
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from models.sleep_lite_cnn import SleepLiteCNN
import numpy as np


# ========== 数据集 (z-score 归一化) ==========
class SleepDataset(Dataset):
    def __init__(self, h5_path, normalize=True):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.x = torch.tensor(f["X"][()], dtype=torch.float32)
            self.y = torch.tensor(f["y"][()], dtype=torch.long)
        assert len(self.x) == len(self.y)
        if normalize:
            self.x = self._zscore(self.x)

    def _zscore(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        return (x - mean) / std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ========== 计算类别权重 ==========
def compute_class_weights(labels):
    unique, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_dir = "D:/1/course project/course project/SLEEP"
    train_path = os.path.join(base_dir, "train.h5")
    val_path = os.path.join(base_dir, "val.h5")

    train_dataset = SleepDataset(train_path, normalize=True)
    val_dataset = SleepDataset(val_path, normalize=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\n{'='*60}")
    print(f"[SLEEP Lite CNN] 纯 CNN 模型")
    print(f"训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)}")
    print(f"输入: {train_dataset.x.shape} | 5-class sleep staging")
    print(f"设备: {args.device}")
    print(f"超参: lr={args.lr}, wd={args.weight_decay}, dropout={args.dropout}")
    print(f"{'='*60}\n")

    model = SleepLiteCNN(num_classes=5, num_channels=6, num_timepoints=6000, dropout=args.dropout)
    model = model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 加权 CrossEntropy (处理类别不平衡)
    class_weights = compute_class_weights(train_dataset.y.numpy())
    print(f"类别权重: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(args.device))

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
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, args.device)
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
            }, "checkpoints/best_SLEEP_lite_v2.pth")
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
    print(f"最佳模型: checkpoints/best_SLEEP_lite_v2.pth")
    print(f"{'='*60}")

    result = {
        "dataset": "SLEEP", "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss), "best_val_acc": float(best_val_acc), "best_val_f1": float(best_val_f1),
        "args": vars(args), "num_params": num_params,
        "training_time_sec": elapsed, "history": history,
    }
    with open("checkpoints/history_SLEEP_lite_v2.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
