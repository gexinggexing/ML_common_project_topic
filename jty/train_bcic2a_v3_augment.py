"""
BCIC2A 训练脚本 v3 - 时域数据增强 + 改进 DeepConvNetLite

在 v2 (53.6%) 基础上改进:
1. 训练时加入数据增强: circular shift + 高斯噪声 + 幅度缩放
2. 降低 dropout (0.7->0.5) + 降低 weight_decay (0.1->0.01) + 提高 lr (0.0005->0.001)
3. 更长的 epochs (300) + 更大的 patience (50)

用法:
    python train_bcic2a_v3_augment.py
"""
import os
import argparse
import json
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from models.multimodel_v2 import build_model


def load_data(h5_path):
    with h5py.File(h5_path, "r") as f:
        x = f["X"][()].astype(np.float32)
        y = f["y"][()].astype(np.int64)
    return x, y


def augment_time(x, shift_range=80, noise_std=0.02, scale_range=0.15):
    """时域增强"""
    if np.random.rand() < 0.5:
        shift = np.random.randint(-shift_range, shift_range + 1)
        x = np.roll(x, shift, axis=-1)
    if np.random.rand() < 0.5:
        x = x + np.random.normal(0, noise_std, x.shape).astype(np.float32)
    if np.random.rand() < 0.3:
        scale = 1.0 + np.random.uniform(-scale_range, scale_range)
        x = x * scale
    return x


class AugDataset(Dataset):
    def __init__(self, x_data, y_data, augment=False, num_aug=2, normalize=True):
        self.x_raw = x_data
        self.y = torch.tensor(y_data, dtype=torch.long)
        self.augment = augment
        self.num_aug = num_aug
        self.normalize = normalize

    def __len__(self):
        return len(self.y) * (self.num_aug + 1) if self.augment else len(self.y)

    def _normalize(self, x):
        if self.normalize:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + 1e-6
            x = (x - mean) / std
        return x

    def __getitem__(self, idx):
        if self.augment:
            orig_idx = idx // (self.num_aug + 1)
            aug_idx = idx % (self.num_aug + 1)
            if aug_idx == 0:
                x = torch.tensor(self.x_raw[orig_idx], dtype=torch.float32)
            else:
                x_aug = augment_time(self.x_raw[orig_idx].copy())
                x = torch.tensor(x_aug, dtype=torch.float32)
            return self._normalize(x), self.y[orig_idx]
        else:
            x = torch.tensor(self.x_raw[idx], dtype=torch.float32)
            return self._normalize(x), self.y[idx]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--no_augment", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_name = "BCIC2A"
    base_dir = f"D:/1/course project/course project/{dataset_name}"
    train_x, train_y = load_data(os.path.join(base_dir, "train.h5"))
    val_x, val_y = load_data(os.path.join(base_dir, "val.h5"))

    print(f"\n{'='*60}")
    print(f"[BCIC2A v3] 时域增强 + 改进 DeepConvNetLite")
    print(f"训练: {train_x.shape} | 验证: {val_x.shape}")
    print(f"数据增强: {'开' if not args.no_augment else '关'} (circular shift + noise + scale)")
    print(f"超参: lr={args.lr}, wd={args.weight_decay}, dropout={args.dropout}")
    print(f"{'='*60}\n")

    train_dataset = AugDataset(train_x, train_y, augment=not args.no_augment, num_aug=2, normalize=True)
    val_dataset = AugDataset(val_x, val_y, augment=False, normalize=True)
    print(f"增强后训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = args.device
    model, cfg = build_model(dataset_name)
    # 修改 dropout
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
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
            }, "checkpoints/best_BCIC2A_v3.pth")
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
        "dataset": dataset_name,
        "method": "DeepConvNetLite+Aug",
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "best_val_f1": float(best_val_f1),
        "args": vars(args),
        "num_params": num_params,
        "training_time_sec": elapsed,
        "history": history,
    }
    with open("checkpoints/history_BCIC2A_v3.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
