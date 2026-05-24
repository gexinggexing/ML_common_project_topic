"""
统一训练脚本 (Multi-Model 版)
为每个数据集自动选择对应的 SOTA 模型架构

用法:
    python train_all.py --dataset BCIC2A --epochs 200
    python train_all.py --dataset SEED --epochs 200 --lr 0.001
    python train_all.py --dataset SLEEP --epochs 100 --batch_size 32
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
from models.multimodel import build_model, DATASET_CONFIGS


# ========== 数据集 ==========
class TrainDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.x = torch.tensor(f["X"][()], dtype=torch.float32)
            self.y = torch.tensor(f["y"][()], dtype=torch.long)
        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ValDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.x = torch.tensor(f["X"][()], dtype=torch.float32)
            self.y = torch.tensor(f["y"][()], dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集名称: BCIC2A / CHINESE / MDD / SEED / SLEEP")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_name = args.dataset.upper()
    base_dir = f"D:/1/course project/course project/{dataset_name}"

    train_path = os.path.join(base_dir, "train.h5")
    val_path = os.path.join(base_dir, "val.h5")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据不存在: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"验证数据不存在: {val_path}")

    # 加载数据
    train_dataset = TrainDataset(train_path)
    val_dataset = ValDataset(val_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\n{'='*50}")
    print(f"数据集: {dataset_name}")
    print(f"模型: {DATASET_CONFIGS[dataset_name].get('model_name', 'Auto-selected')}")
    print(f"训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)}")
    print(f"输入形状: {train_dataset.x.shape} | 类别数: {DATASET_CONFIGS[dataset_name]['num_classes']}")
    print(f"设备: {args.device}")
    print(f"{'='*50}\n")

    # 构建模型
    model, cfg = build_model(dataset_name)
    model = model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 优化器、损失函数、学习率调度
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 早停
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

        # 保存最优模型（按 val loss）
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
            }, f"checkpoints/best_{dataset_name}.pth")
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
    print(f"\n{'='*50}")
    print(f"训练完成！耗时: {elapsed:.1f}s")
    print(f"最佳验证指标 -> Epoch {best_epoch} | Loss: {best_val_loss:.4f} | Acc: {best_val_acc:.4f} | F1: {best_val_f1:.4f}")
    print(f"最佳模型: checkpoints/best_{dataset_name}.pth")
    print(f"{'='*50}")

    # 保存完整训练记录
    result = {
        "dataset": dataset_name,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "best_val_f1": float(best_val_f1),
        "config": cfg,
        "args": vars(args),
        "num_params": num_params,
        "training_time_sec": elapsed,
        "history": history,
    }
    with open(f"checkpoints/history_{dataset_name}.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
