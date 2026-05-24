"""
统一测试脚本 (Multi-Model 版)
为无标签的 test_x_only.h5 生成预测结果

用法:
    python test_all.py --dataset BCIC2A --checkpoint checkpoints/best_BCIC2A.pth
"""
import os
import argparse
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.multimodel import build_model


class TestDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.x = torch.tensor(f["X"][()], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


def predict(model, dataloader, device):
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集名称: BCIC2A / CHINESE / MDD / SEED / SLEEP")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重路径，如 checkpoints/best_BCIC2A.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset_name = args.dataset.upper()
    base_dir = f"D:/1/course project/course project/{dataset_name}"
    test_path = os.path.join(base_dir, "test_x_only.h5")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"测试数据不存在: {test_path}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"模型权重不存在: {args.checkpoint}")

    # 加载数据
    test_dataset = TestDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"\n===== Dataset: {dataset_name} =====")
    print(f"测试样本: {len(test_dataset)} | 输入形状: {test_dataset.x.shape}")

    # 加载模型
    model, cfg = build_model(dataset_name)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    print(f"已加载模型: {args.checkpoint}")
    if "val_acc" in checkpoint:
        print(f"验证集最佳指标 -> Epoch {checkpoint.get('epoch', '?')} | Acc: {checkpoint['val_acc']:.4f} | F1: {checkpoint.get('val_f1', 0):.4f}")

    # 预测
    preds, probs = predict(model, test_loader, args.device)

    # 保存结果
    os.makedirs("results", exist_ok=True)
    result_file = f"results/predictions_{dataset_name}.json"

    result = {
        "dataset": dataset_name,
        "num_samples": int(len(preds)),
        "predictions": [int(p) for p in preds],
        "probabilities": probs.tolist(),
    }

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    # 类别分布
    unique, counts = np.unique(preds, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))

    print(f"\n预测完成！结果保存在: {result_file}")
    print(f"预测类别分布: {dist}")

    # 同时保存为 .npy
    np.save(f"results/predictions_{dataset_name}.npy", preds)
    print(f"同时保存为 numpy: results/predictions_{dataset_name}.npy")


if __name__ == "__main__":
    main()
