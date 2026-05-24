"""
SLEEP Lite 专用测试脚本
"""
import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.sleep_lite_cnn import SleepLiteCNN


class TestDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            self.x = torch.tensor(f["X"][()], dtype=torch.float32)
        # z-score 归一化 (per-sample)
        mean = self.x.mean(dim=-1, keepdim=True)
        std = self.x.std(dim=-1, keepdim=True) + 1e-6
        self.x = (self.x - mean) / std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


def predict(model, dataloader, device):
    model.eval()
    all_preds, all_probs = [], []
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_path = "D:/1/course project/course project/SLEEP/test_x_only.h5"
    ckpt_path = "checkpoints/best_SLEEP_lite.pth"

    test_dataset = TestDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"SLEEP Lite test: {len(test_dataset)} samples, shape={test_dataset.x.shape}")

    model = SleepLiteCNN(num_classes=5, num_channels=6, num_timepoints=6000, dropout=0.3)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Loaded: {ckpt_path} | epoch={ckpt.get('epoch')} | val_acc={ckpt.get('val_acc',0):.4f}")

    preds, probs = predict(model, test_loader, device)

    os.makedirs("results", exist_ok=True)
    result = {
        "dataset": "SLEEP",
        "num_samples": int(len(preds)),
        "predictions": [int(p) for p in preds],
        "probabilities": probs.tolist(),
    }
    with open("results/predictions_SLEEP.json", "w") as f:
        json.dump(result, f, indent=2)
    np.save("results/predictions_SLEEP.npy", preds)
    print(f"Saved: results/predictions_SLEEP.json / .npy")
    print(f"Class dist: {dict(zip(*np.unique(preds, return_counts=True)))}")


if __name__ == "__main__":
    main()
