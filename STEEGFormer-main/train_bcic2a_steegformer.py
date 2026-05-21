import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "benchmark" / "neural_networks" / "models"))
import models_vit_eeg  # noqa: E402


BCIC2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]


class H5EEGDataset(Dataset):
    def __init__(self, path, has_labels=True, standardize=True, label_offset=None):
        self.path = str(path)
        self.has_labels = has_labels
        self.standardize = standardize
        self.label_offset = label_offset
        self._h5 = None
        with h5py.File(self.path, "r") as f:
            self.length = int(f["X"].shape[0])
            self.n_channels = int(f["X"].shape[1])
            self.n_times = int(f["X"].shape[2])
            if has_labels:
                y = f["y"][()]
                self.label_min = int(np.min(y))
                self.label_max = int(np.max(y))
            else:
                self.label_min = None
                self.label_max = None

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self._file()
        x = f["X"][idx].astype(np.float32)
        if self.standardize:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            x = (x - mean) / (std + 1e-6)
        x = torch.from_numpy(x)

        if not self.has_labels:
            return x

        y = int(f["y"][idx])
        if self.label_offset is not None:
            y -= int(self.label_offset)
        return x, torch.tensor(y, dtype=torch.long)


def infer_label_info(*paths):
    ys = []
    for path in paths:
        with h5py.File(path, "r") as f:
            ys.append(f["y"][()])
    y_all = np.concatenate(ys).astype(np.int64)
    unique = np.unique(y_all)
    label_min = int(unique.min())
    label_max = int(unique.max())
    if np.array_equal(unique, np.arange(label_min, label_max + 1)):
        label_offset = label_min
        num_classes = int(label_max - label_min + 1)
    else:
        raise ValueError(f"Labels must be contiguous integers, got {unique.tolist()}")
    return num_classes, label_offset, unique.tolist()


def build_channel_index(n_channels, mode):
    if mode == "arange":
        return torch.arange(n_channels, dtype=torch.long)
    if mode != "bcic2a":
        raise ValueError(f"Unknown channel mode: {mode}")
    if n_channels != len(BCIC2A_CHANNELS):
        raise ValueError(f"BCIC2A channel mode expects {len(BCIC2A_CHANNELS)} channels, got {n_channels}")

    pkl_path = ROOT / "pretrain" / "senloc_file" / "sen_chan_idx.pkl"
    if not pkl_path.exists():
        return torch.arange(len(BCIC2A_CHANNELS), dtype=torch.long)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    mapping = data.get("channels_mapping")
    if not mapping:
        return torch.arange(len(BCIC2A_CHANNELS), dtype=torch.long)

    lower_map = {str(k).lower(): int(v) for k, v in mapping.items()}
    indices = []
    missing = []
    for ch in BCIC2A_CHANNELS:
        key = ch.lower()
        if key in lower_map:
            indices.append(lower_map[key])
        else:
            missing.append(ch)

    if missing:
        print(f"Warning: channel(s) not found in sen_chan_idx.pkl: {missing}. Using arange fallback.")
        return torch.arange(len(BCIC2A_CHANNELS), dtype=torch.long)
    return torch.tensor(indices, dtype=torch.long)


def load_pretrained(model, checkpoint_path):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model = checkpoint.get("model", checkpoint)
    state_dict = model.state_dict()
    for key in ["head.weight", "head.bias"]:
        if key in checkpoint_model and checkpoint_model[key].shape != state_dict[key].shape:
            del checkpoint_model[key]
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {msg.missing_keys}")
    print(f"Unexpected keys: {msg.unexpected_keys}")


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    if hasattr(model, "fc_norm"):
        for param in model.fc_norm.parameters():
            param.requires_grad = True


def finetune_last_blocks(model, n_blocks):
    for param in model.parameters():
        param.requires_grad = False

    n_blocks = int(n_blocks)
    if n_blocks > 0:
        for block in model.blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    for param in model.head.parameters():
        param.requires_grad = True
    if hasattr(model, "fc_norm"):
        for param in model.fc_norm.parameters():
            param.requires_grad = True


def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def run_epoch(model, loader, chan_idx, criterion, optimizer, device, train):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ci = chan_idx.unsqueeze(0).expand(x.shape[0], -1).to(device)

        with torch.set_grad_enabled(train):
            logits = model(x, ci)
            loss = criterion(logits, y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bs = x.shape[0]
        total_loss += float(loss.detach()) * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += bs

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def predict(model, loader, chan_idx, device):
    model.eval()
    preds = []
    probs = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        ci = chan_idx.unsqueeze(0).expand(x.shape[0], -1).to(device)
        logits = model(x, ci)
        prob = torch.softmax(logits, dim=1)
        preds.extend(prob.argmax(dim=1).cpu().numpy().tolist())
        probs.extend(prob.cpu().numpy().tolist())
    return np.asarray(preds, dtype=np.int64), np.asarray(probs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=r"\\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\BCIC2A")
    parser.add_argument("--output-dir", default="outputs/bcic2a_steegformer")
    parser.add_argument("--model", default="vit_small_patch16", choices=["vit_small_patch16", "vit_base_patch16", "vit_large_patch16"])
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--label-offset", type=int, default=None)
    parser.add_argument("--submission-offset", type=int, default=None)
    parser.add_argument("--channel-mode", default="arange", choices=["arange", "bcic2a"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--finetune-last-n", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inferred_num_classes, inferred_label_offset, labels = infer_label_info(data_dir / "train.h5", data_dir / "val.h5")
    num_classes = args.num_classes if args.num_classes is not None else inferred_num_classes
    label_offset = args.label_offset if args.label_offset is not None else inferred_label_offset
    submission_offset = args.submission_offset if args.submission_offset is not None else label_offset

    train_ds = H5EEGDataset(data_dir / "train.h5", has_labels=True, standardize=not args.no_standardize, label_offset=label_offset)
    val_ds = H5EEGDataset(data_dir / "val.h5", has_labels=True, standardize=not args.no_standardize, label_offset=label_offset)
    test_ds = H5EEGDataset(data_dir / "test_x_only.h5", has_labels=False, standardize=not args.no_standardize)

    print(f"Labels: raw={labels}, offset={label_offset}, submission_offset={submission_offset}, num_classes={num_classes}")
    print(f"Train: n={len(train_ds)}, shape=({train_ds.n_channels}, {train_ds.n_times}), labels=[{train_ds.label_min}, {train_ds.label_max}]")
    print(f"Val:   n={len(val_ds)}, shape=({val_ds.n_channels}, {val_ds.n_times}), labels=[{val_ds.label_min}, {val_ds.label_max}]")
    print(f"Test:  n={len(test_ds)}, shape=({test_ds.n_channels}, {test_ds.n_times})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    model = models_vit_eeg.__dict__[args.model](num_classes=num_classes, drop_path_rate=0.1, global_pool=True)
    load_pretrained(model, args.pretrained)
    if args.finetune_last_n is not None:
        finetune_last_blocks(model, args.finetune_last_n)
    elif args.freeze_backbone:
        freeze_backbone(model)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total_params:,}")

    chan_idx = build_channel_index(train_ds.n_channels, args.channel_mode)
    print(f"Channel idx: {chan_idx.tolist()}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    best_path = output_dir / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, chan_idx, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, chan_idx, criterion, optimizer, device, train=False)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "args": vars(args), "val_acc": best_acc}, best_path)

        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d}/{args.epochs} | lr {lr:.2e} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} | best {best_acc:.4f}")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    preds, probs = predict(model, test_loader, chan_idx, device)

    pred_path = output_dir / "test_predictions.csv"
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for i, pred in enumerate(preds):
            writer.writerow([i, int(pred) + int(submission_offset)])

    prob_path = output_dir / "test_probabilities.npy"
    np.save(prob_path, probs)
    print(f"Best val acc: {best_acc:.4f}")
    print(f"Saved predictions: {pred_path}")
    print(f"Saved probabilities: {prob_path}")


if __name__ == "__main__":
    main()
