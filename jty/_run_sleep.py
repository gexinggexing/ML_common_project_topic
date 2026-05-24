import os, json, time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from models.sleep_lite_cnn import SleepLiteCNN

class SleepDataset(Dataset):
    def __init__(self, h5_path, normalize=True):
        with h5py.File(h5_path, 'r') as f:
            self.x = torch.tensor(f['X'][()], dtype=torch.float32)
            self.y = torch.tensor(f['y'][()], dtype=torch.long)
        if normalize:
            mean = self.x.mean(dim=-1, keepdim=True)
            std = self.x.std(dim=-1, keepdim=True) + 1e-6
            self.x = (self.x - mean) / std
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

def compute_class_weights(labels):
    unique, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)

def train_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro', zero_division=0)

def eval_epoch(model, loader, crit, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            total_loss += loss.item()
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro', zero_division=0)

# ===== hyperparams =====
EPOCHS, BATCH, LR, WD, PATIENCE, DROPOUT = 100, 64, 0.001, 0.005, 25, 0.2
SEED, DEVICE = 42, 'cpu'

torch.manual_seed(SEED)
base_dir = 'D:/1/course project/course project/SLEEP'
train_ds = SleepDataset(os.path.join(base_dir, 'train.h5'), normalize=True)
val_ds = SleepDataset(os.path.join(base_dir, 'val.h5'), normalize=True)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

print(f'[SLEEP LiteCNN v2] epochs={EPOCHS} dropout={DROPOUT} wd={WD} patience={PATIENCE}')
print(f'Train={len(train_ds)} Val={len(val_ds)} input={train_ds.x.shape}')

model = SleepLiteCNN(num_classes=5, num_channels=6, num_timepoints=6000, dropout=DROPOUT)
model = model.to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
print(f'Params: {num_params:,}')

cw = compute_class_weights(train_ds.y.numpy())
crit = nn.CrossEntropyLoss(weight=cw.to(DEVICE))
opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

best_loss, best_acc, best_f1, best_epoch = float('inf'), 0.0, 0.0, 0
patience_cnt = 0
history = []
os.makedirs('checkpoints', exist_ok=True)

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    tl, ta, tf1 = train_epoch(model, train_loader, opt, crit, DEVICE)
    vl, va, vf1 = eval_epoch(model, val_loader, crit, DEVICE)
    sched.step()
    history.append({'epoch': epoch, 'train_loss': tl, 'train_acc': ta, 'train_f1': tf1, 'val_loss': vl, 'val_acc': va, 'val_f1': vf1})
    if vl < best_loss:
        best_loss, best_acc, best_f1, best_epoch = vl, va, vf1, epoch
        patience_cnt = 0
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': vl, 'val_acc': va, 'val_f1': vf1}, 'checkpoints/best_SLEEP_lite_v2.pth')
    else:
        patience_cnt += 1
    if epoch % 10 == 0 or patience_cnt == 0:
        lr = opt.param_groups[0]['lr']
        print(f'Ep {epoch:03d}/{EPOCHS} | train_acc={ta:.4f} val_acc={va:.4f} val_f1={vf1:.4f} lr={lr:.6f}')
    if patience_cnt >= PATIENCE:
        print(f'Early stop @ epoch {best_epoch}')
        break

elapsed = time.time() - t0
print(f'DONE | best_epoch={best_epoch} val_acc={best_acc:.4f} val_f1={best_f1:.4f} time={elapsed:.1f}s')

result = {'dataset': 'SLEEP', 'best_epoch': best_epoch, 'best_val_loss': float(best_loss), 'best_val_acc': float(best_acc), 'best_val_f1': float(best_f1), 'num_params': num_params, 'training_time_sec': elapsed, 'history': history}
with open('checkpoints/history_SLEEP_lite_v2.json', 'w') as f:
    json.dump(result, f, indent=2)
