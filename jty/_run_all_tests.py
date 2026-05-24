"""
统一生成所有数据集的 test 预测
使用当前最佳 checkpoint
"""
import os, json, h5py, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):
    def __init__(self, h5_path, normalize=True):
        with h5py.File(h5_path, 'r') as f:
            self.x = torch.tensor(f['X'][()], dtype=torch.float32)
        if normalize:
            mean = self.x.mean(dim=-1, keepdim=True)
            std = self.x.std(dim=-1, keepdim=True) + 1e-6
            self.x = (self.x - mean) / std
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx]

def gen_predictions(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
    return np.array(preds)

configs = {
    'CHINESE': {
        'model_module': 'models.multimodel',
        'build_fn': 'build_model',
        'ckpt': 'checkpoints/best_CHINESE.pth',
        'data': 'D:/1/course project/course project/CHINESE/test_x_only.h5',
        'normalize': True,
    },
    'BCIC2A': {
        'model_module': 'models.multimodel_v2',
        'build_fn': 'build_model',
        'ckpt': 'checkpoints/best_BCIC2A_v2.pth',
        'data': 'D:/1/course project/course project/BCIC2A/test_x_only.h5',
        'normalize': True,
    },
    'MDD': {
        'model_module': 'models.multimodel',
        'build_fn': 'build_model',
        'ckpt': 'checkpoints/best_MDD.pth',
        'data': 'D:/1/course project/course project/MDD/test_x_only.h5',
        'normalize': True,
    },
    'SEED': {
        'model_module': 'models.multimodel',
        'build_fn': 'build_model',
        'ckpt': 'checkpoints/best_SEED.pth',
        'data': 'D:/1/course project/course project/SEED/test_x_only.h5',
        'normalize': True,
    },
    'SLEEP': {
        'model_module': 'models.sleep_lite_cnn',
        'build_fn': 'SleepLiteCNN',
        'ckpt': 'checkpoints/best_SLEEP_lite_v2.pth',
        'data': 'D:/1/course project/course project/SLEEP/test_x_only.h5',
        'normalize': True,
        'extra_args': {'num_classes': 5, 'num_channels': 6, 'num_timepoints': 6000, 'dropout': 0.2},
    },
    'BCI_SPEECH': {
        'model_module': 'models.multimodel_v2',
        'build_fn': 'build_model',
        'ckpt': 'checkpoints/best_BCI_SPEECH_v2.pth',
        'data': 'D:/1/course project/course project/BCI_Speech/test_x_only.h5',
        'normalize': True,
    },
}

device = 'cpu'
os.makedirs('results', exist_ok=True)

for name, cfg in configs.items():
    print(f'\n=== {name} ===')
    if not os.path.exists(cfg['ckpt']):
        print("  SKIP: checkpoint missing: " + cfg['ckpt'])
        continue
    if not os.path.exists(cfg['data']):
        print("  SKIP: data missing: " + cfg['data'])
        continue

    # load model
    mod = __import__(cfg['model_module'], fromlist=[cfg['build_fn']])
    if name == 'SLEEP':
        model_cls = getattr(mod, cfg['build_fn'])
        model = model_cls(**cfg['extra_args'])
    else:
        build_fn = getattr(mod, cfg['build_fn'])
        model, _ = build_fn(name)

    ckpt = torch.load(cfg['ckpt'], map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    # load data
    ds = TestDataset(cfg['data'], normalize=cfg['normalize'])
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    preds = gen_predictions(model, loader, device)

    # save
    out_json = f'results/predictions_{name}.json'
    out_npy = f'results/predictions_{name}.npy'
    json.dump(preds.tolist(), open(out_json, 'w'))
    np.save(out_npy, preds)

    unique, counts = np.unique(preds, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(unique, counts)}
    print(f'  samples={len(preds)} saved -> {out_json}')
    print(f'  class dist: {dist}')

print('\nAll done!')
