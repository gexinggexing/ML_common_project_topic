import torch, os

for name, path in [
    ('CHINESE', 'checkpoints/best_CHINESE.pth'),
    ('BCIC2A_v1', 'checkpoints/best_BCIC2A.pth'),
    ('BCIC2A_v2', 'checkpoints/best_BCIC2A_v2.pth'),
    ('MDD_v1', 'checkpoints/best_MDD.pth'),
    ('MDD_v2', 'checkpoints/best_MDD_v2.pth'),
    ('SEED_v1', 'checkpoints/best_SEED.pth'),
    ('SEED_v2', 'checkpoints/best_SEED_v2.pth'),
    ('SLEEP_lite', 'checkpoints/best_SLEEP_lite.pth'),
    ('SLEEP_lite_v2', 'checkpoints/best_SLEEP_lite_v2.pth'),
    ('BCI_SPEECH_v1', 'checkpoints/best_BCI_SPEECH.pth'),
    ('BCI_SPEECH_v2', 'checkpoints/best_BCI_SPEECH_v2.pth'),
]:
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu')
        ep = ckpt.get('epoch', '?')
        va = ckpt.get('val_acc', 0)
        print(f'{name:15s} epoch={ep} val_acc={va:.4f}')
    else:
        print(f'{name:15s} MISSING')
