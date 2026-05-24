import json, os

files = [
    ('CHINESE', 'checkpoints/history_CHINESE.json'),
    ('BCIC2A', 'checkpoints/history_BCIC2A.json'),
    ('BCIC2A_v2', 'checkpoints/history_BCIC2A_v2.json'),
    ('MDD', 'checkpoints/history_MDD.json'),
    ('MDD_v2', 'checkpoints/history_MDD_v2.json'),
    ('SEED', 'checkpoints/history_SEED.json'),
    ('SEED_v2', 'checkpoints/history_SEED_v2.json'),
    ('BCI_SPEECH', 'checkpoints/history_BCI_SPEECH.json'),
    ('BCI_SPEECH_v2', 'checkpoints/history_BCI_SPEECH_v2.json'),
]

for name, path in files:
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        h = d.get('history', [])
        if h:
            best = max(h, key=lambda x: x['val_acc'])
            print(f"{name:15s} best_val_acc={best['val_acc']:.4f} epoch={best['epoch']} f1={best.get('val_f1',0):.4f}")
        else:
            print(f"{name:15s} no history")
    else:
        print(f"{name:15s} MISSING")

# Also check SLEEP lite v2 if history exists
for extra in ['history_SLEEP_lite.json', 'history_SLEEP_lite_v2.json']:
    if os.path.exists(extra):
        with open(extra) as f:
            d = json.load(f)
        h = d.get('history', [])
        if h:
            best = max(h, key=lambda x: x['val_acc'])
            print(f"{extra:25s} best_val_acc={best['val_acc']:.4f} epoch={best['epoch']}")
    else:
        print(f"{extra:25s} MISSING")
