import json

for ds in ['CHINESE', 'BCIC2A', 'SEED']:
    with open('checkpoints/history_{}.json'.format(ds)) as f:
        h = json.load(f)
    hist = h['history']
    print('=== {} ==='.format(ds))
    for i, e in enumerate(hist):
        if i == 0 or (i+1) % 10 == 0 or i == len(hist)-1:
            print('  Epoch {:3d}: TrainAcc={:.4f} ValAcc={:.4f} ValF1={:.4f}'.format(
                e['epoch'], e['train_acc'], e['val_acc'], e['val_f1']))
    print('  Best: Epoch {} ValAcc={:.4f}'.format(h['best_epoch'], h['best_val_acc']))
    print()
