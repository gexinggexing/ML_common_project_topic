import json
for name in ['SLEEP_lite','BCIC2A','BCIC2A_v2','BCIC2A_specialist','BCI_SPEECH','SEED','MDD','CHINESE']:
    try:
        with open('checkpoints/history_'+name+'.json') as f:
            d=json.load(f)
            print(name, 'acc=', d.get('best_val_acc'), 'epoch=', d.get('best_epoch'))
    except: pass
