import json

# 当前进度汇总
progress = {
    "date": "2026-05-24",
    "datasets": {
        "CHINESE": {"status": "done", "best_acc": 0.495, "target": None, "note": "不再优化"},
        "BCIC2A": {"status": "wip", "best_acc": 0.536, "target": 0.63, "note": "CSP+LDA 42% 失败，v2 53.6%，需新方法"},
        "MDD": {"status": "done", "best_acc": 0.947, "target": None, "note": "v1 保留"},
        "SEED": {"status": "wip", "best_acc": 0.349, "target": None, "note": "v2 32% 更差，过拟合"},
        "SLEEP": {"status": "wip", "best_acc": 0.683, "target": 0.76, "note": "SleepLiteCNN 68.3%，test 预测已生成，需提升"},
        "BCI_SPEECH": {"status": "wip", "best_acc": 0.195, "target": None, "note": "v2 训练反复卡住，未成功"},
    },
    "today_goals": ["BCIC2A -> 63%", "SLEEP -> 76%"],
    "code_dir": "D:\\1\\course project\\course project",
}

with open("D:/1/course project/course project/progress.json", "w") as f:
    json.dump(progress, f, indent=2)
