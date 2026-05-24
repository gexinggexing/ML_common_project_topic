"""
批量训练所有数据集
用法:
    python run_all.py
或:
    python run_all.py --epochs 200 --batch_size 32
"""
import os
import subprocess
import sys
import argparse


def run(cmd):
    print(f"\n{'='*60}")
    print(f"执行: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"警告: 命令返回非零状态 {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    datasets = ["BCIC2A", "CHINESE", "MDD", "SEED", "SLEEP"]

    print(f"\n开始批量训练 {len(datasets)} 个数据集...")
    print(f"通用配置: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, patience={args.patience}")

    results = {}
    for ds in datasets:
        # 根据数据集调整 batch_size（SLEEP 数据量大可以用更大的）
        bs = 64 if ds == "SLEEP" else args.batch_size
        # 小数据集早停可以更短
        patience = 20 if ds in ["CHINESE", "BCIC2A"] else args.patience
        # 训练epoch也可以调整
        epochs = 100 if ds == "CHINESE" else args.epochs

        cmd = (f"python train_all.py --dataset {ds} --epochs {epochs} "
               f"--batch_size {bs} --lr {args.lr} --patience {patience}")
        ret = run(cmd)
        results[ds] = "成功" if ret == 0 else f"失败 (code={ret})"

    print(f"\n{'='*60}")
    print("批量训练结果汇总:")
    print(f"{'='*60}")
    for ds, status in results.items():
        print(f"  {ds:8s}: {status}")
    print(f"{'='*60}\n")

    # 尝试自动测试（如果checkpoint存在）
    print("尝试为训练成功的数据集生成测试预测...")
    for ds in datasets:
        ckpt = f"checkpoints/best_{ds}.pth"
        if os.path.exists(ckpt):
            cmd = f"python test_all.py --dataset {ds} --checkpoint {ckpt}"
            run(cmd)


if __name__ == "__main__":
    main()
