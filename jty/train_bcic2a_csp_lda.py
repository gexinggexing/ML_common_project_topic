"""
BCIC2A 专用训练脚本 - CSP 特征提取 + LDA 分类器

这是运动想象跨被试分类的标准方法：
1. 用 Common Spatial Patterns (CSP) 提取空域特征
2. 用 Linear Discriminant Analysis (LDA) 做分类

注意：CSP 是二分类方法，对 4 类用 OVR (One-vs-Rest) 策略，
每类 vs 其他类做一组 CSP 滤波器，最后拼接特征用 LDA 分类。

用法：
    python train_bcic2a_csp_lda.py --n_components 4
"""
import os
import argparse
import json
import time
import h5py
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score


def load_data(h5_path):
    """加载 h5 数据，返回 (N, C, T) 的 numpy 数组"""
    with h5py.File(h5_path, "r") as f:
        x = f["X"][()]
        y = f["y"][()]
    return x.astype(np.float64), y.astype(np.int64)


def cov_matrix(X):
    """计算样本协方差矩阵，X: (C, T)"""
    return np.dot(X, X.T) / np.trace(np.dot(X, X.T))


def csp_filter(X1, X2, n_components=4):
    """
    计算 CSP 空间滤波器
    X1: class 1 的样本，shape (N1, C, T)
    X2: class 2 的样本，shape (N2, C, T)
    返回滤波器矩阵 W (C, n_components*2) 和对应特征值
    """
    # 计算平均协方差矩阵
    C1 = np.mean([cov_matrix(x) for x in X1], axis=0)
    C2 = np.mean([cov_matrix(x) for x in X2], axis=0)

    # 广义特征值分解
    C_sum = C1 + C2
    # 正则化
    C_sum += 0.001 * np.eye(C_sum.shape[0])

    # 白化变换
    eigvals, eigvecs = np.linalg.eigh(C_sum)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # 去掉接近零的特征值
    eigvals = np.maximum(eigvals, 1e-10)
    P = np.dot(eigvecs, np.diag(1.0 / np.sqrt(eigvals)))

    # 变换后的协方差
    S1 = np.dot(np.dot(P.T, C1), P)
    S2 = np.dot(np.dot(P.T, C2), P)

    # 对 S1 做特征值分解
    eigvals_s, eigvecs_s = np.linalg.eigh(S1)
    idx = eigvals_s.argsort()[::-1]
    eigvals_s = eigvals_s[idx]
    eigvecs_s = eigvecs_s[:, idx]

    # 组合滤波器: P * eigvecs_s
    W = np.dot(P, eigvecs_s)

    # 取前 n_components 和后 n_components
    W_selected = np.concatenate([W[:, :n_components], W[:, -n_components:]], axis=1)
    return W_selected


def extract_csp_features(X, W):
    """
    用 CSP 滤波器提取特征
    X: (N, C, T)
    W: (C, n_filters)
    返回 log(var) 特征 (N, n_filters)
    """
    N, C, T = X.shape
    n_filters = W.shape[1]
    features = np.zeros((N, n_filters))
    for i in range(N):
        Z = np.dot(W.T, X[i])  # (n_filters, T)
        var = np.var(Z, axis=1)
        features[i] = np.log(var + 1e-10)
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=4,
                        help="每对 CSP 取前后 n_components 个滤波器")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    base_dir = "D:/1/course project/course project/BCIC2A"
    train_x, train_y = load_data(os.path.join(base_dir, "train.h5"))
    val_x, val_y = load_data(os.path.join(base_dir, "val.h5"))

    print(f"\n{'='*60}")
    print(f"[BCIC2A CSP + LDA]")
    print(f"训练: {train_x.shape} | 验证: {val_x.shape}")
    print(f"类别: {np.unique(train_y)} | n_components={args.n_components}")
    print(f"{'='*60}\n")

    n_classes = len(np.unique(train_y))
    n_filters_per_pair = args.n_components * 2

    start_time = time.time()

    # OVR 策略: 每类 vs 其他类
    all_train_features = []
    all_val_features = []

    for cls in range(n_classes):
        print(f"Computing CSP for class {cls} vs others...")
        X_cls = train_x[train_y == cls]
        X_others = train_x[train_y != cls]

        W = csp_filter(X_cls, X_others, n_components=args.n_components)
        train_feat = extract_csp_features(train_x, W)
        val_feat = extract_csp_features(val_x, W)

        all_train_features.append(train_feat)
        all_val_features.append(val_feat)

    # 拼接所有 CSP 特征
    train_features = np.hstack(all_train_features)  # (N, n_classes * n_filters_per_pair)
    val_features = np.hstack(all_val_features)

    print(f"\n拼接后特征维度: {train_features.shape[1]}")

    # LDA 分类
    lda = LinearDiscriminantAnalysis(solver='svd')
    lda.fit(train_features, train_y)

    train_pred = lda.predict(train_features)
    val_pred = lda.predict(val_features)

    train_acc = accuracy_score(train_y, train_pred)
    train_f1 = f1_score(train_y, train_pred, average="macro", zero_division=0)
    val_acc = accuracy_score(val_y, val_pred)
    val_f1 = f1_score(val_y, val_pred, average="macro", zero_division=0)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"训练完成！耗时: {elapsed:.1f}s")
    print(f"Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"Val   Acc: {val_acc:.4f} | Val   F1: {val_f1:.4f}")
    print(f"{'='*60}")

    # 保存模型和结果
    os.makedirs("checkpoints", exist_ok=True)
    result = {
        "dataset": "BCIC2A",
        "method": "CSP+LDA",
        "n_components": args.n_components,
        "feature_dim": int(train_features.shape[1]),
        "train_acc": float(train_acc),
        "train_f1": float(train_f1),
        "best_val_acc": float(val_acc),
        "best_val_f1": float(val_f1),
        "training_time_sec": elapsed,
        "args": vars(args),
    }
    with open("checkpoints/history_BCIC2A_csp_lda.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"结果保存: checkpoints/history_BCIC2A_csp_lda.json")

    # 保存 LDA 模型
    import joblib
    joblib.dump(lda, "checkpoints/best_BCIC2A_csp_lda.pkl")
    print(f"模型保存: checkpoints/best_BCIC2A_csp_lda.pkl")

    # 保存 CSP 滤波器
    np.savez("checkpoints/best_BCIC2A_csp_filters.npz",
             W_list=[W for W in [csp_filter(train_x[train_y==c], train_x[train_y!=c], args.n_components) for c in range(n_classes)]])
    print(f"滤波器保存: checkpoints/best_BCIC2A_csp_filters.npz")


if __name__ == "__main__":
    main()
