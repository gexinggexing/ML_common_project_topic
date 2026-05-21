"""
SEED-ViT 模型适配版 (合并所有被试版本)
针对以下需求修改：
1. 读取独立的 train.h5, val.h5, test_x_only.h5 文件
2. 进行3分类（标签：-1, 0, 1）
3. 训练后保存对test_x_only.h5的预测结果
4. 适配62通道 x 400时间点的输入数据
5. 【新增】将所有被试数据合并训练一个通用模型
"""

import argparse
import os
# 【需要修改】设置您要使用的GPU编号
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import sys
import h5py  # 改为使用h5py

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
import torch.optim as optim

from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# ====================== 模型定义部分 ======================
# 这部分基本保持原样，只修改了n_classes参数

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()

        # 【修改】调整网络参数以适配400时间点的输入
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1), padding=(0, 12)),  # 添加padding保持时间维度
            nn.Conv2d(40, 40, (62, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 40)),  # 【关键修改】步长从15改为40，适配400时间点
            nn.Dropout(0.3),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        # 【修改】计算正确的输入维度
        # 经过PatchEmbedding后，输出形状为(batch_size, seq_len, emb_size)
        # 其中seq_len是经过卷积和池化后的时间点数量
        # 对于400个时间点的输入，经过AvgPool2d((1,75), (1,40))后，时间维度是(400-75)/40+1 = 9.125，取整为9
        # 所以seq_len = 9
        # 展平后的维度 = seq_len * emb_size = 9 * 40 = 360
        
        self.fc = nn.Sequential(
            nn.Linear(360, 256),  # 【关键修改】从190 * 40改为360
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)  # 输出3个类别
        )

    def forward(self, x):
        # 展平特征
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out  # 只返回输出

class ViT(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=3, **kwargs):  # 【关键修改】n_classes=3
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

# ====================== 主训练类 (合并所有被试版本) ======================
class SEEDViT_Trainer_Merged:
    def __init__(self, data_root, save_root, subject_ids):
        """
        初始化训练器 (合并所有被试版本)
        Args:
            data_root: .h5文件所在的根目录
            save_root: 结果保存的根目录
            subject_ids: 要处理的所有被试ID列表
        """
        super(SEEDViT_Trainer_Merged, self).__init__()
        
        # 训练参数
        self.batch_size = 128
        self.n_epochs = 100
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        
        # 数据信息
        self.data_root = data_root
        self.save_root = save_root
        self.subject_ids = subject_ids
        
        # 文件路径
        self.train_h5 = os.path.join(data_root, 'train.h5')
        self.val_h5 = os.path.join(data_root, 'val.h5')
        self.test_h5 = os.path.join(data_root, 'test_x_only.h5')
        
        # 创建保存目录
        os.makedirs(save_root, exist_ok=True)
        
        # 初始化模型
        self.model = ViT(emb_size=40, depth=6, n_classes=3).cuda()
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        
        # 损失函数和优化器
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        
        # 标签映射
        # 重要：模型训练时使用0,1,2作为标签
        # 原始数据标签：0(负向), 1(中性), 2(正向)
        # 预测时输出0,1,2，然后映射为-1,0,1
        self.original_to_model = {0: 0, 1: 1, 2: 2}  # 训练时使用
        self.model_to_emotion = {0: -1, 1: 0, 2: 1}  # 预测时映射为情感标签
        
        # 日志文件
        self.log_file = os.path.join(save_root, f"merged_all_subjects_training_log.txt")
        self.log_write = open(self.log_file, "w")
        
    def load_h5_data_for_subject(self, filepath, has_labels=True, subject_id=None):
        """
        加载指定被试的.h5文件数据
        Args:
            filepath: .h5文件路径
            has_labels: 文件是否包含标签
            subject_id: 被试ID
        Returns:
            data: numpy数组，形状为(样本数, 1, 62, 时间点)
            labels: 标签数组（如果没有标签则为None）
        """
        try:
            with h5py.File(filepath, 'r') as f:
                # 检查文件结构
                keys = list(f.keys())
                
                # 尝试不同的键名
                possible_keys = ['x', 'X', 'data', 'eeg', 'EEG']
                data_key = None
                for key in possible_keys:
                    if key in f:
                        data_key = key
                        break
                
                if data_key is None:
                    # 如果直接没有这些键，可能是按subject分组的
                    subject_key = f'subject{subject_id}'
                    if subject_key in f:
                        data = np.array(f[f'{subject_key}/x'])
                        if has_labels and f'{subject_key}/y' in f:
                            labels = np.array(f[f'{subject_key}/y']).flatten()
                        else:
                            labels = None
                    else:
                        # 尝试第一个键
                        first_key = list(f.keys())[0]
                        if isinstance(f[first_key], h5py.Group):
                            data = np.array(f[f'{first_key}/x'])
                            if has_labels and f'{first_key}/y' in f:
                                labels = np.array(f[f'{first_key}/y']).flatten()
                            else:
                                labels = None
                        else:
                            data = np.array(f[first_key])
                            labels = None
                else:
                    data = np.array(f[data_key])
                    if has_labels and 'y' in f:
                        labels = np.array(f['y']).flatten()
                    elif has_labels and 'Y' in f:
                        labels = np.array(f['Y']).flatten()
                    elif has_labels and 'label' in f:
                        labels = np.array(f['label']).flatten()
                    else:
                        labels = None
                
                # 确保数据是4维的: (样本, 通道, 高度, 宽度) -> (样本, 1, 62, 时间点)
                if len(data.shape) == 2:  # (样本, 特征)
                    # 需要知道时间点长度，假设是400
                    n_samples = data.shape[0]
                    data = data.reshape(n_samples, 1, 62, -1)
                elif len(data.shape) == 3:  # (样本, 62, 时间点)
                    data = np.expand_dims(data, axis=1)
                
                return data, labels
                
        except Exception as e:
            print(f"加载文件 {filepath} 时出错 (subject {subject_id}): {e}")
            return None, None
    
    def load_all_subjects_data(self):
        """加载并合并所有被试的数据"""
        all_train_data = []
        all_train_labels = []
        all_val_data = []
        all_val_labels = []
        all_test_data = {}  # 保存每个被试的测试数据
        
        print("开始加载所有被试的数据...")
        
        for subject_id in self.subject_ids:
            print(f"  加载被试 {subject_id} 的数据...")
            
            # 加载训练数据
            train_data, train_labels = self.load_h5_data_for_subject(
                self.train_h5, has_labels=True, subject_id=subject_id
            )
            
            # 加载验证数据
            val_data, val_labels = self.load_h5_data_for_subject(
                self.val_h5, has_labels=True, subject_id=subject_id
            )
            
            # 加载测试数据
            test_data, _ = self.load_h5_data_for_subject(
                self.test_h5, has_labels=False, subject_id=subject_id
            )
            
            if train_data is None or val_data is None or test_data is None:
                print(f"  警告: 被试 {subject_id} 的数据加载失败，跳过")
                continue
            
            # 处理标签：确保标签是0,1,2
            if train_labels is not None:
                train_labels = train_labels.astype(int)
                # 检查标签范围
                unique_labels = np.unique(train_labels)
                if not set(unique_labels).issubset({0, 1, 2}):
                    print(f"  警告: 被试 {subject_id} 的训练标签超出范围: {unique_labels}")
                
            if val_labels is not None:
                val_labels = val_labels.astype(int)
                unique_labels = np.unique(val_labels)
                if not set(unique_labels).issubset({0, 1, 2}):
                    print(f"  警告: 被试 {subject_id} 的验证标签超出范围: {unique_labels}")
            
            # 添加到总数据
            all_train_data.append(train_data)
            all_train_labels.append(train_labels)
            all_val_data.append(val_data)
            all_val_labels.append(val_labels)
            all_test_data[subject_id] = test_data
            
            print(f"    训练: {train_data.shape}, 验证: {val_data.shape}, 测试: {test_data.shape}")
        
        # 合并所有被试的数据
        if all_train_data:
            merged_train_data = np.concatenate(all_train_data, axis=0)
            merged_train_labels = np.concatenate(all_train_labels, axis=0)
            merged_val_data = np.concatenate(all_val_data, axis=0)
            merged_val_labels = np.concatenate(all_val_labels, axis=0)
            
            print(f"\n数据合并完成:")
            print(f"  合并训练集: {merged_train_data.shape}, 标签: {merged_train_labels.shape}")
            print(f"  合并验证集: {merged_val_data.shape}, 标签: {merged_val_labels.shape}")
            print(f"  测试集: 共 {len(all_test_data)} 个被试")
            
            return merged_train_data, merged_train_labels, merged_val_data, merged_val_labels, all_test_data
        else:
            raise ValueError("没有成功加载任何被试的数据")
    
    def prepare_data(self):
        """准备所有被试的合并数据"""
        merged_train_data, merged_train_labels, merged_val_data, merged_val_labels, all_test_data = self.load_all_subjects_data()
        
        # 标准化：使用合并训练数据的统计量
        train_mean = np.mean(merged_train_data)
        train_std = np.std(merged_train_data)
        
        merged_train_data = (merged_train_data - train_mean) / (train_std + 1e-8)
        merged_val_data = (merged_val_data - train_mean) / (train_std + 1e-8)
        
        # 对每个被试的测试集也使用相同的统计量标准化
        for subject_id in all_test_data:
            all_test_data[subject_id] = (all_test_data[subject_id] - train_mean) / (train_std + 1e-8)
        
        # 检查标签分布
        unique_labels, label_counts = np.unique(merged_train_labels, return_counts=True)
        print(f"合并训练集标签分布: {dict(zip(unique_labels, label_counts))}")
        
        unique_labels, label_counts = np.unique(merged_val_labels, return_counts=True)
        print(f"合并验证集标签分布: {dict(zip(unique_labels, label_counts))}")
        
        return merged_train_data, merged_train_labels, merged_val_data, merged_val_labels, all_test_data
    
    def train(self):
        """使用合并的所有被试数据训练一个通用模型"""
        # 准备数据
        train_data, train_labels, val_data, val_labels, all_test_data = self.prepare_data()
        
        # 转换为张量
        train_data_tensor = torch.FloatTensor(train_data).cuda()
        train_labels_tensor = torch.LongTensor(train_labels).cuda()
        val_data_tensor = torch.FloatTensor(val_data).cuda()
        val_labels_tensor = torch.LongTensor(val_labels).cuda()
        
        # 创建数据加载器
        train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 训练循环
        best_val_acc = 0.0
        best_model_state = None
        
        print(f"\n开始训练合并模型 (使用所有被试数据)...")
        self.log_write.write(f"训练合并模型 (使用所有 {len(self.subject_ids)} 个被试的数据)\n")
        self.log_write.write(f"训练样本: {len(train_data)}, 验证样本: {len(val_data)}\n")
        
        for epoch in range(self.n_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(data)
                loss = self.criterion_cls(outputs, target)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 统计
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            # 验证阶段
            val_acc, val_loss = self.evaluate(val_loader)
            
            # 记录日志
            train_acc = 100. * train_correct / train_total
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.n_epochs - 1:
                log_msg = (f'Epoch: {epoch+1:03d}/{self.n_epochs} | '
                          f'Train Loss: {train_loss/len(train_loader):.4f} | '
                          f'Train Acc: {train_acc:.2f}% | '
                          f'Val Loss: {val_loss:.4f} | '
                          f'Val Acc: {100.*val_acc:.2f}%')
                print(log_msg)
                self.log_write.write(log_msg + '\n')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                print(f'  [新最佳模型] 验证准确率: {100.*best_val_acc:.2f}%')
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f'合并模型最佳验证准确率: {100.*best_val_acc:.2f}%')
        self.log_write.write(f'最佳验证准确率: {100.*best_val_acc:.2f}%\n')
        
        # 用训练好的模型为每个被试的测试集进行预测
        all_predictions = {}
        for subject_id, test_data in all_test_data.items():
            test_data_tensor = torch.FloatTensor(test_data).cuda()
            subject_predictions = self.predict_test_set(test_data_tensor)
            all_predictions[subject_id] = subject_predictions
            
            # 保存每个被试的预测结果
            self.save_predictions(subject_predictions, subject_id, best_val_acc)
        
        # 关闭日志文件
        self.log_write.close()
        
        return best_val_acc, all_predictions
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                outputs = self.model(data)
                loss = self.criterion_cls(outputs, target)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return accuracy, avg_loss
    
    def predict_test_set(self, test_data):
        """预测测试集"""
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            # 分批预测
            batch_size = min(self.batch_size, len(test_data))
            for i in range(0, len(test_data), batch_size):
                end_idx = min(i + batch_size, len(test_data))
                batch = test_data[i:end_idx]
                
                outputs = self.model(batch)
                predictions = torch.argmax(outputs, dim=1)
                
                # 将预测结果从模型输出(0,1,2)映射为情感标签(-1,0,1)
                predictions_np = predictions.cpu().numpy()
                predictions_mapped = np.array([self.model_to_emotion[p] for p in predictions_np])
                all_predictions.append(predictions_mapped)
        
        # 合并所有预测
        if all_predictions:
            all_predictions = np.concatenate(all_predictions)
        else:
            all_predictions = np.array([])
        
        return all_predictions
    
    def save_predictions(self, predictions, subject_id, val_acc):
        """保存预测结果"""
        # 保存为.npy文件
        npy_path = os.path.join(self.save_root, f"merged_subject{subject_id}_predictions.npy")
        np.save(npy_path, predictions)
        
        # 保存为.txt文件
        txt_path = os.path.join(self.save_root, f"merged_subject{subject_id}_predictions.txt")
        with open(txt_path, 'w') as f:
            f.write(f"受试者: {subject_id}\n")
            f.write(f"模型类型: 合并所有被试训练的通用模型\n")
            f.write(f"验证集最佳准确率: {val_acc:.4f} ({100.*val_acc:.2f}%)\n")
            f.write(f"测试集样本数: {len(predictions)}\n")
            f.write(f"预测结果分布:\n")
            f.write(f"  -1 (负向): {np.sum(predictions==-1)} 个\n")
            f.write(f"   0 (中性): {np.sum(predictions==0)} 个\n")
            f.write(f"   1 (正向): {np.sum(predictions==1)} 个\n")
            f.write("-" * 50 + "\n")
            f.write("详细预测结果:\n")
            for i, pred in enumerate(predictions):
                f.write(f"样本{i:04d}: {pred}\n")
        
        print(f"被试 {subject_id} 预测结果已保存:")
        print(f"  - {npy_path} (npy格式)")
        print(f"  - {txt_path} (txt格式)")
        print(f"  预测分布: -1: {np.sum(predictions==-1)}, 0: {np.sum(predictions==0)}, 1: {np.sum(predictions==1)}")

# ====================== 主函数 ======================
def main():
    """主函数"""
    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 【需要修改】配置参数
    DATA_ROOT = "data/SEED"  # 包含train.h5, val.h5, test_x_only.h5的目录
    SAVE_ROOT = "results_merged"  # 结果保存目录
    SUBJECT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 要处理的所有被试
    
    print("=" * 70)
    print("SEED-ViT 模型训练与测试 (合并所有被试版本)")
    print(f"开始时间: {datetime.datetime.now()}")
    print(f"数据目录: {DATA_ROOT}")
    print(f"保存目录: {SAVE_ROOT}")
    print(f"处理被试: {SUBJECT_IDS}")
    print("=" * 70)
    
    # 检查数据文件是否存在
    required_files = ['train.h5', 'val.h5', 'test_x_only.h5']
    for f in required_files:
        filepath = os.path.join(DATA_ROOT, f)
        if not os.path.exists(filepath):
            print(f"错误: 找不到文件 {filepath}")
            print("请确保:")
            print(f"1. DATA_ROOT 路径正确 (当前: {DATA_ROOT})")
            print(f"2. 文件 {f} 存在于该目录中")
            return
    
    # 创建汇总结果文件
    summary_file = os.path.join(SAVE_ROOT, "merged_all_subjects_summary.txt")
    os.makedirs(SAVE_ROOT, exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as sf:
        sf.write("SEED数据集情感分类实验结果汇总 (合并所有被试版本)\n")
        sf.write("=" * 60 + "\n")
        sf.write(f"实验时间: {datetime.datetime.now()}\n")
        sf.write(f"模型: EEG-Conformer (ViT变体) - 合并所有被试训练\n")
        sf.write(f"数据路径: {DATA_ROOT}\n")
        sf.write(f"被试数量: {len(SUBJECT_IDS)}\n")
        sf.write("=" * 60 + "\n\n")
    
    start_time = time.time()
    
    try:
        # 创建合并训练器实例
        trainer = SEEDViT_Trainer_Merged(
            data_root=DATA_ROOT,
            save_root=SAVE_ROOT,
            subject_ids=SUBJECT_IDS
        )
        
        # 训练并获取所有被试的预测
        best_val_acc, all_predictions = trainer.train()
        
        # 记录到汇总文件
        with open(summary_file, 'a', encoding='utf-8') as sf:
            sf.write(f"合并模型验证集最佳准确率: {best_val_acc:.4f} ({100.*best_val_acc:.2f}%)\n")
            sf.write(f"训练总耗时: {time.time()-start_time:.1f} 秒\n\n")
            
            sf.write("各被试测试集预测结果:\n")
            sf.write("-" * 50 + "\n")
            
            total_samples = 0
            label_counts = {-1: 0, 0: 0, 1: 0}
            
            for subject_id, predictions in all_predictions.items():
                n_samples = len(predictions)
                n_neg = np.sum(predictions == -1)
                n_neu = np.sum(predictions == 0)
                n_pos = np.sum(predictions == 1)
                
                total_samples += n_samples
                label_counts[-1] += n_neg
                label_counts[0] += n_neu
                label_counts[1] += n_pos
                
                sf.write(f"被试 {subject_id}:\n")
                sf.write(f"  测试样本数: {n_samples}\n")
                sf.write(f"  负向(-1): {n_neg} ({100.*n_neg/n_samples:.1f}%)\n")
                sf.write(f"  中性(0): {n_neu} ({100.*n_neu/n_samples:.1f}%)\n")
                sf.write(f"  正向(1): {n_pos} ({100.*n_pos/n_samples:.1f}%)\n")
                sf.write("\n")
            
            sf.write("\n" + "=" * 60 + "\n")
            sf.write("总体统计结果\n")
            sf.write("=" * 60 + "\n")
            sf.write(f"总测试样本数: {total_samples}\n")
            sf.write(f"负向(-1)总数: {label_counts[-1]} ({100.*label_counts[-1]/total_samples:.1f}%)\n")
            sf.write(f"中性(0)总数: {label_counts[0]} ({100.*label_counts[0]/total_samples:.1f}%)\n")
            sf.write(f"正向(1)总数: {label_counts[1]} ({100.*label_counts[1]/total_samples:.1f}%)\n")
            sf.write(f"结束时间: {datetime.datetime.now()}\n")
        
        print(f"\n{'='*60}")
        print("合并模型训练完成!")
        print(f"最佳验证准确率: {100.*best_val_acc:.2f}%")
        print(f"总耗时: {time.time()-start_time:.1f}秒")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"训练合并模型时出错: {e}")
        import traceback
        traceback.print_exc()
        
        with open(summary_file, 'a', encoding='utf-8') as sf:
            sf.write(f"训练合并模型时出错: {str(e)}\n")
            sf.write("-" * 50 + "\n")
    
    print(f"\n结果已保存到: {SAVE_ROOT}")
    print(f"汇总文件: {summary_file}")

if __name__ == "__main__":
    print(f"开始时间: {datetime.datetime.now()}")
    main()
    print(f"\n结束时间: {datetime.datetime.now()}")