"""
训练和评估模块
"""

import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


class BPRTrainDataset(Dataset):
    """
    BPR 训练数据集
    每个样本返回 (user, pos_item, neg_item)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        train_edges: np.ndarray,
        user_item_dict: Dict[int, set],
        neg_sample_ratio: int = 1
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.train_edges = train_edges
        self.user_item_dict = user_item_dict
        self.neg_sample_ratio = neg_sample_ratio
        self.all_items = set(range(num_items))

    def __len__(self):
        return len(self.train_edges)

    def __getitem__(self, idx):
        user, pos_item = self.train_edges[idx]

        # 负采样：在用户未交互的物品中采样
        neg_item = random.choice(list(self.all_items - self.user_item_dict.get(user, set())))

        return (
            torch.LongTensor([user]),
            torch.LongTensor([pos_item]),
            torch.LongTensor([neg_item])
        )


def evaluate(
    model: nn.Module,
    test_data: np.ndarray,
    user_item_dict_train: Dict[int, set],
    num_users: int,
    num_items: int,
    topk: List[int] = [10, 20],
    batch_size: int = 2048
) -> Dict[str, float]:
    """
    评估模型性能

    Args:
        model: 模型
        test_data: 测试数据 (每行: [user_idx, pos_item_1, pos_item_2, ...])
        user_item_dict_train: 训练集中的用户交互字典
        num_users: 用户数量
        num_items: 物品数量
        topk: 评估的 K 值列表
        batch_size: 批处理大小

    Returns:
        评估指标字典
    """
    model.eval()
    device = next(model.parameters()).device

    # 获取完整的用户和物品表示
    with torch.no_grad():
        representation = model.forward()
        user_rep = representation[:num_users]
        item_rep = representation[num_users:]

    # 计算所有用户与物品的分数矩阵
    num_test_users = len(test_data)
    results = {f'HR@{k}': [] for k in topk}
    results.update({f'NDCG@{k}': [] for k in topk})

    for k in topk:
        hr_sum = 0.0
        ndcg_sum = 0.0

        for data in test_data:
            user_idx = data[0]
            pos_items = set(data[1:].tolist())
            num_pos = len(pos_items)

            if num_pos == 0:
                continue

            # 获取用户表示
            user_vec = user_rep[user_idx:user_idx+1]  # [1, dim]

            # 计算与所有物品的分数
            scores = torch.matmul(user_vec, item_rep.t()).squeeze()  # [num_items]

            # 排除训练集中用户已交互的物品
            train_pos = user_item_dict_train.get(user_idx, set())
            for pos_item in train_pos:
                scores[pos_item] = -1e9

            # 获取 Top-K
            _, topk_items = torch.topk(scores, min(k, num_items))

            # 计算 HR
            num_hit = len(set(topk_items.tolist()) & pos_items)
            hr = num_hit / min(k, num_pos)
            hr_sum += hr

            # 计算 NDCG
            ndcg = 0.0
            max_ndcg = 0.0
            for i in range(min(num_hit, k)):
                max_ndcg += 1 / math.log2(i + 2)
            if max_ndcg > 0:
                for i, item in enumerate(topk_items.tolist()):
                    if item in pos_items:
                        ndcg += 1 / math.log2(i + 2)
                ndcg = ndcg / max_ndcg
            ndcg_sum += ndcg

        results[f'HR@{k}'] = hr_sum / num_test_users
        results[f'NDCG@{k}'] = ndcg_sum / num_test_users

    return results


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    gradient_clip: float = 5.0,
    print_freq: int = 100
) -> Dict[str, float]:
    """
    训练一个 epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch
        gradient_clip: 梯度裁剪阈值
        print_freq: 打印频率

    Returns:
        训练指标
    """
    model.train()

    total_loss = 0.0
    total_bpr_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (users, pos_items, neg_items) in enumerate(pbar):
        users = users.squeeze().to(device)
        pos_items = pos_items.squeeze().to(device)
        neg_items = neg_items.squeeze().to(device)

        # 前向传播
        losses = model.bpr_loss(users, pos_items, neg_items, reg_weight=0.0001)

        loss = losses['total_loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        # 记录损失
        total_loss += loss.item()
        total_bpr_loss += losses['bpr_loss'].item()
        total_reg_loss += losses['reg_loss'].item()
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{total_loss / num_batches:.4f}',
            'bpr': f'{total_bpr_loss / num_batches:.4f}',
            'reg': f'{total_reg_loss / num_batches:.4f}',
        })

    return {
        'loss': total_loss / num_batches,
        'bpr_loss': total_bpr_loss / num_batches,
        'reg_loss': total_reg_loss / num_batches,
    }


class EarlyStopping:
    """
    早停策略
    当验证集指标不再提升时停止训练
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
