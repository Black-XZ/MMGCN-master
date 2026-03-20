"""
MMGCN-VIDRec 主模型
Multi-modal Graph Convolution Network for Video Recommendation (Frozen Encoder)

该模型结合 ID embedding 和多模态内容特征，通过图卷积网络进行多模态协同过滤推荐。
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .basemodel import ModalGCN, ModalFusion


class MMGCNVIDRec(nn.Module):
    """
    MMGCN-VIDRec 模型

    融合 ID 侧信号和多模态内容侧信号的图卷积推荐模型

    Args:
        num_users: 用户数量
        num_items: 物品数量
        train_edge_index: 训练集的边索引 [2, num_edges]
        v_feat: 视觉特征 [num_items, dim_v_feat] (Frozen)
        t_feat: 文本特征 [num_items, dim_t_feat] (Frozen)
        dim_id: ID embedding 维度
        dim_latent: 潜在表示维度 (用于特征投影)
        num_layers: GCN 层数
        aggr_mode: 聚合方式 ['mean', 'max', 'add']
        concat: 是否拼接融合
        has_id: 是否使用 ID embedding
        modal_fusion: 模态融合方式 ['mean', 'weighted', 'gating']
        use_frozen_encoder: 是否冻结内容编码器
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        train_edge_index: torch.Tensor,
        v_feat: torch.Tensor,
        t_feat: torch.Tensor,
        dim_id: int = 64,
        dim_latent: int = 256,
        num_layers: int = 2,
        aggr_mode: str = 'mean',
        concat: bool = False,
        has_id: bool = True,
        modal_fusion: str = 'mean',
        use_frozen_encoder: bool = True,
    ):
        super(MMGCNVIDRec, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.dim_id = dim_id
        self.dim_latent = dim_latent
        self.num_layers = num_layers
        self.aggr_mode = aggr_mode
        self.concat = concat
        self.has_id = has_id
        self.modal_fusion = modal_fusion
        self.use_frozen_encoder = use_frozen_encoder

        # 准备边索引（无向图：双向边）
        # train_edge_index 格式: (2, num_edges)，第一行用户，第二行物品
        edge_index = train_edge_index.contiguous()
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        # 注册为 buffer，model.to() 时会自动移动到正确设备
        self.register_buffer('edge_index', edge_index)

        # ============== 冻结特征设置 ==============
        # 视觉特征 (Frozen)
        self.v_feat = v_feat
        if not isinstance(v_feat, nn.Parameter):
            self.v_feat = nn.Parameter(v_feat, requires_grad=False)
        self.dim_v_feat = v_feat.shape[1] if v_feat is not None else 0

        # 文本特征 (Frozen)
        self.t_feat = t_feat
        if not isinstance(t_feat, nn.Parameter):
            self.t_feat = nn.Parameter(t_feat, requires_grad=False)
        self.dim_t_feat = t_feat.shape[1] if t_feat is not None else 0

        # ============== ID Embedding ==============
        # 可学习的 ID embedding（用户和物品共享）
        self.id_embedding = nn.Parameter(
            torch.randn(num_users + num_items, dim_id) * 0.1
        )

        # ============== 多模态 GCN ==============
        # 视觉模态 GCN
        if self.dim_v_feat > 0:
            self.v_gcn = ModalGCN(
                edge_index=self.edge_index,
                num_user=num_users,
                num_item=num_items,
                dim_feat=self.dim_v_feat,
                dim_id=dim_id,
                dim_latent=dim_latent,
                aggr_mode=aggr_mode,
                concate=concat,
                num_layer=num_layers,
                has_id=has_id,
            )
        else:
            self.v_gcn = None

        # 文本模态 GCN
        if self.dim_t_feat > 0:
            self.t_gcn = ModalGCN(
                edge_index=self.edge_index,
                num_user=num_users,
                num_item=num_items,
                dim_feat=self.dim_t_feat,
                dim_id=dim_id,
                dim_latent=dim_latent,
                aggr_mode=aggr_mode,
                concate=concat,
                num_layer=num_layers,
                has_id=has_id,
            )
        else:
            self.t_gcn = None

        # ============== 模态融合 ==============
        num_modalities = sum([self.v_gcn is not None, self.t_gcn is not None])
        if num_modalities > 1:
            self.modal_fusion_layer = ModalFusion(
                num_modalities=num_modalities,
                dim_hidden=dim_id,
                fusion_type=modal_fusion,
            )
        else:
            self.modal_fusion_layer = None

        # ============== 最终输出投影 ==============
        # 融合 ID 表示和内容表示
        self.output_projection = nn.Linear(dim_id * 2, dim_id) if concat else nn.Identity()

    def forward(self) -> torch.Tensor:
        """
        前向传播

        Returns:
            融合后的节点表示 [num_nodes, dim_id]
        """
        modal_representations = []

        # 视觉模态 GCN
        if self.v_gcn is not None and self.v_feat is not None:
            v_rep = self.v_gcn(self.v_feat, self.id_embedding)
            modal_representations.append(v_rep)

        # 文本模态 GCN
        if self.t_gcn is not None and self.t_feat is not None:
            t_rep = self.t_gcn(self.t_feat, self.id_embedding)
            modal_representations.append(t_rep)

        # 模态融合
        if len(modal_representations) > 1 and self.modal_fusion_layer is not None:
            representation = self.modal_fusion_layer(modal_representations)
        elif len(modal_representations) == 1:
            representation = modal_representations[0]
        else:
            # 没有多模态特征时，仅使用 ID embedding
            representation = self.id_embedding

        # ID embedding 增强
        if self.has_id:
            representation = representation + self.id_embedding

        return representation

    def get_user_representations(self, full_representation: torch.Tensor) -> torch.Tensor:
        """获取用户表示"""
        return full_representation[:self.num_users]

    def get_item_representations(self, full_representation: torch.Tensor) -> torch.Tensor:
        """获取物品表示"""
        return full_representation[self.num_users:]

    def predict(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        representation: torch.Tensor
    ) -> torch.Tensor:
        """
        计算用户-物品交互分数

        Args:
            users: 用户索引 [batch_size]
            items: 物品索引 [batch_size]
            representation: 融合后的表示 [num_nodes, dim_id]

        Returns:
            交互分数 [batch_size]
        """
        user_emb = representation[users]
        item_emb = representation[items + self.num_users]  # item 索引需要偏移

        # 内积计算分数
        scores = torch.sum(user_emb * item_emb, dim=1)

        return scores

    def loss(
        self,
        user_tensor: torch.Tensor,
        item_tensor: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算 BPR 损失或 Cross-Entropy 损失

        Args:
            user_tensor: 用户索引 [batch_size * 2] (正样本+负样本)
            item_tensor: 物品索引 [batch_size * 2]
            labels: 标签 [batch_size * 2] (1 for pos, 0 for neg)

        Returns:
            损失字典
        """
        representation = self.forward()

        # 获取用户和物品表示
        user_emb = representation[user_tensor]
        item_emb = representation[item_tensor + self.num_users]

        # 计算分数
        scores = torch.sum(user_emb * item_emb, dim=1)

        # Cross-Entropy Loss
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        # L2 正则化
        reg_loss = (
            (self.id_embedding ** 2).mean() +
            sum([p ** 2 for p in self.parameters() if p.requires_grad and p.dim() > 1]).mean()
        )

        total_loss = loss + 0.0001 * reg_loss

        return {
            'total_loss': total_loss,
            'ce_loss': loss,
            'reg_loss': reg_loss,
        }

    def bpr_loss(
        self,
        user_tensor: torch.Tensor,
        pos_item_tensor: torch.Tensor,
        neg_item_tensor: torch.Tensor,
        reg_weight: float = 0.0001
    ) -> Dict[str, torch.Tensor]:
        """
        BPR (Bayesian Personalized Ranking) 损失

        Args:
            user_tensor: 用户索引 [batch_size]
            pos_item_tensor: 正样本物品索引 [batch_size]
            neg_item_tensor: 负样本物品索引 [batch_size]
            reg_weight: 正则化权重

        Returns:
            损失字典
        """
        representation = self.forward()

        # 获取表示
        user_emb = representation[user_tensor]
        pos_item_emb = representation[pos_item_tensor + self.num_users]
        neg_item_emb = representation[neg_item_tensor + self.num_users]

        # 计算 BPR 损失
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)

        # BPR loss = -log(sigmoid(pos_score - neg_score))
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))

        # L2 正则化
        reg_embedding_loss = (
            (self.id_embedding[user_tensor] ** 2).mean() +
            (self.id_embedding[pos_item_tensor + self.num_users] ** 2).mean() +
            (self.id_embedding[neg_item_tensor + self.num_users] ** 2).mean()
        )

        total_loss = bpr_loss + reg_weight * reg_embedding_loss

        return {
            'total_loss': total_loss,
            'bpr_loss': bpr_loss,
            'reg_loss': reg_weight * reg_embedding_loss,
            'pos_scores': pos_scores.mean(),
            'neg_scores': neg_scores.mean(),
        }


def build_mmgcn_vidrec(config: Dict, data: Dict) -> MMGCNVIDRec:
    """
    构建 MMGCN-VIDRec 模型的工厂函数

    Args:
        config: 配置字典
        data: 数据字典

    Returns:
        MMGCNVIDRec 模型实例
    """
    # 转换特征为 Tensor
    v_feat = torch.tensor(data['v_feat'], dtype=torch.float32)
    t_feat = torch.tensor(data['t_feat'], dtype=torch.float32)

    # 转换边索引为 Tensor
    train_edge_index = torch.tensor(data['train_edges'].T, dtype=torch.long)

    model = MMGCNVIDRec(
        num_users=data['num_users'],
        num_items=data['num_items'],
        train_edge_index=train_edge_index,
        v_feat=v_feat,
        t_feat=t_feat,
        dim_id=config['dim_id'],
        dim_latent=config.get('dim_latent', 256),
        num_layers=config['num_layers'],
        aggr_mode=config['aggr_mode'],
        concat=config['concat'],
        has_id=True,
        modal_fusion=config['modal_fusion'],
        use_frozen_encoder=config['use_frozen_encoder'],
    )

    return model
