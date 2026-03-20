"""
MMGCN 模型基础组件
基于 torch-geometric 实现图卷积层
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GraphConvolution(nn.Module):
    """
    图卷积层 (Graph Convolution Layer)
    用于在用户-物品二部图上进行消息传递

    公式: H^{l+1} = Activation(D^{-1} A H^l W^l)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggr: str = 'mean'
    ):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggr = aggr

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 [num_nodes, in_features]
            edge_index: 边索引 [2, num_edges]

        Returns:
            更新后的节点特征 [num_nodes, out_features]
        """
        # 矩阵乘法
        x = torch.matmul(x, self.weight)

        # 构建邻接矩阵并归一化
        num_nodes = x.size(0)

        # 方法1: 使用简单的度归一化
        row, col = edge_index[0], edge_index[1]

        # 计算度 (无向图需要考虑双向)
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg = torch.clamp(deg, min=1.0)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

        # 归一化权重
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 聚合邻居节点特征
        out = torch.zeros((num_nodes, self.out_features), device=x.device)

        if self.aggr == 'mean':
            out.index_add_(0, row, x[col] * norm.unsqueeze(1))
        elif self.aggr == 'max':
            # Max pooling 聚合
            out = out.scatter_max(0, row.unsqueeze(1).expand_as(x[col]), x[col])[0]
        elif self.aggr == 'add':
            out.index_add_(0, row, x[col])

        return out

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, aggr={self.aggr}'


class ModalGCN(nn.Module):
    """
    单模态 GCN 模块

    在用户-物品二部图上进行多轮图卷积，学习节点表示

    Args:
        edge_index: 图的边索引 [2, num_edges]
        num_user: 用户数量
        num_item: 物品数量
        dim_feat: 物品特征维度
        dim_id: ID embedding 维度
        dim_latent: 潜在表示维度（可选）
        aggr_mode: 聚合方式 ['mean', 'max', 'add']
        concate: 是否拼接
        num_layer: GCN 层数
        has_id: 是否使用 ID embedding
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_user: int,
        num_item: int,
        dim_feat: int,
        dim_id: int,
        dim_latent: Optional[int] = None,
        aggr_mode: str = 'mean',
        concate: bool = False,
        num_layer: int = 2,
        has_id: bool = True,
    ):
        super(ModalGCN, self).__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.num_nodes = num_user + num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent if dim_latent else dim_feat
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id
        # 注册为 buffer，model.to() 时会自动移动到正确设备
        self.register_buffer('edge_index', edge_index)

        # 用户偏好初始化（可学习）
        self.preference = nn.Parameter(
            torch.randn(num_user, self.dim_latent) * 0.1
        )

        # 特征投影层（将多模态特征投影到统一维度）
        if dim_feat != self.dim_latent:
            self.feat_projection = nn.Linear(dim_feat, self.dim_latent)
        else:
            self.feat_projection = None

        # 创建 GCN 层
        self.gcns = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.g_layers = nn.ModuleList()

        for layer_idx in range(num_layer):
            in_dim = self.dim_latent if layer_idx == 0 else dim_id
            out_dim = dim_id

            self.gcns.append(GraphConvolution(in_dim, out_dim, aggr=aggr_mode))
            self.linears.append(nn.Linear(in_dim, out_dim))

            if concate:
                self.g_layers.append(nn.Linear(out_dim + out_dim, out_dim))
            else:
                self.g_layers.append(nn.Linear(out_dim, out_dim))

    def forward(self, features: torch.Tensor, id_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features: 物品特征 [num_item, dim_feat]
            id_embedding: ID embedding [num_nodes, dim_id]

        Returns:
            节点表示 [num_nodes, dim_id]
        """
        # 投影特征到统一维度
        if self.feat_projection is not None:
            temp_features = self.feat_projection(features)
        else:
            temp_features = features

        # 拼接用户偏好和物品特征
        x = torch.cat([self.preference, temp_features], dim=0)
        x = F.normalize(x, p=2, dim=1)

        # 多层图卷积
        for layer_idx in range(self.num_layer):
            # 图卷积
            h = F.leaky_relu(self.gcns[layer_idx](x, self.edge_index))

            # ID embedding 增强
            x_hat = F.leaky_relu(self.linears[layer_idx](x))
            if self.has_id:
                x_hat = x_hat + id_embedding

            # 融合
            if self.concate:
                x = F.leaky_relu(self.g_layers[layer_idx](torch.cat([h, x_hat], dim=1)))
            else:
                x = F.leaky_relu(self.g_layers[layer_idx](h) + x_hat)

        return x


class ModalFusion(nn.Module):
    """
    模态融合模块
    支持多种融合方式: mean, weighted, gating
    """

    def __init__(
        self,
        num_modalities: int,
        dim_hidden: int,
        fusion_type: str = 'mean'
    ):
        super(ModalFusion, self).__init__()
        self.num_modalities = num_modalities
        self.dim_hidden = dim_hidden
        self.fusion_type = fusion_type

        if fusion_type == 'weighted':
            self.weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        elif fusion_type == 'gating':
            self.gate_fc = nn.Sequential(
                nn.Linear(dim_hidden * num_modalities, dim_hidden),
                nn.Sigmoid()
            )

    def forward(self, modal_representations: list) -> torch.Tensor:
        """
        融合多模态表示

        Args:
            modal_representations: 各模态的表示列表 [num_modalities, num_nodes, dim_hidden]

        Returns:
            融合后的表示 [num_nodes, dim_hidden]
        """
        if self.fusion_type == 'mean':
            return torch.stack(modal_representations, dim=0).mean(dim=0)

        elif self.fusion_type == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            weighted_reps = [
                w * rep for w, rep in zip(weights, modal_representations)
            ]
            return sum(weighted_reps)

        elif self.fusion_type == 'gating':
            concat_rep = torch.cat(modal_representations, dim=1)
            gate = self.gate_fc(concat_rep)
            stacked = torch.stack(modal_representations, dim=0)
            weighted = stacked * gate.unsqueeze(0)
            return weighted.sum(dim=0)

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
