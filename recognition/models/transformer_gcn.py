"""Transformer + GCN 对比模型（消融 System C/D 用）。

结构：先用若干 ST-GCN 块做空间图特征提取，再将时序帧视为 token 序列送进
Transformer 编码器，最后对序列做全局池化 + 分类头。

输入：接受 (B, T, 27, 3) 或 (B, C=3, T, 27) -> (B, num_classes)
图邻接同 ST-GCN：HAND_EDGES + POSE_EDGES（单主手 + 上身）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import build_adjacency
from .layout import to_b_t_n_c


class GraphConv(nn.Module):
    """单层空间图卷积（节点级线性 + 邻接聚合）。"""

    def __init__(self, in_channels, out_channels, num_points=27, dropout=0.0):
        super().__init__()
        self.num_points = num_points
        A = build_adjacency(num_points, self_loop=True)
        D = A.sum(1)
        D[D == 0] = 1.0
        Dinv = np.diag(1.0 / D.astype(np.float32))
        Ahat = Dinv @ A
        self.register_buffer("Ahat", torch.from_numpy(Ahat.copy()))
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(num_points)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, C, N)
        B, C, N = x.shape
        assert N == self.num_points, f"GraphConv 期望 N={self.num_points}，收到 {N}"
        h = self.linear(x.transpose(1, 2))          # (B, N, C')
        h = h.transpose(1, 2)                       # (B, C', N)
        # 邻接 Ahat 是 2 维 (N,N)，故 einsum 方程只能是 "nv,..."，不能用 3 下标 "nvc"。
        # 语义：out[b,c,n] = Σ_v Ahat[n,v] · h[b,c,v]
        h = torch.einsum("nv,bcv->bcn", self.Ahat, h)  # (B, C', N)
        # BN 作用在节点维：BatchNorm1d(N) 需把 N 放 dim=1
        h = self.bn(h.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        h = self.relu(h)
        h = self.dropout(h)
        return h


class TransformerGCN(nn.Module):
    def __init__(self, num_classes: int = 1000, in_channels: int = 3,
                 num_points: int = 27, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dropout: float = 0.1,
                 gcn_channels: tuple = (64, 128), tcn_kernel: int = 9):
        super().__init__()
        self.num_points = num_points
        self.d_model = d_model

        # 1) 空间 GCN 初步提取：把 (x,y,z) 投影到 d_model，帧内图卷积
        self.input_proj = nn.Linear(in_channels, d_model)
        self.gcn_layers = nn.ModuleList(
            [GraphConv(d_model, d_model, num_points, dropout) for _ in gcn_channels]
        )
        # 时间下采样（可选 1D 卷积）
        self.temp = nn.Conv1d(d_model, d_model, kernel_size=tcn_kernel,
                              padding=(tcn_kernel - 1) // 2)

        # 2) Transformer 编码器（帧作为 token）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3) 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 27, 3) 或 (B, C=3, T, 27) -> (B, num_classes)。"""
        x = to_b_t_n_c(x, self.num_points)
        B, T, N, C = x.shape
        assert N == self.num_points, f"期望 N={self.num_points}，收到 {N}"
        # 投影到 d_model
        h = self.input_proj(x)            # (B, T, N, d_model)
        # 逐帧空间 GCN：把 (B,T) 折叠进批维一次性并行（GraphConv 输入 (B', C, N)）
        h = h.reshape(B * T, N, self.d_model).transpose(1, 2).contiguous()  # (B*T, d_model, N)
        for gcn in self.gcn_layers:
            h = gcn(h)                    # (B*T, d_model, N)
        h = h.transpose(1, 2).reshape(B, T, N, self.d_model)  # (B, T, N, d_model)
        # 时间卷积（沿帧维）：(B,T,N,d_model) -> (B*N, d_model, T) 后 Conv1d
        h = h.permute(0, 2, 3, 1).contiguous()                 # (B, N, d_model, T)
        h = self.temp(h.reshape(B * N, self.d_model, T))       # (B*N, d_model, T)
        h = h.view(B, N, self.d_model, T).permute(0, 3, 1, 2)  # (B, T, N, d_model)
        # 合并节点维作为 token 序列：每个时间步聚合所有关节 -> (B, T, d_model)
        h = h.mean(dim=2)                   # (B, T, d_model)
        h = self.transformer(h)             # (B, T, d_model)
        h = h.mean(dim=1)                   # (B, d_model)
        return self.head(h)                 # (B, num_classes)
