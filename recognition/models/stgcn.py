"""ST-GCN 基线：时空图卷积骨干（骨骼模态）。

输入：(B, T, 27, 3)   — B 批大小, T 帧, 27 关节, 3 通道(x,y,z)
输出：(B, num_classes) — 手语词分类 logits

图邻接由 perception/skeleton.py 的 HAND_EDGES（主手 20 边）+ POSE_EDGES（上身 7 边）
拼成 27 点邻接（单主手 + 上身，非双手），见 recognition/models/graph.py。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import build_adjacency, partition_strategy


class STGCNGraphConv(nn.Module):
    """时空图卷积单元（空间图卷积 + 时间 1D 卷积）。

    采用 ST-GCN 经典实现：将邻接按向心分区拆成 K=3 个子图（中心/向心/离心），
    每个子图独立 1x1 空间卷积再求和，随后时间维做 1D 卷积。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, dropout: float = 0.0, num_points: int = 27,
                 residual: bool = True):
        super().__init__()
        assert kernel_size % 2 == 1, "时间卷积核须为奇数"
        self.num_points = num_points
        self.kernel_size = kernel_size
        self.stride = stride

        # 构造分区邻接 A_k (K, N, N)
        A = build_adjacency(num_points, self_loop=True)        # (N, N)
        labels = partition_strategy(num_points)                # (N,)
        K = 3
        A_k = np.zeros((K, num_points, num_points), dtype=np.float32)
        for k in range(K):
            mask = (labels == k)
            # 子图 = 全邻接中"以中心分区 k 为起点"的连接（ST-GCN 标准做法：源点限定在分区 k）
            sub = A * mask[None, :].astype(np.float32)
            d = sub.sum(axis=1, keepdims=True)
            d[d == 0] = 1.0
            A_k[k] = sub / d
        self.register_buffer("A", torch.from_numpy(A_k.copy()))  # (K, N, N)

        # 空间图卷积：对每个子图做 1x1 卷积再求和 -> 用分组/批卷积实现
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True,
        )
        # 时间卷积
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(kernel_size, 1), stride=(stride, 1),
                      padding=((kernel_size - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1),
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, N) -> (B, C', T', N)。"""
        # 分区聚合：对每个子图做节点级线性（1x1 卷积对通道），按邻接加权求和
        # 将 x 与 A_k 结合：xA = sum_k A_k @ (conv_part_k(x))
        B, C, T, N = x.shape
        # 对每个分区：先 1x1 空间卷积得到 (B,C',T,N)，再与 A_k 做节点乘加
        out_parts = []
        feat = self.conv(x)  # (B, C', T, N)
        for k in range(self.A.shape[0]):
            Ak = self.A[k]  # (N, N)
            # 节点聚合：对节点维做 (N,N) 矩阵乘
            # 输出[b,c,t,n] = Σ_v Ak[n,v] · feat[b,c,t,v]
            agg = torch.einsum("nv,bctv->bctn", Ak, feat)  # (B, C', T, N)
            out_parts.append(agg)
        out = sum(out_parts)  # (B, C', T, N)
        out = self.relu(out + self.residual(x))
        out = self.tcn(out)
        return out


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dropout=0.0, num_points=27, residual=True):
        super().__init__()
        self.gcn = STGCNGraphConv(in_channels, out_channels, kernel_size,
                                  stride, dropout, num_points, residual)

    def forward(self, x):
        return self.gcn(x)


class STGCN(nn.Module):
    def __init__(self, num_classes: int = 1000, in_channels: int = 3,
                 num_points: int = 27, hidden_channels: tuple = (64, 128, 256),
                 kernel_size: int = 3, dropout: float = 0.5,
                 temporal_strides: tuple = (1, 1, 1), tcn_kernel: int = 9):
        super().__init__()
        self.num_points = num_points
        self.data_bn = nn.BatchNorm1d(in_channels * num_points)
        self.dropout = dropout

        # 关键：默认各 block 时间维 stride=1，保持 T 不变，使残差分支 T 对齐
        # （避免 T 下采样导致 residual(x) 与 gcn 输出时间维不一致）。
        layers = []
        in_ch = in_channels
        for i, out_ch in enumerate(hidden_channels):
            stride = temporal_strides[i] if i < len(temporal_strides) else 1
            layers.append(STGCNBlock(in_ch, out_ch, kernel_size=3, stride=stride,
                                     dropout=dropout, num_points=num_points))
            in_ch = out_ch
        self.stage = nn.Sequential(*layers)

        # 全局时间 + 空间池化 -> 分类头
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(in_ch, num_classes, kernel_size=(1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 27, 3) -> (B, num_classes)。"""
        B, T, N, C = x.shape
        assert N == self.num_points, f"期望 N={self.num_points}，收到 {N}"
        # (B,T,N,C) -> (B,C,T,N)
        x = x.permute(0, 3, 1, 2).contiguous()
        # 通道-节点展平后做 data BN
        x = x.view(B, C * N, T)
        x = self.data_bn(x)
        x = x.view(B, C, T, N)
        x = self.stage(x)            # (B, C', T', N)
        # 全局平均池化（时间+空间）
        x = F.avg_pool2d(x, x.size()[2:])  # (B, C', 1, 1)
        x = x.view(B, -1)
        x = self.fc(x.unsqueeze(-1).unsqueeze(-1))  # -> (B, num_classes, 1, 1) 后压
        x = x.view(B, -1)
        return x
