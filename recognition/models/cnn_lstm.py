"""CNN + LSTM 轻量对比模型（消融 / 轻量基线，System C 用）。

思路：把骨骼点特征沿关节维做 1D 卷积（局部关节时序模式），再用 LSTM 建模长时
动态，最后取末隐藏态 + 全局池化做分类。

输入：接受 (B, T, 27, 3) 或 (B, C=3, T, 27) -> (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layout import to_b_t_n_c


class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int = 1000, in_channels: int = 3,
                 num_points: int = 27, cnn_channels: tuple = (32, 64),
                 lstm_hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.num_points = num_points
        self.in_channels = in_channels
        self.lstm_hidden = lstm_hidden

        # 每帧特征：把 (x,y,z) 与关节维拼成特征：每帧 (num_points * in_channels) = 81
        # 把该 81 维作为 CNN 的通道，在时间维 T 上做 1D 卷积（建模局部时序模式）。
        in_ch = num_points * in_channels  # 81
        cnn = []
        for ch in cnn_channels:
            cnn.append(nn.Conv1d(in_ch, ch, kernel_size=3, padding=1))
            cnn.append(nn.BatchNorm1d(ch))
            cnn.append(nn.ReLU(inplace=True))
            cnn.append(nn.MaxPool1d(2))
            in_ch = ch
        self.cnn = nn.Sequential(*cnn)

        # LSTM 输入维 = CNN 实际输出通道数（时间维经 MaxPool 后长度只影响序列长度，不影响特征维）。
        # 这里用真实前向探针测量，而不是硬编码 cnn_channels[-1]：
        # 一旦 cnn_channels / 池化层数变化，input_size 仍与送入 LSTM 的最后一维严格相等。
        probe_len = 2 ** (len(cnn_channels) + 2)  # 保证能被每一级 MaxPool1d(2) 整除
        with torch.no_grad():
            probe = torch.zeros(1, num_points * in_channels, probe_len)
            lstm_input = int(self.cnn(probe).shape[1])
        self.lstm_input_size = lstm_input

        self.lstm = nn.LSTM(
            input_size=lstm_input, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, dropout=dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 27, 3) 或 (B, C=3, T, 27) -> (B, num_classes)。"""
        x = to_b_t_n_c(x, self.num_points)
        B, T, N, C = x.shape
        assert N == self.num_points, f"期望 N={self.num_points}，收到 {N}"
        assert C == self.in_channels, f"期望 C={self.in_channels}，收到 {C}"
        # 逐帧特征展平：(B, T, N*C)
        f = x.reshape(B, T, N * C).contiguous()
        # 通道维移到前面做 1D 卷积：(B, N*C, T)
        f = f.permute(0, 2, 1)
        f = self.cnn(f)                     # (B, ch, T')
        # 回到 (B, T', ch) 作为 LSTM 的序列
        f = f.permute(0, 2, 1).contiguous()  # (B, T', lstm_input)
        # 送入 LSTM 前校验最后一维必须等于 input_size（否则报 "expected X got Y"）
        assert f.shape[-1] == self.lstm.input_size, (
            f"LSTM 输入维不匹配：送入 {f.shape[-1]}，lstm.input_size={self.lstm.input_size}"
        )
        self.lstm.flatten_parameters()
        out, (hn, _) = self.lstm(f)         # out: (B, T', 2*H)
        h_last = torch.cat([hn[-2], hn[-1]], dim=1)  # (B, 2*H)
        h_mean = out.mean(dim=1)            # (B, 2*H)
        h = self.dropout(h_last + h_mean)
        return self.head(h)                # (B, num_classes)
