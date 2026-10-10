"""输入布局归一化：让识别模型同时接受 (B, T, N, C) 与 (B, C, T, N)。

项目内存在两种骨架张量排布：
  - 感知层 / 数据集 / predict.py 产出：(B, T, 27, 3)   （T 帧, N 关节, C=3 通道）
  - recognition/predict.py 对窗口做 permute 后的排布：(B, C=3, T, N=27)

识别层对外承诺"两种都吃"，内部统一转成 (B, T, N, C) 再送进网络，
避免上层因排布不一致而崩溃（排布判断只依据关节数 N，不依赖批大小）。
"""

from __future__ import annotations

import torch

NUM_POINTS = 27


def to_b_t_n_c(x: torch.Tensor, num_points: int = NUM_POINTS) -> torch.Tensor:
    """把输入统一成 (B, T, N, C)。

    判定规则（按优先级）：
      1. x.shape[2] == num_points 且 x.shape[3] != num_points -> 已是 (B,T,N,C)
      2. x.shape[1] == num_points 且 x.shape[2] != num_points -> (B,N,T,C)，转置中间两维
      3. x.shape[1] == num_points 且 x.shape[2] == num_points -> 依最后一维是否为通道维判断
      4. x.shape[1] != num_points 且 x.shape[2] != num_points -> (B,C,T,N) 排布，permute
    """
    if x.dim() != 4:
        raise ValueError(f"识别层输入须为 4 维张量，收到 {tuple(x.shape)}")

    b, d1, d2, d3 = x.shape

    # 情况 1：标准 (B, T, N, C)
    if d2 == num_points and d3 != num_points:
        return x

    # 情况 2：(B, N, T, C)
    if d1 == num_points and d2 != num_points:
        return x.permute(0, 2, 1, 3).contiguous()

    # 情况 3：退化 (B, N, N, C) 或 (B, T, N, N) —— 按最小歧义取 (B,T,N,C)
    if d1 == num_points and d2 == num_points:
        return x

    # 情况 4：(B, C, T, N) —— T 在 dim=2, N 在 dim=3
    if d3 == num_points:
        return x.permute(0, 2, 3, 1).contiguous()

    raise ValueError(
        f"无法识别输入布局（未找到关节维 {num_points}）：{tuple(x.shape)}；"
        f"期望 (B, T, {num_points}, C) 或 (B, C, T, {num_points})"
    )


__all__ = ["to_b_t_n_c", "NUM_POINTS"]
