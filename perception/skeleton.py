"""骨架结构定义与归一化 / 数据增强。

契约：识别层输入为 27 点骨架序列 (T, 27, 3)。
  - 索引 0-20  : 主手 21 点（MediaPipe Hands 标准 21 点）
  - 索引 21-26 : 上身 6 点（Pose 子集：鼻/左肩/右肩/左肘/右肘/左腕）

MediaPipe Hands 21 点定义（与开源实现一致）：
  0 手腕, 1-4 拇指, 5-8 食指, 9-12 中指, 13-16 无名指, 17-20 小指
"""
from __future__ import annotations

import numpy as np

# ---- 骨架尺寸常量 ----
NUM_POINTS = 27
HAND_POINTS = 21
POSE_POINTS = 6
HAND_OFFSET = 0
POSE_OFFSET = 21

# MediaPipe Pose 中上身 6 点对应的索引（33 点 Pose 模型）
POSE_LANDMARK_IDX = [0, 11, 12, 13, 14, 15]  # nose, L肩, R肩, L肘, R肘, L腕

# 归一化参考点
_ROOT = POSE_OFFSET + 0          # 鼻（origin）
_LSH = POSE_OFFSET + 1           # 左肩
_RSH = POSE_OFFSET + 2           # 右肩
_WRIST = POSE_OFFSET + 5         # 左腕（手臂朝向参考）

# 关节骨连接（用于可视化与可选图卷积邻接）
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
POSE_EDGES = [
    (0, 1), (0, 2), (1, 2),          # 鼻-双肩三角
    (1, 3), (3, 5),                  # 左肩-左肘-左腕
    (2, 4), (4, 5),                  # 右肩-右肘-左腕
]


def normalize_sequence(seq, vis=None, max_missing_ratio=0.3):
    """对骨序列做平移 / 缩放 / 旋转不变归一化。

    1. 平移：以鼻为原点
    2. 缩放：以肩宽为尺度（避免尺度差异影响识别 / 图网络）
    3. 旋转：绕 Y 轴旋转，使肩线水平（消除左右视角倾斜）

    Args:
        seq: (T, 27, 3) numpy，缺失帧可全 0
        vis: (T, 27) 可见性（0/1），None 时由非零推断
        max_missing_ratio: 单帧全缺比例阈值，超过则丢弃该帧

    Returns:
        norm: (T', 27, 3)
        vis:  (T', 27)
        kept: list[int] 保留的帧索引
    """
    seq = np.asarray(seq, dtype=np.float32)
    T = seq.shape[0]
    if vis is None:
        vis = (np.abs(seq).sum(axis=-1) > 1e-6).astype(np.float32)

    missing = (vis.sum(axis=-1) < 1.0)
    if missing.mean() > max_missing_ratio:
        # 整体缺失过多，按常规返回（下游可据 vis 判断）
        pass

    # 计算归一化帧参数（只用有效帧，对鼻/双肩/腕求平均）
    valid = ~missing
    if valid.sum() == 0:
        return seq, vis, list(range(T))

    root_all = seq[valid, _ROOT, :]          # (V,3)
    lsh_all = seq[valid, _LSH, :]
    rsh_all = seq[valid, _RSH, :]
    wrist_all = seq[valid, _WRIST, :]

    root = root_all.mean(axis=0)
    lsh = lsh_all.mean(axis=0)
    rsh = rsh_all.mean(axis=0)
    wrist = wrist_all.mean(axis=0)

    scale = float(np.linalg.norm(lsh - rsh))
    if scale < 1e-6:
        scale = 1.0

    # 肩线向量（在 X-Z 平面），绕 Y 轴旋转使其指向 +X
    a = rsh - lsh                       # 从 L 肩指向 R 肩
    ang = np.arctan2(a[2], a[0])        # 绕 Y 轴角
    ca, sa = float(np.cos(-ang)), float(np.sin(-ang))

    def _rot_y(p):
        x, y, z = p[0], p[1], p[2]
        return np.array([ca * x + sa * z, y, -sa * x + ca * z], dtype=np.float32)

    out = np.zeros_like(seq)
    out_vis = vis.copy()
    for t in range(T):
        if missing[t]:
            continue
        # 逐帧以鼻（root 关节）为原点平移，保证每帧平移无关
        p = seq[t] - seq[t][_ROOT]
        p = np.stack([_rot_y(p[i]) for i in range(p.shape[0])], axis=0)
        out[t] = p / scale

    kept = [t for t in range(T) if not missing[t]]
    if len(kept) == 0:
        return seq, vis, list(range(T))
    return out[kept], out_vis[kept], kept


def smooth_sequence(seq, vis=None, window=5):
    """对 (T, 27, 3) 做移动平均（仅对可见点平滑）。"""
    seq = np.asarray(seq, dtype=np.float32)
    T = seq.shape[0]
    if T <= window:
        return seq
    if vis is None:
        vis = (np.abs(seq).sum(axis=-1) > 1e-6).astype(np.float32)
    k = window // 2
    out = seq.copy()
    for t in range(T):
        lo, hi = max(0, t - k), min(T, t + k + 1)
        w = vis[lo:hi].sum(axis=0)                    # (27,)
        acc = (seq[lo:hi] * vis[lo:hi, :, None]).sum(axis=0)  # (27,3)
        denom = np.maximum(w, 1e-6)
        out[t] = acc / denom[:, None]                 # (27,3) / (27,1)
    return out


def augment_sequence(seq, vis=None, rotate_range=(-15, 15),
                     scale_range=(0.9, 1.1), shift_range=(-0.05, 0.05),
                     rng=None):
    """训练时空间增强（在归一化空间操作）。

    - 绕 Y 轴随机旋转（模拟视角）
    - 随机整体缩放
    - 随机平移
    """
    rng = rng or np.random
    seq = np.asarray(seq, dtype=np.float32)
    ang = np.deg2rad(rng.uniform(*rotate_range))
    ca, sa = float(np.cos(ang)), float(np.sin(ang))
    s = rng.uniform(*scale_range)
    sh = rng.uniform(*shift_range, size=3)

    # 绕 Y 轴旋转：(x,z) 旋转；向量化对整个 (T,27,3) 操作
    x = seq[..., 0]
    z = seq[..., 2]
    nx = ca * x + sa * z
    nz = -sa * x + ca * z
    out = seq.copy()
    out[..., 0] = nx
    out[..., 2] = nz
    out = out * s + sh
    return out, vis
