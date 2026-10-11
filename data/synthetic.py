"""合成手语骨架数据集生成器（临时替代 ISW-1000）。

生成 K=60 个中文孤立手语词类的 27 点骨架序列 (T, 27, 3) float32，
严格对齐 tests/contract.py 的骨骼布局与 perception/skeleton.normalize_sequence
的归一化（鼻原点 / 肩宽尺度 / 绕 Y 轴使肩线水平）。

每个类用**参数化轨迹**驱动主手 21 点相对上身 6 点的运动（画圆 / 直线 /
上下摆 / 左右摆 / 对角 / 8 字 / 弧线 / 抖动 等），参数（半径 / 频率 /
相位 / 方向 / 中心 / 手势开合度）按类索引确定，使 ST-GCN 能学到类间差异。
固定随机种子保证可复现。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from perception.skeleton import normalize_sequence

# ---------------------------------------------------------------------------
# 60 个中文孤立手语词 gloss（写进 classes.txt）
# ---------------------------------------------------------------------------
CLASSES = [
    "你", "我", "他", "她", "好", "谢谢", "请", "是", "不", "吃",
    "喝", "水", "饭", "书", "笔", "车", "家", "学校", "老师", "学生",
    "朋友", "爱", "想", "知道", "时间", "今天", "明天", "天气", "冷", "热",
    "大", "小", "上", "下", "开", "关", "灯", "门", "走", "坐",
    "站", "说", "听", "看", "做", "帮", "买", "卖", "问", "答",
    "红", "绿", "白", "黑", "一", "二", "三", "四", "五", "六",
]
NUM_CLASSES = len(CLASSES)  # 60

# ---------------------------------------------------------------------------
# 生成超参
# ---------------------------------------------------------------------------
SEED = 20261010                      # 固定种子，保证可复现
SPLITS = {"train": 24, "val": 6, "test": 6}  # 每类样本数（共 36/类）
T_MIN, T_MAX = 30, 90                # 每样本帧数随机区间

HAND_OFFSET = 0
POSE_OFFSET = 21

# ---------------------------------------------------------------------------
# 手部局部坐标（wrist 在原点，手指朝上 +y，掌心朝 +z）
# ---------------------------------------------------------------------------
OPEN_HAND = np.array([
    (0.00, 0.00, 0.00),  # 0  wrist
    (0.05, -0.02, 0.00), # 1  thumb cmc
    (0.08, 0.00, 0.00),  # 2  thumb mcp
    (0.10, 0.03, 0.00),  # 3  thumb ip
    (0.11, 0.06, 0.00),  # 4  thumb tip
    (-0.03, 0.06, 0.00), # 5  index mcp
    (-0.03, 0.11, 0.00), # 6  index pip
    (-0.03, 0.15, 0.00), # 7  index dip
    (-0.03, 0.18, 0.00), # 8  index tip
    (0.00, 0.06, 0.00),  # 9  middle mcp
    (0.00, 0.12, 0.00),  # 10 middle pip
    (0.00, 0.16, 0.00),  # 11 middle dip
    (0.00, 0.19, 0.00),  # 12 middle tip
    (0.03, 0.06, 0.00),  # 13 ring mcp
    (0.03, 0.11, 0.00),  # 14 ring pip
    (0.03, 0.15, 0.00),  # 15 ring dip
    (0.03, 0.18, 0.00),  # 16 ring tip
    (0.06, 0.05, 0.00),  # 17 pinky mcp
    (0.06, 0.09, 0.00),  # 18 pinky pip
    (0.06, 0.12, 0.00),  # 19 pinky dip
    (0.06, 0.14, 0.00),  # 20 pinky tip
], dtype=np.float32)

# 握拳：手指向掌心卷曲（z 负向）并缩短
FIST_HAND = np.array([
    (0.00, 0.00, 0.00),
    (0.05, -0.02, 0.00),
    (0.07, -0.01, 0.01),
    (0.07, 0.00, 0.03),
    (0.06, 0.01, 0.05),
    (-0.03, 0.05, -0.01),
    (-0.03, 0.04, -0.03),
    (-0.03, 0.03, -0.05),
    (-0.03, 0.02, -0.06),
    (0.00, 0.05, -0.01),
    (0.00, 0.04, -0.03),
    (0.00, 0.03, -0.05),
    (0.00, 0.02, -0.06),
    (0.03, 0.05, -0.01),
    (0.03, 0.04, -0.03),
    (0.03, 0.03, -0.05),
    (0.03, 0.02, -0.06),
    (0.06, 0.04, -0.01),
    (0.06, 0.03, -0.03),
    (0.06, 0.02, -0.04),
    (0.06, 0.01, -0.05),
], dtype=np.float32)


# ---------------------------------------------------------------------------
# 工具：欧拉旋转矩阵（Y-X-Z 顺序）
# ---------------------------------------------------------------------------
def _rot_matrix(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (Rz @ Rx @ Ry).astype(np.float32)


# ---------------------------------------------------------------------------
# 手部相对上身的参数化运动（按类索引确定，类间可区分）
# ---------------------------------------------------------------------------
def _motion_offset(mtype, p, r, freq, phase):
    """返回 (T,3) 手腕相对上身中心的轨迹偏移。p 为归一化进度 (T,)。"""
    th = 2 * np.pi * freq * p + phase
    T = p.shape[0]
    z = np.zeros(T, dtype=np.float32)
    if mtype == 0:   # 画圆（x-y 平面）
        return np.stack([r * np.cos(th), r * np.sin(th), z], axis=-1)
    if mtype == 1:   # 画圆（x-z 平面）
        return np.stack([r * np.cos(th), z, r * np.sin(th)], axis=-1)
    if mtype == 2:   # 水平直线往返
        return np.stack([r * np.sin(th), z, z], axis=-1)
    if mtype == 3:   # 垂直上下摆动
        return np.stack([z, r * np.sin(th), z], axis=-1)
    if mtype == 4:   # 对角 + 深度
        return np.stack([r * np.cos(th) * 0.8, r * np.sin(th) * 0.8,
                         r * 0.3 * np.sin(2 * th)], axis=-1)
    if mtype == 5:   # 8 字（x-y）
        return np.stack([r * np.sin(th), r * np.sin(2 * th), z], axis=-1)
    if mtype == 6:   # 扫动（带包络）
        return np.stack([r * (2 * p - 1) * np.sin(th), r * 0.4 * np.sin(th), z], axis=-1)
    if mtype == 7:   # 弧线
        return np.stack([r * np.sin(th), r * 0.5 * (1 - np.cos(th)),
                         r * 0.4 * np.sin(th)], axis=-1)
    if mtype == 8:   # 快速抖动
        return np.stack([r * 0.6 * np.sin(6 * th), r * 0.6 * np.sin(5 * th + phase),
                         r * 0.3 * np.sin(3 * th)], axis=-1)
    if mtype == 9:   # 呼吸圆（幅度随进度变化）
        return np.stack([r * np.sin(th) * np.cos(p * np.pi), r * 0.5 * np.sin(th), z], axis=-1)
    return np.stack([z, z, z], axis=-1)


def _class_params(c):
    """根据类索引确定该类的运动与手势参数。"""
    mtype = c % 10
    freq = 0.6 + (c % 6) * 0.18
    r = 0.10 + (c % 4) * 0.025
    phase = (c * 0.5) % (2 * np.pi)
    cx = 0.05 * ((c % 5) - 2)
    cy = 0.03 * ((c % 3) - 1)
    cz = 0.02 * ((c % 2) * 2 - 1)
    g = 0.15 + 0.7 * ((c % 5) / 4.0)  # 手势开合度（0.15 开 -> 0.85 接近握拳）
    return dict(mtype=mtype, freq=freq, r=r, phase=phase, cx=cx, cy=cy, cz=cz, g=g)


# ---------------------------------------------------------------------------
# 单样本生成
# ---------------------------------------------------------------------------
def generate_sample(class_idx, T, rng):
    """生成 1 个 (T, 27, 3) float32 归一化骨架序列。"""
    c = int(class_idx)
    pr = _class_params(c)

    # 手部局部形状（开手 <-> 握拳 插值），再施加每样本朝向
    local = (1.0 - pr["g"]) * OPEN_HAND + pr["g"] * FIST_HAND   # (21,3)
    yaw = float(rng.uniform(-0.3, 0.3) + 0.15 * np.sin(c))
    pitch = float(rng.uniform(-0.2, 0.2))
    roll = float(rng.uniform(-0.15, 0.15))
    R = _rot_matrix(yaw, pitch, roll)
    hand = local @ R.T  # (21,3)

    # 手腕轨迹（相对上身中心）
    p = np.linspace(0.0, 1.0, T, dtype=np.float32)
    off = _motion_offset(pr["mtype"], p, pr["r"], pr["freq"], pr["phase"])  # (T,3)
    center = np.array([pr["cx"], 1.25 + pr["cy"], 0.30 + pr["cz"]], dtype=np.float32)
    wrist = center[None, :] + off                                   # (T,3)
    hand_pts = wrist[:, None, :] + hand[None, :, :]                 # (T,21,3)
    hand_pts = hand_pts + rng.normal(0, 0.004, size=hand_pts.shape).astype(np.float32)

    # 上身 6 点（静态；含轻微随类别变化的姿态偏移：头部前倾 + 肘部微屈）
    nose_z = 0.03 * np.sin(c * 0.7)
    base_pose = np.array([
        [0.00, 1.55, nose_z],                       # 0 鼻
        [-0.20, 1.35, 0.00],                        # 1 左肩
        [0.20, 1.35, 0.00],                         # 2 右肩
        [-0.35, 1.15, 0.05 + 0.02 * np.sin(c)],     # 3 左肘
        [0.35, 1.15, 0.05 + 0.02 * np.cos(c)],      # 4 右肘
        [-0.45, 1.00, 0.10],                        # 5 左腕
    ], dtype=np.float32)
    pose = np.broadcast_to(base_pose, (T, 6, 3)).copy()

    seq = np.concatenate([hand_pts, pose], axis=1).astype(np.float32)  # (T,27,3)

    # 与 contract 一致的归一化（鼻原点 / 肩宽尺度 / 肩线水平）
    norm, _, _ = normalize_sequence(seq)
    return norm.astype(np.float32)


# ---------------------------------------------------------------------------
# 数据集生成（写 npz + classes.txt）
# ---------------------------------------------------------------------------
def generate_dataset(root, seed=SEED, splits=None, force=True):
    """生成 train/val/test 三切分，保存到 root 下 *.npz 与 classes.txt。

    Returns: {split: ((N, T, 27, 3), N)}
    """
    splits = splits or SPLITS
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    classes_arr = np.array(CLASSES, dtype=object)
    np.savetxt(root / "classes.txt", classes_arr, fmt="%s")

    info = {}
    for si, (split, count) in enumerate(splits.items()):
        split_seed = seed + si * 1_000_000
        samples, labels = [], []
        for c in range(NUM_CLASSES):
            for s in range(count):
                rng = np.random.default_rng(split_seed + c * 1000 + s)
                T = int(rng.integers(T_MIN, T_MAX + 1))
                samples.append(generate_sample(c, T, rng))
                labels.append(c)

        Tmax = max(s.shape[0] for s in samples)
        padded = []
        for s in samples:
            if s.shape[0] < Tmax:
                pad = np.broadcast_to(s[-1], (Tmax - s.shape[0], 27, 3))
                padded.append(np.concatenate([s, pad], axis=0))
            else:
                padded.append(s)
        skeletons = np.stack(padded, axis=0).astype(np.float32)
        labels = np.array(labels, dtype=np.int64)

        np.savez(
            root / f"{split}.npz",
            skeletons=skeletons,
            labels=labels,
            classes=classes_arr,
        )
        info[split] = (skeletons.shape, int(labels.shape[0]))
    return info


# 兼容别名
def make_synthetic(root, seed=SEED, splits=None, force=True):
    return generate_dataset(root, seed=seed, splits=splits, force=force)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    info = generate_dataset(here / "synthetic", seed=SEED, splits=SPLITS)
    for split, (sh, n) in info.items():
        print(f"[synthetic] {split}: {n} samples, skeletons{sh}")
