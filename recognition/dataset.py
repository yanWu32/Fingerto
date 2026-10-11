"""识别层数据集与 dataloader。

支持两种数据来源：
  1) 真实数据：.npz 文件，含 {'skeleton': (T,27,3) 或 (T,N,C), 'label': str}
     （与 data.load_dataset 兼容；真实数据待 ISW-1000 到位）
  2) mock 数据：随机生成 (T,27,3) + 随机类别，无数据时冒烟训练/测试

骨架契约形状 (T, 27, 3)：
   主手 21 点 (HAND_OFFSET=0) + 上身 6 点 (POSE_OFFSET=21)。
"""

from __future__ import annotations

import os
import random
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

NUM_POINTS = 27
NUM_HAND_POINTS = 21
NUM_POSE_POINTS = 6
HAND_OFFSET = 0
POSE_OFFSET = 21


def _load_npz(path: str):
    """读取单个 .npz，返回 (skeleton (T,27,3) np.float32, label str)。"""
    data = np.load(path, allow_pickle=True)
    sk = np.asarray(data["skeleton"], dtype=np.float32)
    label = str(data["label"]) if "label" in data.files else "unknown"
    # 兼容 (T, N, C) -> 强制 (T, 27, 3)
    if sk.ndim == 3:
        if sk.shape[1] != NUM_POINTS or sk.shape[2] != 3:
            # 尝试按 (T, C, N) 转置
            if sk.shape[1] == 3 and sk.shape[2] == NUM_POINTS:
                sk = sk.transpose(0, 2, 1)
    return sk, label


class GlossDataset(Dataset):
    """读取 .npz 目录或 mock 生成。

    Args:
        root: .npz 目录（None 时用 mock）
        classes: 类别名列表；None 时从 labels 推断
        max_frames: 序列统一长度（截断/补零）
        pad_mode: 'last' | 'zero'
        mock_size: root=None 时生成的样本数
        mock_classes: mock 模式类别数/名称
        rng_seed: 随机种子
    """

    def __init__(self, root: str | None = None, classes: list[str] | None = None,
                 max_frames: int = 120, pad_mode: str = "last",
                 mock_size: int = 64, mock_classes: int | list[str] = 60,
                 rng_seed: int = 42):
        self.max_frames = max_frames
        self.pad_mode = pad_mode
        self.samples = []          # list[(skeleton, label)]
        self.classes = list(classes) if classes is not None else None

        if root is not None and os.path.isdir(root):
            files = sorted(glob(os.path.join(root, "*.npz")))
            if not files:
                raise FileNotFoundError(f"目录 {root} 下未找到 .npz 文件")
            for fp in files:
                sk, label = _load_npz(fp)
                self.samples.append((sk, label))
            if self.classes is None:
                self.classes = sorted({s[1] for s in self.samples})
        else:
            # mock 模式
            rng = np.random.default_rng(rng_seed)
            if isinstance(mock_classes, int):
                self.classes = [f"gloss_{i}" for i in range(mock_classes)]
            else:
                self.classes = list(mock_classes)
            for _ in range(mock_size):
                T = int(rng.integers(30, max_frames + 1))
                sk = rng.standard_normal((T, NUM_POINTS, 3)).astype(np.float32) * 0.3
                # 让不同类别略有不同的骨架分布，便于模型学到区分（仅 mock）
                cid = int(rng.integers(0, len(self.classes)))
                sk[..., :] += cid * 0.05
                self.samples.append((sk, self.classes[cid]))

        self.label2idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def _normalize_length(self, sk: np.ndarray) -> np.ndarray:
        T = sk.shape[0]
        if T >= self.max_frames:
            return sk[: self.max_frames]
        if self.pad_mode == "zero":
            pad = np.zeros((self.max_frames - T, *sk.shape[1:]), dtype=sk.dtype)
        else:  # last
            last = np.repeat(sk[-1:], self.max_frames - T, axis=0)
            pad = last
        return np.concatenate([sk, pad], axis=0)

    def __getitem__(self, idx):
        sk, label = self.samples[idx]
        sk = self._normalize_length(sk)
        return torch.from_numpy(sk.copy()), self.label2idx[label]


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs, 0)          # (B, T, 27, 3)
    ys = torch.as_tensor(ys, dtype=torch.long)
    return xs, ys


def build_dataloader(root: str | None = None, classes: list[str] | None = None,
                     batch_size: int = 8, max_frames: int = 120,
                     pad_mode: str = "last", mock_size: int = 64,
                     mock_classes: int | list[str] = 60, shuffle: bool = True,
                     num_workers: int = 0, rng_seed: int = 42):
    """构造 DataLoader（真实目录或 mock）。"""
    ds = GlossDataset(root=root, classes=classes, max_frames=max_frames,
                      pad_mode=pad_mode, mock_size=mock_size,
                      mock_classes=mock_classes, rng_seed=rng_seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, collate_fn=collate_fn)
    return loader, ds.classes


# ---------------- WLASL 真实数据集加载器（多样本 npz，骨架为 27 点）----------------
def _load_wlasl_split(npz_dir: str, split: str, max_frames: int, pad_mode: str):
    """读取 data/wlasl*/npz/{split}.npz（build_dataset 产出的多样本格式）。

    该格式字段：skeletons (N,T,27,3)、labels (N,)、classes (list[str])。
    这与 GlossDataset 的「单文件 skeleton/label」格式不同，故独立加载。
    """
    import numpy as _np
    path = os.path.join(npz_dir, "npz", f"{split}.npz")
    if not os.path.isfile(path):
        return None, None, None
    data = _np.load(path, allow_pickle=True)
    sk = _np.asarray(data["skeletons"], dtype=_np.float32)
    lb = _np.asarray(data["labels"]).astype(_np.int64)
    cls = [str(c) for c in data["classes"]] if "classes" in data else None
    return sk, lb, cls


class WlaslDataset(Dataset):
    """读取 WLASL 多样本 npz（真实英文 ASL 对照数据）。

    augment: 训练时做轻量骨骼增强（随机关节抖动 + 时间起止偏移 + 随机丢弃单帧），
    缓解真实数据样本稀疏（~5 样本/类）导致的过拟合。验证/测试集关闭。
    """

    def __init__(self, npz_dir: str, split: str = "train",
                 max_frames: int = 90, pad_mode: str = "last",
                 augment: bool = False):
        self.max_frames = max_frames
        self.pad_mode = pad_mode
        self.augment = augment and (split == "train")
        sk, lb, cls = _load_wlasl_split(npz_dir, split, max_frames, pad_mode)
        if sk is None:
            self.samples = []
            self.classes = []
            return
        self.samples = [(sk[i], int(lb[i])) for i in range(sk.shape[0])]
        self.classes = cls if cls is not None else [str(i) for i in range(int(lb.max()) + 1)]

    def __len__(self):
        return len(self.samples)

    def _normalize_length(self, sk: np.ndarray) -> np.ndarray:
        T = sk.shape[0]
        if T >= self.max_frames:
            return sk[: self.max_frames]
        if self.pad_mode == "zero":
            pad = np.zeros((self.max_frames - T, *sk.shape[1:]), dtype=sk.dtype)
        else:
            pad = np.repeat(sk[-1:], self.max_frames - T, axis=0)
        return np.concatenate([sk, pad], axis=0)

    @staticmethod
    def _augment(sk: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # 关节坐标抖动（σ=0.01，相对肩宽归一化尺度下很小）
        sk = sk + rng.normal(0.0, 0.01, size=sk.shape).astype(sk.dtype)
        # 随机时间起止偏移：把序列循环平移最多 ±8 帧
        shift = int(rng.integers(-8, 9))
        if shift != 0:
            sk = np.roll(sk, shift, axis=0)
        return sk

    def __getitem__(self, idx):
        sk, label = self.samples[idx]
        if self.augment:
            sk = self._augment(sk, np.random.default_rng((idx + 1) * 7919))
        sk = self._normalize_length(sk)
        return torch.from_numpy(sk.copy()), label


def build_wlasl_dataloader(npz_dir: str, batch_size: int = 8, max_frames: int = 90,
                           pad_mode: str = "last", num_workers: int = 0,
                           shuffle: bool = True):
    """构造 WLASL 真实数据的 train/val/test DataLoader 字典。

    Args:
        npz_dir: 含 npz/{train,val,test}.npz 与 glosses.txt 的目录
    Returns:
        (loaders dict, classes list)
    """
    loaders: dict = {}
    classes = None
    for split in ("train", "val", "test"):
        ds = WlaslDataset(npz_dir, split=split, max_frames=max_frames,
                          pad_mode=pad_mode, augment=(split == "train"))
        if split == "train":
            classes = ds.classes
        if len(ds) == 0:
            continue
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train" and shuffle),
            num_workers=num_workers, collate_fn=collate_fn,
        )
    if classes is None:
        classes = []
    return loaders, classes


# ---------------- 纯 numpy mock 生成器（供无 torch 依赖场景或快速造数据） ----------------
def generate_mock_window(num_points: int = NUM_POINTS, T: int = 60, rng=None):
    """生成单个随机 (T,27,3) 窗口（无标签）。"""
    rng = rng or np.random
    return rng.standard_normal((T, num_points, 3)).astype(np.float32) * 0.3


def generate_mock_batch(batch_size: int = 2, T: int = 60,
                        num_points: int = NUM_POINTS, num_classes: int = 60, rng=None):
    """生成 (B,T,27,3) 与对应的随机类别索引。"""
    rng = rng or np.random
    x = rng.standard_normal((batch_size, T, num_points, 3)).astype(np.float32)
    y = rng.integers(0, num_classes, size=batch_size).astype(np.int64)
    return x, y
