"""统一骨架数据集加载 API（供识别层 / 实验层只读调用）。

接口与合成集、CSL 加载器完全一致：
    load_dataset(split="train", root="data/synthetic")
        -> (skeletons (N, T, 27, 3), labels (N,), classes list[str])

并暴露 torch.utils.data.Dataset 子类 SyntheticDataset。
所有样本均经 tests/contract.check_skeleton_shape 校验。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np

from tests.contract import (
    check_skeleton_shape,
    NUM_POINTS,
    SKELETON_RANK,
)

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_OK = True
except Exception:  # torch 缺失时 Dataset 退化为 None，仅 load_dataset 可用
    torch = None
    Dataset = object  # type: ignore
    _TORCH_OK = False


DEFAULT_ROOT = os.path.join("data", "synthetic")
SPLITS = ("train", "val", "test")


def load_dataset(split: str = "train", root: str = DEFAULT_ROOT) -> Tuple[np.ndarray, np.ndarray, list]:
    """加载某切分的合成骨架数据集。

    Args:
        split: "train" | "val" | "test"
        root:  含 {split}.npz 与 classes.txt 的目录

    Returns:
        skeletons: (N, T, 27, 3) float32（T 为该切分最大帧数，短序列尾帧重复 pad）
        labels:    (N,) int64
        classes:   list[str] 长度 60
    """
    if split not in SPLITS:
        raise ValueError(f"split 必须是 {SPLITS} 之一，收到 {split!r}")
    root = Path(root)
    npz_path = root / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"未找到 {npz_path}。请先运行 `python data/make_testset.py` 生成合成数据集。"
        )

    data = np.load(npz_path, allow_pickle=True)
    skeletons = np.asarray(data["skeletons"], dtype=np.float32)
    labels = np.asarray(data["labels"]).astype(np.int64)
    classes = list(data["classes"])

    # 形状与契约校验
    if skeletons.ndim != 4 or skeletons.shape[2:] != (NUM_POINTS, SKELETON_RANK):
        raise ValueError(
            f"skeletons 形状非法：{skeletons.shape}，应为 (N, T, 27, 3)"
        )
    if skeletons.shape[0] != labels.shape[0]:
        raise ValueError("skeletons 与 labels 样本数不一致")
    if len(classes) != len(set(classes)):
        raise ValueError("classes.txt 存在重复类")

    # 逐样本契约校验
    for i in range(skeletons.shape[0]):
        if not check_skeleton_shape(skeletons[i]):
            raise ValueError(f"样本 {i} 不满足 check_skeleton_shape（NaN/Inf 或形状非法）")
    if not np.all(labels >= 0) or not np.all(labels < len(classes)):
        raise ValueError("labels 超出 classes 范围")

    return skeletons, labels, classes


def get_classes(root: str = DEFAULT_ROOT) -> list:
    """从 classes.txt 读取类列表（npz 缺失时也可用）。"""
    root = Path(root)
    txt = root / "classes.txt"
    if txt.exists():
        return [l.strip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    return load_dataset("train", root)[2]


class SyntheticDataset(Dataset):
    """torch Dataset 封装：__getitem__ 返回 (skeleton (T,27,3), label int)。"""

    def __init__(self, split: str = "train", root: str = DEFAULT_ROOT):
        if not _TORCH_OK:
            raise RuntimeError("未安装 torch，无法使用 SyntheticDataset")
        self.skeletons, self.labels, self.classes = load_dataset(split, root)
        self.root = root
        self.split = split

    def __len__(self) -> int:
        return self.skeletons.shape[0]

    def __getitem__(self, idx):
        skeleton = torch.as_tensor(self.skeletons[idx], dtype=torch.float32)
        label = int(self.labels[idx])
        return skeleton, label

    @property
    def num_classes(self) -> int:
        return len(self.classes)
