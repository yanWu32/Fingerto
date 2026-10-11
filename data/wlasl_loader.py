"""WLASL 数据集加载器（与 load_dataset 接口对齐）。

load_wlasl(split) -> (skeletons, labels, glosses)
    - skeletons: (N, T, 27, 3) float32
    - labels:    (N,) int64
    - glosses:   list[str] 英文 ASL gloss（独立保存，不混入中文 classes.txt）

⚠️ WLASL 是英文美式手语（ASL）视频数据集，仅作「方法跨语种泛化对照基线」，
   不是中文主实验。每个样本均经 tests.contract.check_skeleton_shape 校验。
数据由 data/wlasl_pose_extract.py 生成（需用户侧先按 wlasl_download_guide.md
下载视频），npz 不入库（*.npz 已被 .gitignore），仅代码 + gloss 文本入库。

无数据时 import 不报错；调用 load_wlasl 时给出清晰提示（与 csl_loader 一致）。
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

DEFAULT_ROOT = os.path.join("data", "wlasl")
NPZ_SUBDIR = "npz"
SPLITS = ("train", "val", "test")


def load_wlasl(split: str = "train", root: str = DEFAULT_ROOT) -> Tuple[np.ndarray, np.ndarray, list]:
    """加载某切分的 WLASL 27 点骨架数据集。

    Args:
        split: "train" | "val" | "test"
        root:  含 npz/{split}.npz 与 glosses.txt 的目录

    Returns:
        skeletons: (N, T, 27, 3) float32
        labels:    (N,) int64
        glosses:   list[str] 英文 ASL gloss（= npz 内 classes 字段）

    Raises:
        ValueError: split 非法 / 形状非法 / 契约校验失败
        FileNotFoundError: 对应 npz 尚未生成
    """
    if split not in SPLITS:
        raise ValueError(f"split 必须是 {SPLITS} 之一，收到 {split!r}")
    root = Path(root)
    npz_path = root / NPZ_SUBDIR / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"未找到 {npz_path}。请先在个人机器按 data/wlasl_download_guide.md 下载"
            " WLASL 视频，再运行 `python data/wlasl_pose_extract.py` 生成骨架 npz。"
        )

    data = np.load(npz_path, allow_pickle=True)
    skeletons = np.asarray(data["skeletons"], dtype=np.float32)
    labels = np.asarray(data["labels"]).astype(np.int64)
    glosses = list(data["classes"]) if "classes" in data else []

    # 形状与契约校验
    if skeletons.ndim != 4 or skeletons.shape[2:] != (NUM_POINTS, SKELETON_RANK):
        raise ValueError(
            f"skeletons 形状非法：{skeletons.shape}，应为 (N, T, 27, 3)"
        )
    if skeletons.shape[0] != labels.shape[0]:
        raise ValueError("skeletons 与 labels 样本数不一致")
    if len(glosses) != len(set(glosses)):
        raise ValueError("glosses.txt 存在重复类")

    # 逐样本契约校验
    for i in range(skeletons.shape[0]):
        if not check_skeleton_shape(skeletons[i]):
            raise ValueError(f"样本 {i} 不满足 check_skeleton_shape（NaN/Inf 或形状非法）")
    if not np.all(labels >= 0) or (len(glosses) > 0 and not np.all(labels < len(glosses))):
        raise ValueError("labels 超出 glosses 范围")

    return skeletons, labels, glosses


def get_glosses(root: str = DEFAULT_ROOT) -> list:
    """读取独立保存的英文 gloss 列表（npz 缺失时也可用）。"""
    root = Path(root)
    txt = root / "glosses.txt"
    if txt.exists():
        return [l.strip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    # npz 内 classes 兜底
    for split in SPLITS:
        npz = root / NPZ_SUBDIR / f"{split}.npz"
        if npz.exists():
            data = np.load(npz, allow_pickle=True)
            return list(data["classes"])
    return []


if __name__ == "__main__":
    import sys
    sp = sys.argv[1] if len(sys.argv) > 1 else "train"
    try:
        sk, lb, gl = load_wlasl(sp)
        print(f"[wlasl_loader] split={sp} N={sk.shape[0]} T={sk.shape[1]} "
              f"类数={len(gl)} dtype={sk.dtype}")
    except FileNotFoundError as e:
        print(f"[wlasl_loader] 数据未就绪：{e}")
