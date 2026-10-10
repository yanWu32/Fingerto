"""TASK-005 数据层验收测试（R-data）。

验证：load_dataset 形状 / classes 长度=60 / 合成样本通过 check_skeleton_shape /
csl_loader import 无错 / SyntheticDataset 可迭代取 (T,27,3)+label。
无外部数据（ISW-1000/CSL）时也应全绿。
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# 把仓库根加入 sys.path，支持 `python -m pytest tests/test_data.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import load_dataset, SyntheticDataset, CLASSES  # noqa: E402
from data.csl_loader import convert_csl, mediapipe_available  # noqa: E402  import 无错
from tests.contract import check_skeleton_shape, NUM_POINTS, SKELETON_RANK  # noqa: E402


def test_load_dataset_shape_and_classes():
    skeletons, labels, classes = load_dataset("train")
    assert skeletons.ndim == 4
    assert skeletons.shape[2:] == (NUM_POINTS, SKELETON_RANK)
    assert skeletons.dtype == np.float32
    assert labels.dtype == np.int64
    assert labels.shape[0] == skeletons.shape[0]
    assert len(classes) == 60
    assert classes == CLASSES
    # 各切分都能加载
    for split in ("train", "val", "test"):
        sk, lb, cl = load_dataset(split)
        assert sk.shape[0] == lb.shape[0]
        assert cl == classes


def test_classes_length_is_60():
    sk, lb, cl = load_dataset("train")
    assert len(cl) == 60
    assert len(set(cl)) == 60  # 无重复


def test_synthetic_samples_pass_contract():
    skeletons, labels, classes = load_dataset("train")
    # 抽样校验：覆盖每类至少一个样本
    seen = set()
    for i in range(skeletons.shape[0]):
        lb = int(labels[i])
        if lb in seen:
            continue
        seen.add(lb)
        assert check_skeleton_shape(skeletons[i]), f"样本 i={i} 未通过契约校验"
        assert skeletons[i].shape == (skeletons.shape[1], NUM_POINTS, SKELETON_RANK)
        assert np.all(np.isfinite(skeletons[i]))
        if len(seen) == 60:
            break
    assert len(seen) == 60


def test_csl_loader_importable():
    # 仅验证 import 无错 + 媒体依赖可用（缺失也不应导致 import 失败）
    assert callable(convert_csl)
    # mediapipe 是否可用不应影响 import
    assert mediapipe_available() in (True, False)


def test_synthetic_dataset_iterable():
    ds = SyntheticDataset("train")
    assert len(ds) > 0
    skel, label = ds[0]
    assert skel.shape[1:] == (NUM_POINTS, SKELETON_RANK)
    assert isinstance(int(label), int)
    assert 0 <= int(label) < ds.num_classes
    # 可迭代
    count = 0
    for skel, label in ds:
        assert skel.shape[1:] == (NUM_POINTS, SKELETON_RANK)
        count += 1
        if count >= 5:
            break
    assert count == 5


def test_split_files_exist():
    from data import dataset as _ds
    for split in ("train", "val", "test"):
        root = Path(_ds.DEFAULT_ROOT)
        assert (root / f"{split}.npz").exists()
    assert (Path(_ds.DEFAULT_ROOT) / "classes.txt").exists()
