"""数据层公共导出。

仅暴露数据加载 / 生成相关公共符号，供识别层、实验层只读调用。
不在此 import 任何会拉起重依赖（mediapipe/opencv）的模块；
csl_loader / synthetic 的 MediaPipe 依赖在内部按需 import，保证本包可轻量 import。
"""
from __future__ import annotations

from data.dataset import (
    load_dataset,
    get_classes,
    SyntheticDataset,
)
from data.synthetic import (
    generate_dataset,
    make_synthetic,
    CLASSES,
    NUM_CLASSES,
    SEED,
)

__all__ = [
    "load_dataset",
    "get_classes",
    "SyntheticDataset",
    "generate_dataset",
    "make_synthetic",
    "CLASSES",
    "NUM_CLASSES",
    "SEED",
]
