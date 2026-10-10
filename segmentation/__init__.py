"""手势切分层 — 运动能量粗切分 + 滑动窗口细识别。

输入：连续骨架序列
输出：候选手势单元（骨架片段 + 时间戳）

对外导出：``SkeletonSegmenter``、``Segment``。
"""
from __future__ import annotations

from .segmenter import Segment, SkeletonSegmenter

__all__ = ["SkeletonSegmenter", "Segment"]
