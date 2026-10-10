"""切分模块单测：运动能量粗切分 + 滑动窗口细识别。

运行：在 Fingerto/ 下 ``pytest tests/test_segmentation.py``（需 fingerto 环境）。

构造合成序列：仅中段（约帧 40-90）有运动，两侧静止，断言：
  - 能检出恰好一段；
  - 段的帧/时间边界正确；
  - sub_sequence 形状合法且有限；
  - 滑动窗口产出 (w,27,3) 窗口。
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.contract import check_skeleton_shape

from segmentation import Segment, SkeletonSegmenter


TOTAL = 150
FPS = 30.0


def _make_synthetic():
    """合成 (T,27,3)：中段有正弦运动，两端静止。"""
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.01, size=(27, 3)).astype(np.float32)
    seq = np.tile(base, (TOTAL, 1, 1)).copy()          # (T,27,3) 全静止

    # 中段运动：让手部腕(0) 与 上半身左腕(26) 沿 x 做持续线性位移，
    # 保证每帧都有非零速度（不会因正弦过零被拆成多段）。
    a, b = 40, 95
    step = 0.03                                    # 每帧位移 > 速度阈值
    for k, t in enumerate(range(a, b)):
        disp = (k + 1) * step
        move = np.zeros((27, 3), dtype=np.float32)
        move[0, 0] = disp                          # 手部腕 x 正向
        move[26, 0] = -disp                        # 上半身左腕 x 反向
        seq[t] = base + move
    return seq


def test_segment_detects_single_middle_segment():
    seq = _make_synthetic()
    assert check_skeleton_shape(seq)

    seg = SkeletonSegmenter()
    segments = seg.segment(seq, fps=FPS)

    # 仅中段有运动 -> 期望检出 1 段（允许邻域 1-2 帧抖动）
    assert len(segments) == 1, f"期望检出 1 段，实际 {len(segments)} 段"
    s = segments[0]
    assert s.start_frame <= 42, s.start_frame
    assert s.end_frame >= 93, s.end_frame
    assert s.start_time == pytest.approx(s.start_frame / FPS, abs=1e-6)
    assert s.end_time == pytest.approx(s.end_frame / FPS, abs=1e-6)

    # sub_sequence 形状与有限性
    assert s.sub_sequence.shape[0] == s.end_frame - s.start_frame + 1
    assert check_skeleton_shape(s.sub_sequence)


def test_segment_subsequence_shape_and_finite():
    seq = _make_synthetic()
    seg = SkeletonSegmenter()
    segments = seg.segment(seq, fps=FPS)
    for s in segments:
        assert s.sub_sequence.ndim == 3
        assert s.sub_sequence.shape[1:] == (27, 3)
        assert np.all(np.isfinite(s.sub_sequence))


def test_sliding_windows_produce_windows():
    seq = _make_synthetic()
    seg = SkeletonSegmenter()
    segments = seg.segment(seq, fps=FPS)
    assert segments

    s = segments[0]
    wins = list(seg.sliding_windows(s))
    assert len(wins) >= 1
    for w in wins:
        assert w.shape[1:] == (27, 3)
        assert check_skeleton_shape(w)
        assert w.shape[0] <= seg.window_size


def test_short_segment_discarded_and_static_empty():
    # 全静止序列不应检出任何段
    rng = np.random.default_rng(1)
    seq_static = rng.normal(0, 0.01, size=(120, 27, 3)).astype(np.float32)
    seg = SkeletonSegmenter()
    assert seg.segment(seq_static, fps=FPS) == []


def test_exported_symbols():
    assert SkeletonSegmenter is not None
    assert Segment is not None


def test_input_contract_rejects_bad_shape():
    seg = SkeletonSegmenter()
    bad = np.random.randn(10, 27, 2)        # 秩错误
    with pytest.raises(ValueError):
        seg.segment(bad, fps=FPS)
