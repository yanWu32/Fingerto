"""感知层契约测试：验证 perception.skeleton 产出 (T, 27, 3) 且有限。

运行：fingerto 环境下 `pytest tests/test_perception_contract.py`。
仅依赖 numpy（不触发 mediapipe/torch），可独立快速跑。
"""

import numpy as np

from perception.skeleton import NUM_POINTS, normalize_sequence, smooth_sequence
from tests.contract import check_skeleton_shape


def test_normalize_shape_and_finite():
    # 合成：T=8, 27点，鼻为原点附近，肩宽≈1
    T = 8
    seq = np.random.randn(T, NUM_POINTS, 3).astype(float)
    out, _, _ = normalize_sequence(seq)
    assert out.shape == (T, NUM_POINTS, 3)
    assert check_skeleton_shape(out) is True


def test_normalize_nose_origin():
    # 鼻 = POSE 索引 0，对应矩阵第 POSE_OFFSET+0 行；normalize 返回 (norm, vis, kept)
    from perception.skeleton import POSE_OFFSET
    seq = np.random.randn(4, NUM_POINTS, 3).astype(float)
    out, _, _ = normalize_sequence(seq)
    nose = out[:, POSE_OFFSET + 0, :]
    assert np.allclose(nose, 0.0, atol=1e-6)


def test_smooth_shape_preserved():
    seq = np.random.randn(20, NUM_POINTS, 3).astype(float)
    out = smooth_sequence(seq, window=3)
    assert out.shape == seq.shape
