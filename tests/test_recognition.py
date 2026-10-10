"""识别层测试：三模型 forward 形状 + predict 通过 validate_gloss_output。

运行：pytest tests/test_recognition.py
（mock 数据，不需要真实 ISW-1000）
"""

import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.contract import validate_gloss_output  # noqa: E402

from recognition.models import build_model, STGCN, TransformerGCN, CNNLSTM  # noqa: E402
from recognition.predict import predict  # noqa: E402
from recognition.dataset import generate_mock_batch  # noqa: E402

NUM_CLASSES = 60
NUM_POINTS = 27
T = 40
B = 2

MODELS = ["stgcn", "transformer_gcn", "cnn_lstm"]


@pytest.mark.parametrize("name", MODELS)
def test_forward_shape(name):
    """三模型前向：(B,T,27,3) -> (B, num_classes)。"""
    model = build_model(name, num_classes=NUM_CLASSES)
    model.eval()
    x, _ = generate_mock_batch(batch_size=B, T=T, num_points=NUM_POINTS,
                               num_classes=NUM_CLASSES, rng=np.random.default_rng(0))
    x = torch.from_numpy(x)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, NUM_CLASSES), f"{name} 输出形状 {tuple(out.shape)} != (B, num_classes)"


@pytest.mark.parametrize("name", MODELS)
def test_backward_step(name):
    """冒烟：前向 + 反向 + 一步优化梯度可更新。"""
    torch.manual_seed(0)
    model = build_model(name, num_classes=NUM_CLASSES)
    x, y = generate_mock_batch(batch_size=B, T=T, num_points=NUM_POINTS,
                               num_classes=NUM_CLASSES, rng=np.random.default_rng(1))
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    # 参数应有梯度
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), f"{name} 无有效梯度"


def test_predict_passes_contract():
    """predict 返回 dict 必须通过 validate_gloss_output。"""
    seq = np.random.default_rng(2).standard_normal((T, NUM_POINTS, 3)).astype(np.float32)
    for name in MODELS:
        out = predict(seq, fps=30.0, classes=[f"gloss_{i}" for i in range(NUM_CLASSES)],
                      model_name=name, num_classes=NUM_CLASSES)
        validate_gloss_output(out)  # 不抛异常即通过
        assert isinstance(out["gloss_sequence"][0], str)
        assert 0.0 <= out["confidence"][0] <= 1.0
        assert len(out["timestamps"][0]) == 2 and out["timestamps"][0][0] <= out["timestamps"][0][1]


def test_predict_timestamps_seconds():
    """时间戳单位应为秒（t/fps）。"""
    seq = np.random.default_rng(3).standard_normal((T, NUM_POINTS, 3)).astype(np.float32)
    out = predict(seq, fps=30.0, model_name="stgcn", num_classes=NUM_CLASSES)
    s, e = out["timestamps"][0]
    assert abs(e - (T / 30.0)) < 1e-3, f"时间戳应为 T/fps={T/30:.3f}，收到 {e}"


def test_predict_window_length_mismatch():
    """变长窗口应自适应（截断/补零）并通过契约。"""
    seq = np.random.default_rng(4).standard_normal((17, NUM_POINTS, 3)).astype(np.float32)
    out = predict(seq, fps=25.0, model_name="cnn_lstm", num_classes=NUM_CLASSES)
    validate_gloss_output(out)
