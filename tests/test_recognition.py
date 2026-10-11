"""识别层自测：三模型 forward 形状 + backward + predict 通过 validate_gloss_output。

本环境（fingerto conda env）未安装 pytest，故本文件写成**可直接执行的独立脚本**：

    python tests/test_recognition.py            # 运行全部断言
    python tests/test_recognition.py -v         # 打印每条用例

若环境里装了 pytest，仍然可以直接 `pytest tests/test_recognition.py`（下面的
test_* 函数会被自动收集）。两种运行方式共用同一套 CASE_LIST，不存在跳过/死测试。
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
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


# ---------------------------------------------------------------------------
# 用例实现（每个函数不抛异常即通过）
# ---------------------------------------------------------------------------
def check_forward_shape_bt_n_c():
    """三模型前向（数据集排布）：(B,T,27,3) -> (B, num_classes)。"""
    for name in MODELS:
        model = build_model(name, num_classes=NUM_CLASSES)
        model.eval()
        x, _ = generate_mock_batch(batch_size=B, T=T, num_points=NUM_POINTS,
                                   num_classes=NUM_CLASSES,
                                   rng=np.random.default_rng(0))
        x = torch.from_numpy(x)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, NUM_CLASSES), \
            f"{name} 输出形状 {tuple(out.shape)} != ({B}, {NUM_CLASSES})"


def check_forward_shape_b_c_t_n():
    """三模型前向（predict.py permute 后的排布）：(B,C=3,T,N=27) -> (B, num_classes)。

    这是本次修复的核心场景：predict.py 会把 (B,T,27,3) 排成 (B,3,T,27)。
    """
    for name in MODELS:
        model = build_model(name, num_classes=NUM_CLASSES)
        model.eval()
        x = torch.randn(1, 3, 90, NUM_POINTS)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, NUM_CLASSES), \
            f"{name} 在 (B,C,T,N) 排布下输出 {tuple(out.shape)} != (1, {NUM_CLASSES})"


def check_backward_step():
    """冒烟：前向 + 反向 + 一步优化，梯度非空。"""
    for name in MODELS:
        torch.manual_seed(0)
        model = build_model(name, num_classes=NUM_CLASSES)
        x, y = generate_mock_batch(batch_size=B, T=T, num_points=NUM_POINTS,
                                   num_classes=NUM_CLASSES,
                                   rng=np.random.default_rng(1))
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), \
            f"{name} 无有效梯度"


def check_forward_no_nan():
    """前向输出必须有限（无 NaN/Inf）——BatchNorm 排错会静默产生 nan。"""
    for name in MODELS:
        model = build_model(name, num_classes=NUM_CLASSES)
        model.eval()
        x = torch.randn(B, T, NUM_POINTS, 3)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all(), f"{name} 前向输出含 NaN/Inf"


def check_predict_passes_contract():
    """predict 返回 dict 必须通过 validate_gloss_output（三模型各一次）。"""
    seq = np.random.default_rng(2).standard_normal((T, NUM_POINTS, 3)).astype(np.float32)
    for name in MODELS:
        out = predict(seq, fps=30.0, classes=[f"gloss_{i}" for i in range(NUM_CLASSES)],
                      model_name=name, num_classes=NUM_CLASSES)
        validate_gloss_output(out)  # 不抛异常即通过
        assert isinstance(out["gloss_sequence"][0], str)
        assert 0.0 <= out["confidence"][0] <= 1.0
        assert len(out["timestamps"][0]) == 2 and out["timestamps"][0][0] <= out["timestamps"][0][1]


def check_predict_timestamps_seconds():
    """时间戳单位应为秒（t/fps）。"""
    seq = np.random.default_rng(3).standard_normal((T, NUM_POINTS, 3)).astype(np.float32)
    out = predict(seq, fps=30.0, model_name="stgcn", num_classes=NUM_CLASSES)
    s, e = out["timestamps"][0]
    assert abs(e - (T / 30.0)) < 1e-3, f"时间戳应为 T/fps={T / 30:.3f}，收到 {e}"


def check_predict_window_length_mismatch():
    """变长窗口应自适应（截断/补零）并通过契约。"""
    seq = np.random.default_rng(4).standard_normal((17, NUM_POINTS, 3)).astype(np.float32)
    out = predict(seq, fps=25.0, model_name="cnn_lstm", num_classes=NUM_CLASSES)
    validate_gloss_output(out)


def check_predict_with_real_classes():
    """用真实 60 类 class 名（中文 gloss）时也要通过契约。"""
    from data.dataset import get_classes
    classes = get_classes(os.path.join(_ROOT, "data", "synthetic"))
    assert len(classes) == NUM_CLASSES, f"合成集类数 {len(classes)} != {NUM_CLASSES}"
    seq = np.random.default_rng(5).standard_normal((60, NUM_POINTS, 3)).astype(np.float32)
    out = predict(seq, fps=30.0, classes=classes, model_name="stgcn",
                  num_classes=NUM_CLASSES)
    validate_gloss_output(out)
    assert out["gloss_sequence"][0] in classes


def check_stgcn_time_preserved():
    """ST-GCN 各 stage 输出时间维必须恒等于输入 T（残差对齐的前提）。"""
    model = build_model("stgcn", num_classes=NUM_CLASSES)
    model.eval()
    x = torch.randn(1, T, NUM_POINTS, 3).permute(0, 3, 1, 2).contiguous()  # (1,3,T,N)
    with torch.no_grad():
        h = x
        for blk in model.stage:
            h = blk(h)
            assert h.shape[2] == T, \
                f"STGCNBlock 时间维被改变：{h.shape[2]} != {T}"


def check_transformer_gcn_einsum_2d_adjacency():
    """TransformerGCN 的 GraphConv 邻接必须是 2 维 (N,N)，einsum 方程为 'nv,bcv->bcn'。"""
    from recognition.models.transformer_gcn import GraphConv
    gc = GraphConv(8, 8, NUM_POINTS)
    assert gc.Ahat.dim() == 2 and gc.Ahat.shape == (NUM_POINTS, NUM_POINTS), \
        f"Ahat 应为 ({NUM_POINTS},{NUM_POINTS}) 2 维，收到 {tuple(gc.Ahat.shape)}"
    x = torch.randn(2, 8, NUM_POINTS)
    out = gc(x)
    assert out.shape == (2, 8, NUM_POINTS), f"GraphConv 输出 {tuple(out.shape)} 异常"


def check_cnn_lstm_input_size_matches():
    """CNN+LSTM：送入 LSTM 的最后一维必须等于 lstm.input_size（不同 channel 配置都成立）。"""
    for channels in [(32, 64), (16, 32, 48), (64,)]:
        model = build_model("cnn_lstm", num_classes=NUM_CLASSES, cnn_channels=channels)
        model.eval()
        assert model.lstm.input_size == channels[-1], \
            f"cnn_channels={channels} 时 input_size={model.lstm.input_size} != {channels[-1]}"
        x = torch.randn(2, T, NUM_POINTS, 3)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES), f"cnn_lstm({channels}) 输出 {tuple(out.shape)} 异常"


def check_models_registry_complete():
    """注册表三模型齐全且构造出的实例类型正确。"""
    for name, cls in [("stgcn", STGCN), ("transformer_gcn", TransformerGCN),
                      ("cnn_lstm", CNNLSTM)]:
        m = build_model(name, num_classes=NUM_CLASSES)
        assert isinstance(m, cls), f"{name} 构造出 {type(m)}，期望 {cls}"


# ---------------------------------------------------------------------------
# pytest 兼容包装（装了 pytest 时同样可被收集，与独立运行共用同一实现）
# ---------------------------------------------------------------------------
def test_forward_shape():
    check_forward_shape_bt_n_c()


def test_forward_shape_b_c_t_n():
    check_forward_shape_b_c_t_n()


def test_backward_step():
    check_backward_step()


def test_forward_no_nan():
    check_forward_no_nan()


def test_predict_passes_contract():
    check_predict_passes_contract()


def test_predict_timestamps_seconds():
    check_predict_timestamps_seconds()


def test_predict_window_length_mismatch():
    check_predict_window_length_mismatch()


def test_predict_with_real_classes():
    check_predict_with_real_classes()


def test_stgcn_time_preserved():
    check_stgcn_time_preserved()


def test_transformer_gcn_einsum_2d_adjacency():
    check_transformer_gcn_einsum_2d_adjacency()


def test_cnn_lstm_input_size_matches():
    check_cnn_lstm_input_size_matches()


def test_models_registry_complete():
    check_models_registry_complete()


# ---------------------------------------------------------------------------
# 独立运行入口（无 pytest）
# ---------------------------------------------------------------------------
CASE_LIST = [
    ("forward 形状 (B,T,N,C) -> (B,60)", check_forward_shape_bt_n_c),
    ("forward 形状 (B,C,T,N) -> (B,60)", check_forward_shape_b_c_t_n),
    ("前向+反向+优化步梯度非空", check_backward_step),
    ("前向输出无 NaN/Inf", check_forward_no_nan),
    ("predict 通过 gloss 契约（三模型）", check_predict_passes_contract),
    ("predict 时间戳单位为秒", check_predict_timestamps_seconds),
    ("变长窗口自适应并通过契约", check_predict_window_length_mismatch),
    ("真实 60 类中文 gloss 推理通过契约", check_predict_with_real_classes),
    ("ST-GCN 各 block 保持时间维 T", check_stgcn_time_preserved),
    ("TransformerGCN 邻接为 2 维且 einsum 正确", check_transformer_gcn_einsum_2d_adjacency),
    ("CNN+LSTM input_size 与实际特征维一致", check_cnn_lstm_input_size_matches),
    ("模型注册表完整", check_models_registry_complete),
]


def main(argv=None):
    verbose = "-v" in (argv or sys.argv[1:])
    torch.manual_seed(42)
    np.random.seed(42)
    passed, failed = 0, []
    print("=" * 72)
    print(f"识别层自测（独立脚本，无 pytest） torch={torch.__version__}")
    print("=" * 72)
    for i, (desc, fn) in enumerate(CASE_LIST, 1):
        try:
            fn()
            passed += 1
            print(f"[PASS] {i:2d}. {desc}")
            if verbose:
                print(f"       -> {fn.__name__} OK")
        except Exception:
            failed.append((i, desc, fn.__name__))
            print(f"[FAIL] {i:2d}. {desc}")
            traceback.print_exc()
    print("=" * 72)
    total = len(CASE_LIST)
    print(f"结果：{passed}/{total} 通过，{len(failed)} 失败")
    if failed:
        for i, desc, name in failed:
            print(f"  - 失败 #{i}: {desc} ({name})")
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
