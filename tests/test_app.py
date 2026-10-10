"""应用层串联测试（无 torch / 无 gradio 环境下必须全绿）。

策略：
- 用注入的假识别器 + fake_completion 走通 "骨架 -> 切分 -> 识别 -> 智能体" 全链路，
  避免依赖 torch / 真实模型权重 / 真实 LLM。
- 依赖 torch 的真实识别器用例用 pytest.importorskip("torch") 优雅跳过。
- import app / import app.app 必须无异常（gradio 缺失时 app.app 不应崩）。
"""
import os
import sys

import numpy as np
import pytest

# 让 tests 能 import 仓库根模块（app / segmentation / agent / configs / data）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import load_config
from agent import run_agent
from tests.contract import validate_agent_output, validate_gloss_output
from app.pipeline_demo import run_demo


# ---------------------------------------------------------------------------
# 假识别器 / 假 LLM
# ---------------------------------------------------------------------------
class FakePredictor:
    """可注入的假识别器：每个窗口返回一个 gloss，按给定词表轮转。"""

    def __init__(self, words):
        self.words = list(words)
        self.i = 0

    def __call__(self, window, fps=30.0, **_kw):
        w = self.words[self.i % len(self.words)]
        self.i += 1
        return {
            "gloss_sequence": [w],
            "confidence": [0.9],
            "timestamps": [[0.0, 0.5]],
        }


def fake_completion(prompt: str) -> str:
    """确定性假 LLM（仿 test_agent_systems 写法）。"""
    is_decision = "允许控制动作" in prompt
    if "开" in prompt and "灯" in prompt:
        resp = ('{"intent": "command", "task": "turn_on_light", '
                '"natural_language": "开灯", "need_clarify": false}'
                if is_decision else "请把灯打开")
    elif "关" in prompt and "灯" in prompt:
        resp = ('{"intent": "command", "task": "turn_off_light", '
                '"natural_language": "关灯", "need_clarify": false}'
                if is_decision else "请关灯")
    else:
        resp = ('{"intent": "chat", "task": "none", '
                '"natural_language": "你好", "need_clarify": false}'
                if is_decision else "我想喝水")
    return resp


GLOSS_ON_LIGHT = {
    "gloss_sequence": ["开", "灯"],
    "confidence": [0.9, 0.9],
    "timestamps": [[0.0, 0.5], [0.5, 1.0]],
}


def _make_moving_skeleton(T=120, num_points=27, motion_start=35, motion_end=84,
                          seed=0):
    """构造一段含清晰手势的 (T,27,3) 骨架。

    设计：绝大部分帧是低噪静态背景（能量 < 切分阈值），仅 ``motion_start``
    到 ``motion_end`` 这段主手手腕(idx0)做大幅匀速运动（能量 > 阈值且占比 <50%），
    使得切分器稳定切出 1 个活跃段，并在窗口(stride)下产出多个识别窗口。
    """
    rng = np.random.default_rng(seed)
    # 低噪静态背景：两参考关节(idx0, idx26) 帧间差 ~0.003，和 < 0.02 阈值
    seq = rng.normal(0, 0.002, size=(T, num_points, 3)).astype(np.float32)
    # 手势段：idx0 在 x 方向从 0 匀速到 1.5（每帧差 ~0.03，明显超阈值）
    n_move = motion_end - motion_start
    ramp = np.linspace(0.0, 1.5, n_move)
    seq[motion_start:motion_end, 0, 0] += ramp
    seq[motion_start:motion_end, 0, 1] += 0.4 * np.linspace(0.0, 1.0, n_move)
    return seq


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
def test_imports_no_crash():
    """import app / app.pipeline_demo / app.app 必须无异常（gradio 缺失也不崩）。"""
    import app                          # noqa: F401
    import app.pipeline_demo            # noqa: F401
    import app.app                      # noqa: F401
    # gradio 缺失时 app.app.gradio_available() 应为 False（不抛）
    assert app.app.gradio_available() in (True, False)


def test_run_demo_with_gloss_json():
    """直接给 gloss JSON，跳过切分/识别，直达智能体，校验通过。"""
    cfg = load_config("agent")
    out = run_demo(gloss_json=GLOSS_ON_LIGHT, cfg=cfg, fake_completion=fake_completion)
    assert out["ok"] is True
    assert validate_gloss_output(out["gloss_json"]) is True
    assert validate_agent_output(out["agent_json"]) is True
    # "开"+"灯" 在假 LLM 下应决策为开灯
    assert out["agent_json"]["intent"] == "command"
    assert out["agent_json"]["task"] == "turn_on_light"


def test_run_demo_skeleton_to_agent():
    """骨架 -> 切分 -> 假识别器 -> 智能体，全链路跑通且契约合法。"""
    cfg = load_config("agent")
    skeleton = _make_moving_skeleton()
    predictor = FakePredictor(["开", "灯"])
    out = run_demo(skeleton=skeleton, predictor=predictor, fps=30.0,
                   cfg=cfg, fake_completion=fake_completion)
    assert out["ok"] is True
    assert out["num_segments"] >= 1, "运动骨架应被切出至少一个活跃段"
    assert out["num_windows"] >= 1
    assert validate_gloss_output(out["gloss_json"]) is True
    assert validate_agent_output(out["agent_json"]) is True
    # 聚合后 gloss 长度 == 识别窗口数
    assert len(out["gloss_json"]["gloss_sequence"]) == out["num_windows"]


def test_run_demo_from_data_layer():
    """从 data 层取一段真实合成骨架，走全链路（data 层无 torch 依赖）。"""
    from data.synthetic import generate_sample
    cfg = load_config("agent")
    skeleton = generate_sample(0, 50, np.random.default_rng(1))  # (T,27,3)
    assert skeleton.shape == (50, 27, 3)
    predictor = FakePredictor(["开", "灯"])
    out = run_demo(skeleton=skeleton, predictor=predictor, fps=30.0,
                   cfg=cfg, fake_completion=fake_completion)
    assert out["ok"] is True
    assert validate_gloss_output(out["gloss_json"]) is True
    assert validate_agent_output(out["agent_json"]) is True


def test_run_demo_timestamps_non_decreasing():
    """聚合后的时间戳应随窗口递增（start<=end，且整体非递减）。"""
    cfg = load_config("agent")
    skeleton = _make_moving_skeleton()  # 默认 T=120，可切出多窗口
    predictor = FakePredictor(["我", "想", "喝", "水"])
    out = run_demo(skeleton=skeleton, predictor=predictor, fps=30.0,
                   cfg=cfg, fake_completion=fake_completion)
    ts = out["gloss_json"]["timestamps"]
    assert len(ts) >= 2
    for (s, e) in ts:
        assert s <= e
    for k in range(1, len(ts)):
        assert ts[k][0] >= ts[k - 1][0]


def test_default_predictor_requires_torch():
    """真实识别器（默认路径）依赖 torch；无 torch 时该用例优雅跳过。"""
    torch = pytest.importorskip("torch")  # 无 torch -> 直接跳过
    from app.pipeline_demo import _default_recognition_predictor

    skeleton = _make_moving_skeleton(T=30)
    out = _default_recognition_predictor(skeleton[:30], fps=30.0)
    assert validate_gloss_output(out) is True
    assert len(out["gloss_sequence"]) == 1


def test_run_demo_requires_input():
    """既不给 skeleton 也不给 gloss_json 应抛 ValueError。"""
    with pytest.raises(ValueError):
        run_demo()
