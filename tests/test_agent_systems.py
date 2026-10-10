"""消融系统 A/B/C/D 测试（无需真实 API Key，使用 fake_completion）。

验证：
- 四系统都能跑出符合 contract 的 agent JSON
- System A 离线规则翻译（开灯意图识别）
- System B 无状态单轮翻译
- System C 完整管线（含对话状态/指代）
- System D 无权重时退化为 C（契约一致）
"""

import sys
import os
import pytest

# 让 tests 能 import 仓库根模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import load_config
from agent.systems import build_system, SYSTEM_NAMES
from tests.contract import validate_agent_output, validate_gloss_output


@pytest.fixture(scope="module")
def cfg():
    return load_config("agent")


def fake_completion(prompt: str) -> str:
    """确定性假 LLM：根据 prompt 返回简单文本。

    注意：complete(system, user) 只把 user 文本传给 fake；step3 的"意图/JSON"
    关键词在 system 消息(SYSTEM_DECISION)里。因此用 step3 独有的 user 标记
    "允许控制动作" 区分决策请求，其余按翻译处理。
    """
    is_decision = "允许控制动作" in prompt
    if "开" in prompt and "灯" in prompt:
        return ('{"intent": "command", "task": "turn_on_light", "natural_language": "开灯", "need_clarify": false}'
                if is_decision else "请把灯打开")
    return ('{"intent": "chat", "task": "none", "natural_language": "你好", "need_clarify": false}'
            if is_decision else "我想喝水")


GLOSS_ON_LIGHT = {
    "gloss_sequence": ["开", "灯"],
    "confidence": [0.9, 0.9],
    "timestamps": [[0.0, 0.5], [0.5, 1.0]],
}
GLOSS_DRINK = {
    "gloss_sequence": ["我", "想", "喝", "水"],
    "confidence": [0.9, 0.9, 0.9, 0.9],
    "timestamps": [[0.0, 0.4], [0.4, 0.8], [0.8, 1.2], [1.2, 1.6]],
}


def test_system_names_complete():
    assert set(SYSTEM_NAMES) == {"A", "B", "C", "D"}


def test_system_a_rule(cfg):
    sys_a = build_system("A", cfg)
    out = sys_a.run(GLOSS_ON_LIGHT)
    validate_agent_output(out)
    assert out["intent"] == "command"
    assert out["task"] == "turn_on_light"
    assert "开" in out["natural_language"] and "灯" in out["natural_language"]


def test_system_a_off_light(cfg):
    sys_a = build_system("A", cfg)
    out = sys_a.run({"gloss_sequence": ["关", "灯"], "confidence": [0.9, 0.9],
                     "timestamps": [[0.0, 0.5], [0.5, 1.0]]})
    validate_agent_output(out)
    assert out["task"] == "turn_off_light"


def test_system_b_nostate(cfg):
    sys_b = build_system("B", cfg, fake_completion=fake_completion)
    out = sys_b.run(GLOSS_DRINK)
    validate_agent_output(out)
    assert out["intent"] == "chat"  # 无状态不决策
    assert "水" in out["natural_language"]


def test_system_c_full(cfg):
    sys_c = build_system("C", cfg, fake_completion=fake_completion)
    out = sys_c.run(GLOSS_ON_LIGHT)
    validate_agent_output(out)
    # Step3 决策应给出 command
    assert out["intent"] == "command"
    assert out["task"] == "turn_on_light"


def test_system_d_fallback(cfg):
    sys_d = build_system("D", cfg, fake_completion=fake_completion)
    out = sys_d.run(GLOSS_DRINK)
    validate_agent_output(out)
    assert out["intent"] in {"chat", "command"}


def test_all_systems_return_contract(cfg):
    for name in SYSTEM_NAMES:
        s = build_system(name, cfg, fake_completion=fake_completion)
        out = s.run(GLOSS_DRINK)
        assert validate_agent_output(out) is True
        # 输出含五个契约键
        for k in ("natural_language", "intent", "task", "clarification", "low_confidence_words"):
            assert k in out
