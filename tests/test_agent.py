"""智能体层契约测试：用 fake LLM（无需 API Key）跑通三步管线 + 契约校验。

运行：fingerto 环境下 `pytest tests/test_agent.py`。
覆盖了：正常翻译、低置信澄清路由、消融（关闭 Step1/Step3）、缓存命中。
"""

import sys
import os

import pytest

from configs import load_config
from tests.contract import validate_agent_output, validate_gloss_output

# 确保在 Fingerto 根下能 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cfg():
    return load_config("agent")


def _fake_factory(reply_map=None):
    """返回一个 fake_completion：根据 prompt 关键词返回对应文本。"""
    def fake(prompt: str) -> str:
        if reply_map:
            for k, v in reply_map.items():
                if k in prompt:
                    return v
        # 默认：当作 chat，原样 echo
        return "好的。"
    return fake


def test_pipeline_normal():
    cfg = _cfg()
    fake = _fake_factory({
        "手语 gloss": "我想 开灯",   # step2 翻译
        "用户本轮表达": '{"intent": "command", "task": "turn_on_light", '
                        '"natural_language": "好的，已为您开灯", "need_clarify": false}',
    })
    from agent import build_agent, DialogueState
    agent = build_agent(cfg, fake_completion=fake)
    state = DialogueState(cfg)
    gloss = {"gloss_sequence": ["我想", "开灯"], "confidence": [0.9, 0.92],
             "timestamps": [[0.0, 0.5], [0.6, 1.0]]}
    out = agent.run(gloss, state)
    validate_agent_output(out)
    assert out["intent"] == "command"
    assert out["task"] == "turn_on_light"
    assert out["clarification"] is False
    assert out["low_confidence_words"] == []


def test_pipeline_low_confidence_clarify():
    cfg = _cfg()
    fake = _fake_factory({
        "手语 gloss": "我 想 <unk>",   # step2
    })
    from agent import build_agent, DialogueState
    agent = build_agent(cfg, fake_completion=fake)
    state = DialogueState(cfg)
    gloss = {"gloss_sequence": ["我", "想", "开灯"], "confidence": [0.9, 0.9, 0.3],
             "timestamps": [[0.0, 0.5], [0.6, 1.0], [1.1, 1.5]]}
    out = agent.run(gloss, state)
    validate_agent_output(out)
    # 第三个词置信度 0.3 < clarify_threshold(0.4) -> 触发澄清
    assert out["clarification"] is True
    assert "开灯" in out["low_confidence_words"]


def test_ablation_disable_step1():
    cfg = _cfg()
    cfg["pipeline"]["enable_step1_confirm"] = False
    fake = _fake_factory({"手语 gloss": "开灯"})
    from agent import build_agent, DialogueState
    agent = build_agent(cfg, fake_completion=fake)
    state = DialogueState(cfg)
    gloss = {"gloss_sequence": ["开灯"], "confidence": [0.9],
             "timestamps": [[0.0, 0.5]]}
    out = agent.run(gloss, state)
    validate_agent_output(out)
    # 关闭 Step1 不应报错，输出仍合法
    assert out["natural_language"]


def test_ablation_disable_step3():
    cfg = _cfg()
    cfg["pipeline"]["enable_step3_decide"] = False
    fake = _fake_factory({"手语 gloss": "开灯"})
    from agent import build_agent, DialogueState
    agent = build_agent(cfg, fake_completion=fake)
    state = DialogueState(cfg)
    gloss = {"gloss_sequence": ["开灯"], "confidence": [0.9],
             "timestamps": [[0.0, 0.5]]}
    out = agent.run(gloss, state)
    validate_agent_output(out)
    assert out["intent"] == "chat"   # 关闭决策 -> 退化为 chat
    assert out["task"] == "none"


def test_cache_hit(tmp_path):
    import copy
    cfg = copy.deepcopy(_cfg())
    # 用独立临时缓存路径，避免与磁盘持久化缓存串扰
    cfg["cache"]["path"] = str(tmp_path / "agent_cache.json")
    if os.path.exists(cfg["cache"]["path"]):
        os.remove(cfg["cache"]["path"])
    calls = {"n": 0}

    def fake(prompt: str) -> str:
        # 仅统计 step2 翻译调用（step3 决策每次都会调，不计入缓存验证）
        if "手语 gloss" in prompt:
            calls["n"] += 1
        return "已为您开灯。"
    from agent import build_agent, DialogueState
    agent = build_agent(cfg, fake_completion=fake)
    state = DialogueState(cfg)
    gloss = {"gloss_sequence": ["开灯"], "confidence": [0.9],
             "timestamps": [[0.0, 0.5]]}
    agent.run(gloss, state)
    agent.run(gloss, state)  # 第二次应命中缓存
    assert calls["n"] == 1, "第二次翻译应命中缓存，未调用 LLM"
