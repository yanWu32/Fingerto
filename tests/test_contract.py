"""契约单元测试：验证 tests.contract 里的校验器本身行为正确。

运行：在 Fingerto/ 下 `pytest tests/` （需 fingerto 环境）。
这是"契约地基"的第一道闸——任何层出 schema 偏差都会被这里拦下。
"""

import math

from tests.contract import (
    NUM_POINTS,
    SKELETON_RANK,
    check_skeleton_shape,
    validate_agent_output,
    validate_gloss_output,
)


# ----------------------------- gloss 契约 -----------------------------
def _good_gloss():
    return {
        "gloss_sequence": ["你好", "我"],
        "confidence": [0.92, 0.85],
        "timestamps": [[0.0, 1.2], [1.3, 2.0]],
    }


def test_gloss_ok():
    validate_gloss_output(_good_gloss())  # 不应抛异常


def test_gloss_bad():
    import pytest
    cases = [
        lambda d: d.pop("gloss_sequence"),
        lambda d: d.update(confidence=[0.9]),             # 长度不一致
        lambda d: d.update(confidence=[1.5, 0.2]),        # 越界
        lambda d: d.update(timestamps=[[2.0, 1.0]]),      # start>end
        lambda d: d.update(gloss_sequence=["你好", 1]),   # 非 str
        lambda d: d.update(confidence="0.9"),             # 类型错
    ]
    for mut in cases:
        d = _good_gloss()
        mut(d)
        with pytest.raises(ValueError):
            validate_gloss_output(d)


# ----------------------------- agent 契约 -----------------------------
def _good_agent():
    return {
        "natural_language": "你想查询今天的天气吗？",
        "intent": "query_weather",
        "task": "weather_query",
        "clarification": False,
        "low_confidence_words": [],
    }


def test_agent_ok():
    validate_agent_output(_good_agent())


def test_agent_bad():
    import pytest
    cases = [
        lambda d: d.pop("natural_language"),
        lambda d: d.update(clarification="no"),           # 非 bool
        lambda d: d.update(intent=""),                    # 空 intent
        lambda d: d.update(task=""),                      # 空 task
        lambda d: d.update(low_confidence_words="我"),     # 非 list
    ]
    for mut in cases:
        d = _good_agent()
        mut(d)
        with pytest.raises(ValueError):
            validate_agent_output(d)


# ----------------------------- skeleton 契约 -----------------------------
def test_skeleton_shape_ok():
    import numpy as np
    seq = np.random.randn(10, NUM_POINTS, SKELETON_RANK)
    assert check_skeleton_shape(seq) is True


def test_skeleton_shape_bad_rank():
    import numpy as np
    seq = np.random.randn(10, NUM_POINTS, 2)
    assert check_skeleton_shape(seq) is False


def test_skeleton_shape_nan():
    import numpy as np
    seq = np.full((5, NUM_POINTS, SKELETON_RANK), math.nan)
    assert check_skeleton_shape(seq) is False
