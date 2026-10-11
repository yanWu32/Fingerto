"""层间契约（唯一事实源 / single source of truth）。

本文件定义四层系统的接口契约，任何层都必须导入本模块做自校验：
- 感知层输出 (T, 27, 3) 的骨骼张量
- 识别层输出 gloss JSON
- 智能体层输出 agent JSON

禁止在层内重新定义这些 schema；一律 from tests.contract import * 复用，
保证相邻层"对契约写码、互不阻塞"。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# 骨骼张量契约
# ---------------------------------------------------------------------------
# 27 点布局（已锁定）：主手 21 点（HAND_OFFSET=0，MediaPipe Hands 标准 21 点）
# + 上半身 6 点（POSE_OFFSET=21，Pose 子集：鼻/左肩/右肩/左肘/右肘/左腕）。
# 即"单主手 + 上身"，非双手。若识别层掉点需双手，须先把契约扩到 48 点布局。
# 顺序见 perception/skeleton.py 的 HAND_LANDMARK_IDX / POSE_LANDMARK_IDX。
NUM_POINTS = 27
NUM_HAND_POINTS = 21
NUM_POSE_POINTS = 6
HAND_OFFSET = 0
POSE_OFFSET = 21
# 上半身 6 点对应的 MediaPipe Pose 索引
POSE_LANDMARK_IDX = [0, 11, 12, 13, 14, 15]  # 鼻, 左肩, 右肩, 左肘, 右肘, 左腕
SKELETON_RANK = 3  # (x, y, z)
SKELETON_SHAPE = (-1, NUM_POINTS, SKELETON_RANK)  # (T, 27, 3)


def check_skeleton_shape(seq: Any) -> bool:
    """返回 seq 是否符合 (T, 27, 3) 且 dtype 数值、无 NaN/Inf。"""
    try:
        import numpy as np
    except Exception:  # numpy 缺失时退化为纯结构检查
        np = None
    if np is None:
        return isinstance(seq, (list, tuple)) and len(seq) > 0 and \
            all(len(t) == NUM_POINTS for t in seq) and \
            all(len(p) == SKELETON_RANK for t in seq for p in t)
    arr = np.asarray(seq, dtype=float)
    if arr.ndim != 3:
        return False
    if arr.shape[1] != NUM_POINTS or arr.shape[2] != SKELETON_RANK:
        return False
    return bool(np.all(np.isfinite(arr)))


# ---------------------------------------------------------------------------
# 识别层 → 智能体层 契约（gloss JSON）
# ---------------------------------------------------------------------------
GLOSS_SCHEMA = {
    "gloss_sequence": list,        # List[str]  手语词序列
    "confidence": list,            # List[float] 与 gloss_sequence 等长，[0,1]
    "timestamps": list,            # List[[start, end]] 单位秒，与 gloss_sequence 等长
}


def validate_gloss_output(obj: Any) -> bool:
    """校验识别层输出：合法返回 True，失败抛 ValueError（带说明）。

    合法示例：
        {
          "gloss_sequence": ["你好", "我"],
          "confidence": [0.92, 0.85],
          "timestamps": [[0.0, 1.2], [1.3, 2.0]]
        }
    """
    if not isinstance(obj, dict):
        raise ValueError(f"[gloss] 必须是 dict，收到 {type(obj).__name__}")
    for key, expected in GLOSS_SCHEMA.items():
        if key not in obj:
            raise ValueError(f"[gloss] 缺字段 '{key}'")
        if not isinstance(obj[key], expected):
            raise ValueError(f"[gloss] 字段 '{key}' 类型应为 {expected.__name__}，"
                             f"收到 {type(obj[key]).__name__}")
    n = len(obj["gloss_sequence"])
    if not (len(obj["confidence"]) == n and len(obj["timestamps"]) == n):
        raise ValueError("[gloss] gloss_sequence / confidence / timestamps 长度必须一致")
    for c in obj["confidence"]:
        if not (isinstance(c, (int, float)) and 0.0 <= float(c) <= 1.0):
            raise ValueError(f"[gloss] confidence 必须在 [0,1]，收到 {c!r}")
    for ts in obj["timestamps"]:
        if not (isinstance(ts, (list, tuple)) and len(ts) == 2 and ts[0] <= ts[1]):
            raise ValueError(f"[gloss] timestamps 元素须为 [start,end] 且 start<=end，收到 {ts!r}")
    for g in obj["gloss_sequence"]:
        if not isinstance(g, str):
            raise ValueError(f"[gloss] gloss_sequence 元素须为 str，收到 {g!r}")
    return True


# ---------------------------------------------------------------------------
# 智能体层 → 应用层 契约（agent JSON）
# ---------------------------------------------------------------------------
AGENT_SCHEMA = {
    "natural_language": str,       # 回复的自然语言文本
    "intent": str,                # 意图标识，如 query_weather / smart_home_control
    "task": str,                  # 可执行的任务标识；无任务时为 "none"
    "clarification": bool,        # 是否需要向用户澄清
    "low_confidence_words": list, # List[str] 低置信度手语词（供澄清/高亮）
}


def validate_agent_output(obj: Any) -> bool:
    """校验智能体层输出：合法返回 True，失败抛 ValueError（带说明）。

    合法示例：
        {
          "natural_language": "你想查询今天的天气吗？",
          "intent": "query_weather",
          "task": "weather_query",
          "clarification": False,
          "low_confidence_words": []
        }
    """
    if not isinstance(obj, dict):
        raise ValueError(f"[agent] 必须是 dict，收到 {type(obj).__name__}")
    for key, expected in AGENT_SCHEMA.items():
        if key not in obj:
            raise ValueError(f"[agent] 缺字段 '{key}'")
        val = obj[key]
        if expected is list:
            if not isinstance(val, list):
                raise ValueError(f"[agent] 字段 '{key}' 必须为 list，收到 {type(val).__name__}")
        elif expected is bool:
            if not isinstance(val, bool):
                raise ValueError(f"[agent] 字段 '{key}' 必须为 bool，收到 {type(val).__name__}")
        elif not isinstance(val, expected):
            raise ValueError(f"[agent] 字段 '{key}' 必须为 {expected.__name__}，"
                             f"收到 {type(val).__name__}")
    if obj["intent"] == "" or obj["task"] == "":
        raise ValueError("[agent] intent / task 不能为空字符串（无任务时 task='none'）")
    return True


# ---------------------------------------------------------------------------
# 跨层公共校验入口（供各层在交付前调用）
# ---------------------------------------------------------------------------
def assert_contract(layer: str, obj: Any) -> bool:
    """layer 取值 'gloss' 或 'agent'，对 obj 做对应校验；合法返回 True，失败抛 ValueError。"""
    if layer == "gloss":
        return validate_gloss_output(obj)
    elif layer == "agent":
        return validate_agent_output(obj)
    else:
        raise ValueError(f"未知契约层 {layer!r}（应为 'gloss' 或 'agent'）")
