"""对话状态管理（多轮指代消解 / 用户纠正）。

状态结构（可被 Step3 决策使用）：
- history: List[dict]  {"role", "content", "intent", "task", "gloss"}
- env: 模拟环境状态（智能家居等设备状态，由外部 JSON 提供）
- preferences: 用户偏好（如单位、语言）
- pending_intent: 待澄清/未完成的意图
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List


class DialogueState:
    def __init__(self, cfg: dict):
        dlg = (cfg or {}).get("dialogue", {})
        self.max_history = int(dlg.get("max_history", 10))
        self.max_context_turns = int(dlg.get("max_context_turns", 5))
        self.enable_coreference = bool(dlg.get("enable_coreference", True))
        self.enable_correction = bool(dlg.get("enable_correction", True))

        self.history: List[Dict[str, Any]] = []
        self.env: Dict[str, Any] = {}        # 设备状态
        self.preferences: Dict[str, Any] = {}
        self.pending_intent: Dict[str, Any] = {}  # 待澄清意图
        self.last_entity: str = ""           # 最近提及的实体（指代消解锚点）

    # ---- 写入 ----
    def add_turn(self, role: str, content: str, intent: str = "",
                 task: str = "", gloss: Any = None, entity: str = "") -> None:
        self.history.append({
            "role": role, "content": content,
            "intent": intent, "task": task, "gloss": gloss,
        })
        if entity:
            self.last_entity = entity
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def record_entity(self, entity: str) -> None:
        """记录本轮提及的实体，供后续指代消解复用。"""
        if entity:
            self.last_entity = entity

    def resolve_coreference(self, text: str) -> str:
        """轻量确定性指代消解：把 它/这个/那个 替换为最近提及实体。
        仅当 enable_coreference 且存在锚点时生效（System B 关闭）。
        """
        if not (self.enable_coreference and self.last_entity):
            return text
        for pronoun in ("这个", "那个", "它"):
            text = text.replace(pronoun, self.last_entity)
        return text

    def detect_correction(self, text: str) -> bool:
        """判定用户是否在本轮做了纠正（System C 启用纠正时使用）。"""
        if not self.enable_correction:
            return False
        markers = ("不对", "不是", "错了", "搞错", "我是说", "其实", "重新", "更正")
        return any(m in text for m in markers)

    def set_env(self, env: Dict[str, Any]) -> None:
        self.env = dict(env or {})

    def set_preference(self, key: str, value: Any) -> None:
        self.preferences[key] = value

    def set_pending(self, intent: Dict[str, Any]) -> None:
        self.pending_intent = intent or {}

    def clear_pending(self) -> None:
        self.pending_intent = {}

    # ---- 读取 ----
    def recent_context(self) -> List[Dict[str, Any]]:
        """供 LLM 使用的近 N 轮上下文。"""
        return self.history[-self.max_context_turns:]

    def context_text(self) -> str:
        """把上下文拼成可读文本，便于注入 prompt。"""
        lines = []
        for h in self.recent_context():
            who = "用户" if h["role"] == "user" else "助手"
            lines.append(f"{who}: {h['content']}")
        return "\n".join(lines)

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy({
            "history": self.history,
            "env": self.env,
            "preferences": self.preferences,
            "pending_intent": self.pending_intent,
        })
