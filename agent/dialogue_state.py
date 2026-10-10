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

    # ---- 写入 ----
    def add_turn(self, role: str, content: str, intent: str = "",
                 task: str = "", gloss: Any = None) -> None:
        self.history.append({
            "role": role, "content": content,
            "intent": intent, "task": task, "gloss": gloss,
        })
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

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
