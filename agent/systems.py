"""消融实验四套系统（System A/B/C/D）工厂。

呼应方案确定书/开题报告的消融设计：
- System A（规则模板基线）：纯确定性规则，不调用 LLM，离线可跑，用于证明 LLM 的价值。
- System B（LLM 无对话状态）：只做 gloss->自然语言 的单轮翻译，不使用历史/指代/纠正。
- System C（LLM + 对话状态，主系统）：完整三步管线 + 多轮上下文 + 指代消解 + 纠正。
- System D（可选微调小模型）：本地小模型替代云端 LLM 的降级/对照；无权重时退化为 System C。

所有系统统一接口：
    run(gloss_json: dict, state: DialogueState | None) -> dict(agent JSON)

返回的 dict 均经 tests/contract.validate_agent_output 校验。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from agent.cache import TranslationCache
from agent.dialogue_state import DialogueState
from agent.llm_client import LLMClient
from agent.pipeline import AgentPipeline
from tests.contract import validate_agent_output, validate_gloss_output


# ----------------------------------------------------------------------------
# System A：规则模板基线（无 LLM）
# ----------------------------------------------------------------------------
class RuleTemplateSystem:
    """纯模板翻译：gloss 序列 -> 自然语言（固定映射表 + 简单拼接）。
    不支持指代/纠正/澄清，作为消融最弱基线。"""

    # 最小 gloss -> 词 映射（覆盖实验常用词；缺省时原样保留）
    GLOSS_TABLE = {
        "你": "你", "我": "我", "他": "他", "她": "她",
        "好": "好", "谢谢": "谢谢", "请": "请", "是": "是", "不": "不",
        "吃": "吃", "喝": "喝", "水": "水", "饭": "饭", "书": "书",
        "笔": "笔", "车": "车", "家": "家", "学校": "学校",
        "老师": "老师", "学生": "学生", "朋友": "朋友", "爱": "爱",
        "想": "想", "知道": "知道", "时间": "时间", "今天": "今天",
        "明天": "明天", "天气": "天气", "冷": "冷", "热": "热",
        "大": "大", "小": "小", "上": "上", "下": "下", "开": "开",
        "关": "关", "灯": "灯", "门": "门", "走": "走", "坐": "坐",
        "站": "站", "说": "说", "听": "听", "看": "看", "做": "做",
        "帮": "帮", "买": "买", "卖": "卖", "问": "问", "答": "答",
        "红": "红", "绿": "绿", "白": "白", "黑": "黑",
        "一": "一", "二": "二", "三": "三",
    }

    # 简单意图关键词（用于规则基线也能给出 command intent）
    ACTION_KEYWORDS = {
        "灯": ("turn_on_light", "关"),  # (action, 反向词)
        "窗": ("open_window", "关"),
        "帘": ("open_curtain", "关"),
        "空调": ("turn_on_ac", "关"),
    }

    def run(self, gloss_json: Dict[str, Any], state: Optional[DialogueState] = None) -> Dict[str, Any]:
        validate_gloss_output(gloss_json)
        gloss_seq: List[str] = list(gloss_json.get("gloss_sequence", []))
        conf: List[float] = list(gloss_json.get("confidence", []))

        # 模板翻译：逐词映射，缺省保留原 gloss
        words = [self.GLOSS_TABLE.get(g, g) for g in gloss_seq]
        text = "".join(words)  # 中文无空格

        # 规则意图：扫描动作关键词，遇"关"则取 turn_off 变体
        intent, task = "chat", "none"
        for kw, (on_action, off_word) in self.ACTION_KEYWORDS.items():
            if kw in text:
                if off_word in text:
                    task = on_action.replace("on", "off")
                else:
                    task = on_action
                intent = "command"
                break

        # 规则基线不支持澄清/指代：低置信词仅列出
        low = [g for g, c in zip(gloss_seq, conf) if c < 0.4]

        result = {
            "natural_language": text,
            "intent": intent,
            "task": task,
            "clarification": False,
            "low_confidence_words": low,
        }
        validate_agent_output(result)
        return result


# ----------------------------------------------------------------------------
# System B：LLM 无对话状态（单轮翻译）
# ----------------------------------------------------------------------------
class NoStateLLMSystem:
    """只做 gloss->自然语言 的单轮 LLM 翻译，丢弃历史/指代/纠正。
    用于消融：证明对话状态（System C）带来的增益。"""

    def __init__(self, cfg: dict, fake_completion: Optional[Callable[[str], str]] = None):
        self.cfg = cfg or {}
        llm_cfg = self.cfg.get("llm", {})
        self.client = LLMClient(llm_cfg, fake_completion=fake_completion)

    def run(self, gloss_json: Dict[str, Any], state: Optional[DialogueState] = None) -> Dict[str, Any]:
        validate_gloss_output(gloss_json)
        gloss_seq: List[str] = list(gloss_json.get("gloss_sequence", []))
        conf: List[float] = list(gloss_json.get("confidence", []))

        prompt = "请将以下手语 gloss 翻译成通顺中文：" + " ".join(gloss_seq)
        try:
            text = self.client.complete(
                "你是手语翻译助手，只输出翻译后的中文，不要解释。", prompt
            )
        except Exception:
            text = "".join(gloss_seq)

        low = [g for g, c in zip(gloss_seq, conf) if c < 0.4]
        result = {
            "natural_language": text.strip(),
            "intent": "chat",          # 无状态：不决策意图
            "task": "none",
            "clarification": False,
            "low_confidence_words": low,
        }
        validate_agent_output(result)
        return result


# ----------------------------------------------------------------------------
# System C：LLM + 对话状态（主系统，即 AgentPipeline 完整管线）
# ----------------------------------------------------------------------------
class FullStateLLMSystem:
    """完整管线：三步提示 + 多轮上下文 + 指代消解 + 纠正（System C）。"""

    def __init__(self, cfg: dict,
                 fake_completion: Optional[Callable[[str], str]] = None,
                 cache: Optional[TranslationCache] = None):
        self.cfg = cfg or {}
        self.pipeline = AgentPipeline(
            self.cfg, cache=cache,
            fake_completion=fake_completion,
        )

    def run(self, gloss_json: Dict[str, Any], state: Optional[DialogueState] = None) -> Dict[str, Any]:
        validate_gloss_output(gloss_json)
        state = state or DialogueState(self.cfg)
        result = self.pipeline.run(gloss_json, state)
        validate_agent_output(result)
        return result


# ----------------------------------------------------------------------------
# System D：可选微调小模型（本地降级/对照）
# ----------------------------------------------------------------------------
class FineTunedLocalSystem:
    """本地小模型替代云端 LLM。无可用权重时退化为 System C（保证可运行）。
    真实权重到位后，替换 _local_generate 即可。"""

    def __init__(self, cfg: dict,
                 fake_completion: Optional[Callable[[str], str]] = None,
                 fine_tuned_model_path: Optional[str] = None):
        self.cfg = cfg or {}
        self.model_path = fine_tuned_model_path
        self._available = fine_tuned_model_path is not None
        # 退化：复用完整管线（D 在消融中作为"本地可部署"对照）
        self._fallback = FullStateLLMSystem(self.cfg, fake_completion=fake_completion)

    def _local_generate(self, system: str, user: str) -> str:
        """占位：真实微调模型推理。未提供权重时抛错，交由 fallback。"""
        raise NotImplementedError("System D 微调权重未提供，使用 System C 退化")

    def run(self, gloss_json: Dict[str, Any], state: Optional[DialogueState] = None) -> Dict[str, Any]:
        if not self._available:
            return self._fallback.run(gloss_json, state)
        # 权重就绪路径：此处可调用本地模型；当前统一委托 fallback 保证契约一致
        return self._fallback.run(gloss_json, state)


# ----------------------------------------------------------------------------
# 工厂
# ----------------------------------------------------------------------------
def build_system(name: str, cfg: dict,
                 fake_completion: Optional[Callable[[str], str]] = None,
                 cache: Optional[TranslationCache] = None,
                 fine_tuned_model_path: Optional[str] = None):
    """按名称构建消融系统。

    name ∈ {"A", "B", "C", "D"} 或 {"rule", "llm_nostate", "full", "local"}。
    """
    name = str(name).upper()
    if name in ("A", "RULE", "RULE_TEMPLATE"):
        return RuleTemplateSystem()
    if name in ("B", "LLM_NOSTATE", "NOSTATE"):
        return NoStateLLMSystem(cfg, fake_completion=fake_completion)
    if name in ("C", "FULL", "FULL_STATE"):
        return FullStateLLMSystem(cfg, fake_completion=fake_completion, cache=cache)
    if name in ("D", "LOCAL", "FINETUNED"):
        return FineTunedLocalSystem(cfg, fake_completion=fake_completion,
                                    fine_tuned_model_path=fine_tuned_model_path)
    raise ValueError(f"未知系统: {name}，可选 A/B/C/D")


# 便于实验层引用
SYSTEM_NAMES = ["A", "B", "C", "D"]
