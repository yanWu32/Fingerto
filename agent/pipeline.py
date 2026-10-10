"""智能体层三步提示管线（论文核心）。

Step1 确认：对低置信 gloss 做消歧澄清（可关闭→消融 System B）
Step2 翻译：gloss 序列 -> 自然语言文本
Step3 决策：意图识别 / 任务生成（智能家居等）

入口 `run_pipeline(gloss_json, state, client)` 整合三步，输出符合
tests/contract.py 的 agent JSON：
  {"natural_language", "intent", "task", "clarification", "low_confidence_words"}

设计要点（呼应方案确定书）：
- 识别层只给 gloss **不决策**，所有决策在智能体层。
- 低置信词触发澄清或结合上下文推断（置信度路由见 confidence 配置）。
- 高频词/短语走缓存，降低延迟与 API 开销。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from agent.dialogue_state import DialogueState
from agent.llm_client import LLMClient
from tests.contract import validate_agent_output


# ----- prompt 模板 -----
SYSTEM_TRANSLATE = (
    "你是手语对话系统的智能体。用户输入是手语识别得到的 gloss 序列"
    "（手语词，可能含 <unk> 表示无法识别）。请把它翻译成自然、通顺的中文句子，"
    "保持原意，不要臆造未出现的信息。"
)
SYSTEM_CONFIRM = (
    "手语序列中有低置信词（已标注 <lowconf:词:分数>）。请结合上下文判断其最可能的词，"
    "若不确定请标记为需要澄清。只输出修正后的 gloss 序列，用空格分隔，低置信词用"
    "尖括号标注你仍不确定的：<lowconf:词>。"
)
SYSTEM_DECISION = (
    "根据对话上下文与用户提供/环境，判断用户意图并生成可执行任务。"
    "意图类型限定为：chat（闲聊）/ query（查询，如天气时间）/ command（控制，如智能家居）/ clarify（需澄清）。"
    "若为 command，任务标识取自允许动作列表；否则 task 填 'none'。"
    "输出严格 JSON：{\"intent\": str, \"task\": str, \"natural_language\": str, \"need_clarify\": bool}。"
)

ALLOWED_ACTIONS = [
    "turn_on_light", "turn_off_light", "turn_on_ac", "turn_off_ac",
    "open_window", "close_window", "open_curtain", "close_curtain",
]


def _gloss_tokens(gloss_seq: List[str], conf: List[float]) -> List[str]:
    """把 gloss 与置信度拼成带标注的 token 串，供 prompt。"""
    out = []
    for g, c in zip(gloss_seq, conf):
        out.append(f"<lowconf:{g}:{c:.2f}>" if c < 0.75 else g)
    return out


def _parse_decision_json(text: str) -> Dict[str, Any]:
    """从 LLM 文本中提取 JSON（容错：找第一个 {...}）。"""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        # 兜底：纯文本当 chat
        return {"intent": "chat", "task": "none", "natural_language": text.strip(),
                "need_clarify": False}


class AgentPipeline:
    def __init__(self, cfg: dict, client: Optional[LLMClient] = None,
                 cache=None, fake_completion=None):
        self.cfg = cfg or {}
        self.pipeline_cfg = self.cfg.get("pipeline", {})
        self.conf_cfg = self.cfg.get("confidence", {})
        self.client = client or LLMClient(self.cfg.get("llm", {}),
                                          fake_completion=fake_completion)
        self.cache = cache

    # ----- 置信度路由 -----
    def low_confidence_words(self, gloss_seq: List[str], conf: List[float]) -> List[str]:
        t = float(self.conf_cfg.get("clarify_threshold", 0.4))
        return [g for g, c in zip(gloss_seq, conf) if c < t]

    def needs_clarify(self, conf: List[float]) -> bool:
        t = float(self.conf_cfg.get("clarify_threshold", 0.4))
        return any(c < t for c in conf)

    # ----- 三步 -----
    def step1_confirm(self, gloss_seq: List[str], conf: List[float]) -> List[str]:
        """Step1：低置信消歧。关闭时原样返回。"""
        if not self.pipeline_cfg.get("enable_step1_confirm", True):
            return gloss_seq
        if not self.needs_clarify(conf):
            return gloss_seq
        tokens = _gloss_tokens(gloss_seq, conf)
        prompt = "当前 gloss：" + " ".join(tokens)
        try:
            out = self.client.complete(SYSTEM_CONFIRM, prompt)
            cleaned = [w for w in out.replace("<lowconf:", "").replace(">", " ").split()
                       if w]
            return cleaned if cleaned else gloss_seq
        except Exception:
            return gloss_seq  # 调用失败不阻断，保留原序列

    def step2_translate(self, gloss_seq: List[str], state: DialogueState) -> str:
        """Step2：gloss -> 自然语言。支持缓存。"""
        key = "translate:" + " ".join(gloss_seq)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
        ctx = state.context_text()
        prompt = (f"对话上下文：\n{ctx}\n\n手语 gloss：{' '.join(gloss_seq)}")
        text = self.client.complete(SYSTEM_TRANSLATE, prompt)
        if self.cache is not None:
            self.cache.set(key, text)
        return text

    def step3_decide(self, natural_language: str, state: DialogueState) -> Dict[str, Any]:
        """Step3：意图/任务决策。"""
        if not self.pipeline_cfg.get("enable_step3_decide", True):
            return {"intent": "chat", "task": "none",
                    "natural_language": natural_language, "need_clarify": False}
        ctx = state.context_text()
        env = json.dumps(state.env, ensure_ascii=False)
        prompt = (f"对话上下文：\n{ctx}\n\n环境状态：{env}\n\n"
                  f"用户本轮表达：{natural_language}\n\n"
                  f"允许控制动作：{', '.join(ALLOWED_ACTIONS)}")
        try:
            out = self.client.complete(SYSTEM_DECISION, prompt)
            dec = _parse_decision_json(out)
        except Exception:
            dec = {"intent": "chat", "task": "none",
                   "natural_language": natural_language, "need_clarify": False}
        dec.setdefault("natural_language", natural_language)
        dec.setdefault("intent", "chat")
        dec.setdefault("task", "none")
        dec.setdefault("need_clarify", False)
        return dec

    # ----- 整合入口 -----
    def run(self, gloss_json: Dict[str, Any], state: DialogueState) -> Dict[str, Any]:
        """输入识别层 gloss JSON，输出 agent JSON（经契约校验）。"""
        gloss_seq = list(gloss_json.get("gloss_sequence", []))
        conf = list(gloss_json.get("confidence", []))
        low = self.low_confidence_words(gloss_seq, conf)
        need_clarify = self.needs_clarify(conf)

        # Step1 确认（消歧）
        confirmed = self.step1_confirm(gloss_seq, conf)
        # Step2 翻译
        nl = self.step2_translate(confirmed, state)
        # Step3 决策
        dec = self.step3_decide(nl, state)

        # 写入对话状态
        state.add_turn(role="user", content=nl, intent=dec.get("intent", "chat"),
                       task=dec.get("task", "none"), gloss=gloss_seq)
        if need_clarify:
            dec["need_clarify"] = True

        result = {
            "natural_language": nl,
            "intent": dec.get("intent", "chat"),
            "task": dec.get("task", "none"),
            "clarification": bool(dec.get("need_clarify", False) or need_clarify),
            "low_confidence_words": low,
        }
        validate_agent_output(result)  # 契约闸门
        return result
