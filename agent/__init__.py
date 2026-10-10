"""智能体层（核心）— LLM 三步提示管线 + 对话状态管理。

Step1 确认（低置信消歧）
Step2 翻译（gloss -> 自然语言）
Step3 决策（意图识别 / 任务生成）

契约输入：{"gloss_sequence", "confidence", "timestamps"}
契约输出：{"natural_language", "intent", "task", "clarification", "low_confidence_words"}
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from agent.cache import TranslationCache
from agent.dialogue_state import DialogueState
from agent.llm_client import LLMClient
from agent.pipeline import AgentPipeline
from tests.contract import validate_agent_output, validate_gloss_output


def build_agent(cfg: Dict[str, Any],
                fake_completion: Optional[Callable[[str], str]] = None) -> AgentPipeline:
    """按 configs/agent.yaml 装配智能体：LLM 客户端 + 缓存 + 管线。

    fake_completion 用于测试（无需真实 API Key）。
    """
    cache = TranslationCache(cfg.get("cache", {})) if cfg.get("cache") else None
    client = LLMClient(cfg.get("llm", {}), fake_completion=fake_completion)
    return AgentPipeline(cfg, client=client, cache=cache)


def run_agent(gloss_json: Dict[str, Any],
              state: Optional[DialogueState] = None,
              cfg: Optional[Dict[str, Any]] = None,
              fake_completion: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """便捷入口：gloss JSON -> agent JSON。

    - 校验输入符合 gloss 契约
    - 用 build_agent 装配；无 cfg 时调用方需先 load_config
    - 返回经 agent 契约校验的输出
    """
    validate_gloss_output(gloss_json)  # 输入闸门
    if cfg is None:
        from configs import load_config
        cfg = load_config("agent")
    state = state or DialogueState(cfg)
    agent = build_agent(cfg, fake_completion=fake_completion)
    result = agent.run(gloss_json, state)
    validate_agent_output(result)      # 输出闸门
    return result


__all__ = [
    "AgentPipeline", "DialogueState", "LLMClient", "TranslationCache",
    "build_agent", "run_agent", "validate_gloss_output", "validate_agent_output",
]
