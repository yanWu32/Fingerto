"""LLM 客户端封装（OpenAI 兼容端点）。

支持 deepseek / qwen / local 三 provider，通过 configs/agent.yaml 配置。
API Key 一律从环境变量读取（.env 不入库）。

为便于无 Key 环境下自测，本模块同时支持注入一个 `fake_completion` 回调，
用于测试时替代真实网络调用（见 tests/test_agent.py）。
"""

from __future__ import annotations

import os
import time
from typing import Callable, List, Optional

try:
    from openai import OpenAI
except Exception:  # openai 缺失时，仅当真正调用才报错
    OpenAI = None


class LLMClient:
    """OpenAI 兼容客户端封装。"""

    def __init__(self, cfg: dict, fake_completion: Optional[Callable[[str], str]] = None):
        """
        Args:
            cfg: configs/agent.yaml 的 `llm` 段 dict
            fake_completion: 测试用假回调，接收 prompt 文本，返回 completion 文本
        """
        self.cfg = cfg or {}
        llm = self.cfg
        self.provider = llm.get("provider", "deepseek")
        endpoints = llm.get("endpoints", {})
        models = llm.get("models", {})
        api_key_env = llm.get("api_key_env", {})

        self.base_url = endpoints.get(self.provider, "https://api.deepseek.com/v1")
        self.model = models.get(self.provider, "deepseek-chat")
        self.temperature = float(llm.get("temperature", 0.3))
        self.max_tokens = int(llm.get("max_tokens", 800))
        self.timeout = int(llm.get("timeout", 30))
        self._api_key_env = api_key_env.get(self.provider, "DEEPSEEK_API_KEY")
        self._fake = fake_completion
        self._client = None  # 懒加载

    def _get_client(self):
        if self._client is None:
            if OpenAI is None:
                raise RuntimeError("openai 包未安装，无法创建真实 LLM 客户端")
            api_key = os.environ.get(self._api_key_env, "")
            if not api_key:
                raise RuntimeError(
                    f"未设置环境变量 {self._api_key_env}，无法调用 {self.provider}"
                )
            self._client = OpenAI(
                api_key=api_key, base_url=self.base_url, timeout=self.timeout
            )
        return self._client

    def complete(self, system: str, user: str) -> str:
        """一次对话补全，返回文本。"""
        if self._fake is not None:
            return self._fake(user)
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def build_messages(system: str, user: str) -> List[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
