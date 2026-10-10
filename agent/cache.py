"""高频词/短语翻译缓存（降低 API 延迟与开销）。"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional


class TranslationCache:
    def __init__(self, cfg: dict):
        c = cfg or {}
        self.enabled = bool(c.get("enabled", True))
        self.max_size = int(c.get("max_size", 1000))
        self.ttl = int(c.get("ttl", 86400))
        self.path = c.get("path", "data/cache/agent_cache.json")
        self._store: dict = {}
        if self.enabled:
            self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self._store = json.load(f)
        except Exception:
            self._store = {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._store, f, ensure_ascii=False)
        except Exception:
            pass

    def get(self, key: str) -> Optional[str]:
        if not self.enabled:
            return None
        item = self._store.get(key)
        if item is None:
            return None
        ts, val = item
        if self.ttl > 0 and (time.time() - ts) > self.ttl:
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: str):
        if not self.enabled:
            return
        if len(self._store) >= self.max_size:
            # 简单 FIFO：弹出最早
            oldest = min(self._store, key=lambda k: self._store[k][0])
            self._store.pop(oldest, None)
        self._store[key] = [time.time(), value]
        self._save()
