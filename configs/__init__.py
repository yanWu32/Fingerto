"""配置加载工具。

统一读取 configs/*.yaml，支持点号路径取值（如 "llm.model"）。

用法:
    from configs import load_config
    cfg = load_config("agent")
    print(cfg.get("llm.model"))
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parent


class Config:
    """简单配置容器，支持点号路径取值。"""

    def __init__(self, data: Dict[str, Any], name: str = ""):
        self._data = data or {}
        self.name = name

    def get(self, path: str, default: Any = None) -> Any:
        """按点号路径取值，如 get("llm.model")。"""
        cur: Any = self._data
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default
        return cur

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config(name={self.name!r}, keys={list(self._data.keys())})"


def load_config(name: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """加载配置文件。

    Args:
        name: 配置名（不含扩展名），如 "agent"、"perception"
        overrides: 覆盖项，点号路径 -> 值

    Raises:
        FileNotFoundError: 配置文件不存在
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if overrides:
        for dotpath, value in overrides.items():
            keys = dotpath.split(".")
            cur = data
            for k in keys[:-1]:
                cur = cur.setdefault(k, {})
            cur[keys[-1]] = value

    return Config(data, name=name)


def list_configs() -> list:
    """列出所有可用配置名。"""
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


if __name__ == "__main__":
    print("可用配置:", list_configs())
    for name in list_configs():
        cfg = load_config(name)
        print(f"  {name}: {cfg}")
