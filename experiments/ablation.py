"""消融实验封装：System A/B/C/D × 5 类场景。

职责（实验层，不修改 agent/ 任何代码）：
- 复用 agent.systems.build_system 实例化四套系统
- 定义 5 类评测场景（开题报告口径，mock gloss + 参考输出）
- 在 fake_completion 下跑出 per-turn 记录，交 experiments.metrics 聚合

四系统映射（来自 agent/systems.SYSTEM_NAMES，已冻结）：
  A -> RuleTemplateSystem        规则模板基线（不调 LLM）
  B -> NoStateLLMSystem          LLM 无对话状态（单轮翻译）
  C -> FullStateLLMSystem        LLM + 对话状态（主系统，完整三步管线）
  D -> FineTunedLocalSystem       可选微调小模型（无权重时退化为 C）

识别层只出 gloss、不做决策——本模块绝不在识别层塞决策逻辑，
所有意图/澄清决策均来自四套系统的 agent 输出（契约 natural_language/intent/...）。
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

from experiments.metrics import aggregate

# 复用已冻结的接口
from agent import build_system, SYSTEM_NAMES
from agent.dialogue_state import DialogueState

# 系统可读名（用于报表）
SYSTEM_LABELS = {
    "A": "RuleTemplate(规则基线)",
    "B": "NoStateLLM(无状态)",
    "C": "FullStateLLM(主系统)",
    "D": "FineTunedLocal(退化→C)",
}


# ---------------------------------------------------------------------------
# fake_completion：mock 模式确定性假 LLM
# ---------------------------------------------------------------------------
def ablation_fake_completion(prompt: str) -> str:
    """根据 prompt 模板返回确定性文本，使四系统的真实架构差异得以显现。

    - 决策 prompt（含 '允许控制动作'）：依据上下文/本轮表达判 intent 与 task
    - 确认 prompt（含 '当前 gloss：'）：剥离 <lowconf> 标注，保留原词序列
    - 翻译 prompt（含 '手语 gloss：' 或 B 的 '翻译成通顺中文：'）：回显 gloss
    """
    is_decision = "允许控制动作" in prompt

    if is_decision:
        m = re.search(r"用户本轮表达：(.+)", prompt)
        nl = m.group(1).strip() if m else ""
        # 优先级：窗 > 灯（纠正/多义场景用），其余当 chat
        if "窗" in prompt:
            task, intent = "open_window", "command"
        elif "开" in prompt and "灯" in prompt:
            task, intent = "turn_on_light", "command"
        elif "关" in prompt and "灯" in prompt:
            task, intent = "turn_off_light", "command"
        else:
            task, intent = "none", "chat"
        out_nl = nl if nl else ("开窗" if task == "open_window"
                                else "开灯" if task == "turn_on_light"
                                else "关灯" if task == "turn_off_light" else "好的")
        return json.dumps(
            {"intent": intent, "task": task, "natural_language": out_nl,
             "need_clarify": False}, ensure_ascii=False)

    # 确认 prompt
    if "当前 gloss：" in prompt:
        body = prompt.split("当前 gloss：", 1)[1]
        out = []
        for t in body.split():
            mm = re.match(r"<lowconf:(.+?):[\d.]+>", t)
            out.append(mm.group(1) if mm else t)
        return " ".join(out)

    # 翻译 prompt（C 的 step2）
    m = re.search(r"手语 gloss：(.+)", prompt)
    if m:
        return m.group(1).strip().replace(" ", "")
    # B 的翻译 prompt
    if "翻译成通顺中文" in prompt:
        return prompt.split("翻译成通顺中文：", 1)[1].strip().replace(" ", "")
    return prompt


# ---------------------------------------------------------------------------
# 5 类评测场景（mock gloss + 参考输出）
# ---------------------------------------------------------------------------
def _gloss(seq: List[str], conf: List[float]) -> Dict:
    return {
        "gloss_sequence": seq,
        "confidence": conf,
        "timestamps": [[i * 0.5, (i + 1) * 0.5] for i in range(len(seq))],
    }


def build_scenarios() -> List[Dict]:
    """返回 5 类场景；每场景含 turns(gloss 列表) 与 refs(每轮参考输出)。"""
    # 每个序列单独给置信度，长度必须与 gloss_sequence 对齐
    return [
        {  # ① 孤立词翻译
            "id": "isolated_word",
            "name": "孤立词翻译",
            "turns": [_gloss(["我", "想", "喝", "水"], [0.9, 0.9, 0.9, 0.9])],
            "refs": [{"intent": "chat", "clarification": False, "nl": "我想喝水"}],
        },
        {  # ② 低置信度消歧（喝 0.3 < clarify_threshold 0.4）
            "id": "low_conf_disambig",
            "name": "低置信度消歧",
            "turns": [_gloss(["我", "想", "喝", "水"], [0.9, 0.9, 0.3, 0.9])],
            "refs": [{"intent": "chat", "clarification": True, "nl": "我想喝水"}],
        },
        {  # ③ 多轮指代（turn1 开灯打底，turn2 用「它」指代灯）
            "id": "coreference",
            "name": "多轮指代",
            "turns": [
                _gloss(["开", "灯"], [0.9, 0.9]),
                _gloss(["它", "开"], [0.9, 0.9]),
            ],
            "refs": [
                {"intent": "command", "clarification": False, "nl": "开灯"},
                {"intent": "command", "clarification": False, "nl": "它开"},
            ],
        },
        {  # ④ 用户纠正（turn1 开灯，turn2 纠正为开窗）
            "id": "user_correction",
            "name": "用户纠正",
            "turns": [
                _gloss(["开", "灯"], [0.9, 0.9]),
                _gloss(["不对", "开", "窗"], [0.9, 0.9, 0.9]),
            ],
            "refs": [
                {"intent": "command", "clarification": False, "nl": "开灯"},
                {"intent": "command", "clarification": False, "nl": "不对开窗"},
            ],
        },
        {  # ⑤ 多义手势（单 gloss「开」低置信，需澄清目标）
            "id": "polysemous",
            "name": "多义手势",
            "turns": [_gloss(["开"], [0.3])],
            "refs": [{"intent": "chat", "clarification": True, "nl": "开"}],
        },
    ]


# ---------------------------------------------------------------------------
# 构建系统
# ---------------------------------------------------------------------------
def build_systems(cfg: dict,
                  fake_completion: Optional[Callable[[str], str]] = None
                  ) -> Dict[str, object]:
    """按 SYSTEM_NAMES 构建四套系统。"""
    return {name: build_system(name, cfg, fake_completion=fake_completion)
            for name in SYSTEM_NAMES}


# ---------------------------------------------------------------------------
# 跑单场景 × 单系统 -> per-turn 记录
# ---------------------------------------------------------------------------
def run_scenario_system(system, scenario: Dict, cfg: dict
                        ) -> List[Dict]:
    """对给定系统跑完该场景所有轮，返回 per-turn 记录。多轮场景复用同一 DialogueState。"""
    state = DialogueState(cfg)
    records: List[Dict] = []
    for turn, ref in zip(scenario["turns"], scenario["refs"]):
        t0 = time.perf_counter()
        out = system.run(turn, state)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        records.append({
            "hyp_nl": out.get("natural_language", ""),
            "ref_nl": ref["nl"],
            "pred_intent": out.get("intent", ""),
            "ref_intent": ref["intent"],
            "clarification": bool(out.get("clarification")),
            "ref_clarification": bool(ref["clarification"]),
            "latency_ms": dt_ms,
        })
    return records


def run_all(cfg: dict,
            fake_completion: Optional[Callable[[str], str]] = None,
            scenarios: Optional[List[Dict]] = None
            ) -> Tuple[Dict, List[Dict]]:
    """跑全部 5 场景 × 4 系统。

    Returns:
        results: {scenario_id: {system: metrics_dict}}
        flat:    长表记录 [{scenario, system, ...metrics}]
    """
    scenarios = scenarios or build_scenarios()
    systems = build_systems(cfg, fake_completion=fake_completion)
    results: Dict[str, Dict[str, dict]] = {}
    flat: List[Dict] = []
    for sc in scenarios:
        results[sc["id"]] = {}
        for name, sys_obj in systems.items():
            recs = run_scenario_system(sys_obj, sc, cfg)
            m = aggregate(recs)
            results[sc["id"]][name] = m
            flat.append({
                "scenario": sc["id"],
                "scenario_name": sc["name"],
                "system": name,
                "system_label": SYSTEM_LABELS.get(name, name),
                **m,
            })
    return results, flat
