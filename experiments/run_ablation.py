"""消融实验入口（mock 模式）。

在 mock gloss / agent 上跑 5 场景 × 4 系统，把指标表落盘到 experiments/results/：
  - ablation_metrics.csv   长表（每行一个 场景×系统）
  - ablation_metrics.json  同上 + 元数据（模式、系统映射、生成时间）

真实数据接入：把 load_dataset 换真实 npz 后，仅替换 run_all 的输入（scenarios），
本脚本其余逻辑（指标、落盘）无需重构。

用法（在仓库根 Fingerto 下）：
  python experiments/run_ablation.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

# 保证从仓库根导入 agent / configs / tests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs import load_config
from experiments.ablation import (
    ablation_fake_completion,
    build_scenarios,
    run_all,
    SYSTEM_LABELS,
)
from agent import SYSTEM_NAMES

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CSV_PATH = os.path.join(RESULTS_DIR, "ablation_metrics.csv")
JSON_PATH = os.path.join(RESULTS_DIR, "ablation_metrics.json")

CSV_FIELDS = [
    "scenario", "scenario_name", "system", "system_label",
    "bleu", "rouge1", "rougeL",
    "intent_accuracy", "clarification_rate", "completion_rate", "eot_latency_ms",
]


def _try_real_data() -> Dict:
    """容错加载真实数据集骨架；缺失时返回空，标注 mode=mock。"""
    try:
        from data import load_dataset
        skel, labels, classes = load_dataset("test")
        return {"available": True, "n_samples": int(skel.shape[0]),
                "n_classes": len(classes)}
    except FileNotFoundError:
        return {"available": False, "note": "未找到 data/synthetic/test.npz，使用 mock 场景"}
    except Exception as e:  # numpy 缺失等
        return {"available": False, "note": f"真实数据加载失败({type(e).__name__})，使用 mock 场景"}


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = load_config("agent")
    data_info = _try_real_data()

    results, flat = run_all(cfg, fake_completion=ablation_fake_completion,
                            scenarios=build_scenarios())

    # ---- 写 CSV ----
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in flat:
            w.writerow({k: row.get(k) for k in CSV_FIELDS})

    # ---- 写 JSON ----
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "mock" if not data_info.get("available") else "real_data",
        "data_info": data_info,
        "system_names": SYSTEM_NAMES,
        "system_labels": SYSTEM_LABELS,
        "scenarios": [{"id": s["id"], "name": s["name"]} for s in build_scenarios()],
        "metrics": results,
        "flat": flat,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- 控制台摘要 ----
    print(f"[run_ablation] mode={meta['mode']}  场景数={len(results)}  系统数={len(SYSTEM_NAMES)}")
    print(f"[run_ablation] csv -> {CSV_PATH}")
    print(f"[run_ablation] json-> {JSON_PATH}")
    print("\n场景 / 系统 / 意图准确率 / 澄清触发率 / 对话完成率 / 延迟(ms):")
    for sc_id, by_sys in results.items():
        for name, m in by_sys.items():
            print(f"  {sc_id:18s} {name}  "
                  f"intent={m['intent_accuracy']:.2f}  "
                  f"clar={m['clarification_rate']:.2f}  "
                  f"comp={m['completion_rate']:.2f}  "
                  f"lat={m['eot_latency_ms']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
