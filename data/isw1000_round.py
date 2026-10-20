"""ISW-1000 累积回合驱动（单条命令跑完「抽取→合并→状态」）。

策略：50 词 × 10 人 = 500 视频（gloss 集合锁定自 data/isw1000_subset/gloss_plan.json）。
沙箱会杀长时间后台进程，故每回合只跑一个 90s 时间窗，边跑边存盘；反复调用
本脚本即可逐步累积到 500，再出初步识别指标（Top-1/Top-5）。

用法：
    python -m data.isw1000_round                       # 跑一回合（90s 抽取 + 合并）
    python -m data.isw1000_round --budget-sec 120      # 自定义时间窗
    python -m data.isw1000_round --skip-extract        # 仅合并 + 状态（不抽）
    python -m data.isw1000_round --status-only         # 仅打印进度
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SHARD_DIR = os.path.join("data", "isw1000_subset")
PLAN_PATH = os.path.join(SHARD_DIR, "gloss_plan.json")
MERGE_OUT = os.path.join("data", "isw1000_real")


def main() -> int:
    ap = argparse.ArgumentParser(description="ISW-1000 累积回合驱动")
    ap.add_argument("--budget-sec", type=float, default=90.0)
    ap.add_argument("--shard-dir", default=SHARD_DIR)
    ap.add_argument("--plan", default=PLAN_PATH)
    ap.add_argument("--merge-out", default=MERGE_OUT)
    ap.add_argument("--skip-extract", action="store_true", help="不抽取，仅合并")
    ap.add_argument("--status-only", action="store_true", help="仅打印进度")
    ap.add_argument("--resize-width", type=int, default=640)
    ap.add_argument("--frame-stride", type=int, default=2)
    args = ap.parse_args()

    plan_glosses = None
    if os.path.isfile(args.plan):
        _d = json.load(open(args.plan, encoding="utf-8"))
        plan_glosses = _d.get("glosses") or _d.get("per_gloss_total") or _d
        if isinstance(plan_glosses, dict):
            plan_glosses = list(plan_glosses.keys())

    if args.status_only:
        from data.isw1000_extract_resumable import status
        print("[status]", status(args.shard_dir))
        return 0

    if not args.skip_extract:
        from data.isw1000_extract_resumable import run_budget, status
        print(f"[round] 抽取 {args.budget_sec}s，gloss 锁定={len(plan_glosses) if plan_glosses else '全部'}")
        info = run_budget(
            shard_dir=args.shard_dir,
            budget_sec=args.budget_sec,
            gloss_list=plan_glosses,
            resize_width=args.resize_width,
            frame_stride=args.frame_stride,
        )
        print("[round] extract:", info)
        print("[round] shard status:", status(args.shard_dir))

    # 合并（兼容新旧分片格式）
    from data.isw1000_merge import merge
    print("[round] 合并分片 ->", args.merge_out)
    info = merge(args.shard_dir, args.merge_out)
    print("[round] merge:", {k: (v["num_samples"] if isinstance(v, dict) else v)
                             for k, v in info.get("splits", {}).items()})
    print("[round] 总样本:", sum(v["num_samples"] for v in info.get("splits", {}).values()
                                 if isinstance(v, dict)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
