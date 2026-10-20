"""把各 Participant 分片合并成 train/val/test（全局类表，保证标签一致）。

背景：沙箱禁用 multiprocessing.Pool（创建命名管道报 WinError 5），
故并行改为「每个 Participant 起一个独立进程跑 data/isw1000_extract.py --participants X」，
每个进程产出自己的 npz/{split}.npz（skeletons/labels/classes），最后用本脚本合并。

本脚本同时兼容两种分片格式（来自两段时期的抽取器）：
  - 新格式（data/isw1000_extract_resumable.py 写）：skeletons (N,T,27,3) + glosses (N,) object
  - 旧格式（data/isw1000_extract.py 写）：skeletons + labels (N,) + classes (本地类表)
  通过 data/isw1000_extract_resumable.load_shard 统一读成 (skeletons, glosses)，
  再按 participant 目录名归入 DEFAULT_SPLIT_BY_PARTICIPANT 对应的切分。

为什么需要 remap：全局类表由所有分片出现过的 gloss 排序得到；合并时把每个样本的
gloss 映射成全局索引，否则不同分片的同一 gloss 会拿到不同 label。

用法：
    python -m data.isw1000_merge                      # 默认读 data/isw1000_shards/*
    python -m data.isw1000_merge --out data/isw1000_real
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from data.isw1000_extract import DEFAULT_SPLIT_BY_PARTICIPANT
from data.isw1000_extract_resumable import load_shard
from tests.contract import check_skeleton_shape, NUM_POINTS, SKELETON_RANK

DEFAULT_SHARD_DIR = os.path.join("data", "isw1000_shards")
DEFAULT_OUT = os.path.join("data", "isw1000_real")


def build_global_classes(shard_dir: str) -> List[str]:
    """扫描所有分片（兼容新 skeletons/glosses 与旧 skeletons/labels/classes 两种格式），
    得到全局有序类表。"""
    classes = set()
    for pdir in sorted(Path(shard_dir).iterdir()):
        if not pdir.is_dir() or pdir.name.startswith("_"):
            continue
        for npz in sorted((pdir / "npz").glob("*.npz")):
            # 复用 load_shard：新格式返回 glosses，旧格式把本地 labels 反查成本地 classes
            _, gl = load_shard(npz)
            classes.update(gl)
    return sorted(classes)


def merge(shard_dir: str = DEFAULT_SHARD_DIR,
          out_dir: str = DEFAULT_OUT,
          split_map: Dict[str, List[str]] = None) -> dict:
    split_map = split_map or DEFAULT_SPLIT_BY_PARTICIPANT
    shard_dir_p = Path(shard_dir)
    classes = build_global_classes(shard_dir)
    c2i = {g: i for i, g in enumerate(classes)}

    out = Path(out_dir)
    (out / "npz").mkdir(parents=True, exist_ok=True)

    info: dict = {"dataset": "ISW-1000", "num_classes": len(classes), "splits": {}}

    for split, plist in split_map.items():
        sk_list, lb_list = [], []
        used = []
        for p in plist:
            npz_path = shard_dir_p / p / "npz" / "train.npz"
            if not npz_path.exists():
                print(f"  [warn] 缺少分片 {npz_path}，跳过")
                continue
            # 统一读成 (skeletons, glosses) —— 自动兼容新旧两种格式
            sk, gl = load_shard(npz_path)
            if not sk:
                print(f"  [warn] 分片 {npz_path} 为空，跳过")
                continue
            sk = np.asarray(sk, dtype=np.float32)
            global_labels = np.array([c2i[g] for g in gl], dtype=np.int64)
            sk_list.append(sk)
            lb_list.append(global_labels)
            used.append(p)
            print(f"  [{split}] +{p}: {len(global_labels)} 样本")

        if not sk_list:
            info["splits"][split] = {"num_samples": 0, "shape": None}
            continue

        sk_all = np.concatenate(sk_list, axis=0).astype(np.float32)
        lb_all = np.concatenate(lb_list, axis=0).astype(np.int64)

        # 契约校验（抽样，全量太慢）
        for i in range(0, len(sk_all), max(1, len(sk_all) // 50)):
            if not check_skeleton_shape(sk_all[i]):
                raise ValueError(f"{split} 样本 {i} 未通过 check_skeleton_shape")

        np.savez(
            out / "npz" / f"{split}.npz",
            skeletons=sk_all,
            labels=lb_all,
            classes=np.array(classes, dtype=object),
        )
        info["splits"][split] = {
            "num_samples": int(len(lb_all)),
            "shape": (int(len(lb_all)), sk_all.shape[1], NUM_POINTS, SKELETON_RANK),
            "participants": used,
        }

    (out / "glosses.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    info["glosses_file"] = str(out / "glosses.txt")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="合并 ISW-1000 分片")
    ap.add_argument("--shard-dir", default=DEFAULT_SHARD_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    info = merge(args.shard_dir, args.out)
    print("[isw1000-merge] 各切分：",
          {k: v["num_samples"] for k, v in info["splits"].items()})
    print(json.dumps({k: v for k, v in info.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
