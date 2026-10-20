"""ISW-1000 并行抽取驱动（按参与者并行 → 合并为 train/val/test）。

为什么要并行：单机串行实测 50 个视频≈393s（resize640+stride2，约 7.9s/视频），
全量 9914 视频串行约 21.6 小时，不可接受。按 10 个 Participant 并行（每个约 1000 视频），
墙钟可降到 ~2-3 小时（实际受 CPU 核数限制）。

产出与串行版 data/isw1000_extract.py **完全一致**：
    npz/{train,val,test}.npz（skeletons (N,T,27,3) / labels / classes）+ glosses.txt
因此 recognition.dataset.WlaslDataset / build_wlasl_dataloader 与
recognition/train.py --wlasl-root 可直接读，无需改动识别层。

切分沿用「按参与者划分」以避免签名者泄漏（见 data/isw1000_extract.py 说明）。

用法：
    python -m data.isw1000_extract_parallel                 # 全量并行（默认 workers=min(10, cpu-1)）
    python -m data.isw1000_extract_parallel --limit 20       # 冒烟
    python -m data.isw1000_extract_parallel --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from data.isw1000_extract import (
    DEFAULT_SPLIT_BY_PARTICIPANT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_DIR,
    DEFAULT_RESIZE_WIDTH,
    DEFAULT_FRAME_STRIDE,
    VIDEO_EXTS,
)
from data.wlasl_pose_extract import FIXED_T
from tests.contract import check_skeleton_shape, NUM_POINTS, SKELETON_RANK

DEFAULT_SHARD_DIR = os.path.join("data", "isw1000_shards")


def _worker(args) -> dict:
    """单个参与者的抽取任务（每个进程独立建 KeypointExtractor，MediaPipe 不可跨进程共享）。"""
    root, participant, T, resize, stride, limit, shard_dir = args
    # spawn 模式下子进程需重新导入（PYTHONPATH 已含仓库根）
    from data.wlasl_pose_extract import extract_one_video, KeypointExtractor

    pdir = Path(root) / participant
    vids = sorted([p for p in pdir.iterdir()
                   if p.is_file() and p.suffix in VIDEO_EXTS], key=lambda x: x.name)
    if limit:
        vids = vids[:limit]

    skeletons: List[np.ndarray] = []
    glosses: List[str] = []
    with KeypointExtractor() as ex:  # type: ignore[arg-type]
        for v in vids:
            try:
                seq = extract_one_video(
                    str(v), ex, frame_start=None, frame_end=None, T=T,
                    resize_width=resize, frame_stride=stride,
                )
            except Exception:
                continue  # 单样本失败不阻塞
            if seq is None or not check_skeleton_shape(seq):
                continue
            skeletons.append(seq)
            glosses.append(v.stem)   # 文件名即中文 gloss

    shard_path = Path(shard_dir) / f"{participant}.npz"
    if not skeletons:
        return {"participant": participant, "n": 0, "shard": None}
    np.savez(
        shard_path,
        skeletons=np.stack(skeletons, axis=0).astype(np.float32),
        glosses=np.array(glosses, dtype=object),
    )
    return {"participant": participant, "n": len(skeletons), "shard": str(shard_path)}


def merge_shards(shard_dir: str, split_map: Dict[str, List[str]],
                 out_dir: str, T: int = FIXED_T) -> dict:
    """把各参与者分片合并成 train/val/test（全局类表，保证标签一致）。"""
    shard_dir_p = Path(shard_dir)
    all_glosses = set()
    for f in sorted(shard_dir_p.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        all_glosses.update(str(g) for g in d["glosses"])

    classes = sorted(all_glosses)
    c2i = {g: i for i, g in enumerate(classes)}

    out = Path(out_dir)
    (out / "npz").mkdir(parents=True, exist_ok=True)

    info: dict = {"dataset": "ISW-1000", "num_classes": len(classes), "splits": {}}
    for split, plist in split_map.items():
        sk_list, lb_list = [], []
        for p in plist:
            f = shard_dir_p / f"{p}.npz"
            if not f.exists():
                continue
            d = np.load(f, allow_pickle=True)
            sk_list.append(np.asarray(d["skeletons"], dtype=np.float32))
            lb_list.extend(c2i[str(g)] for g in d["glosses"])

        if not sk_list:
            info["splits"][split] = {"num_samples": 0, "shape": None}
            continue
        np.savez(
            out / "npz" / f"{split}.npz",
            skeletons=np.concatenate(sk_list, axis=0).astype(np.float32),
            labels=np.array(lb_list, dtype=np.int64),
            classes=np.array(classes, dtype=object),
        )
        info["splits"][split] = {
            "num_samples": len(lb_list),
            "shape": (len(lb_list), T, NUM_POINTS, SKELETON_RANK),
            "participants": plist,
        }

    (out / "glosses.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    info["glosses_file"] = str(out / "glosses.txt")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="ISW-1000 并行骨架抽取")
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    ap.add_argument("--shard-dir", default=DEFAULT_SHARD_DIR)
    ap.add_argument("--T", type=int, default=FIXED_T)
    ap.add_argument("--limit", type=int, default=None,
                    help="每个参与者最多抽取 N 个（冒烟）")
    ap.add_argument("--workers", type=int, default=0, help="并行进程数，0=自动")
    ap.add_argument("--resize-width", type=int, default=DEFAULT_RESIZE_WIDTH)
    ap.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"未找到数据集目录：{root}")

    participants = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not participants:
        raise RuntimeError(f"{root} 下未发现 Participant_* 目录")

    cpu = os.cpu_count() or 4
    workers = args.workers if args.workers > 0 else max(1, min(len(participants), cpu - 1))
    print(f"[isw1000-parallel] 参与者={len(participants)}  workers={workers}  cpu={cpu}")

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(args.dataset_root, p, args.T, args.resize_width,
              args.frame_stride, args.limit, str(shard_dir))
             for p in participants]

    t0 = time.time()
    results: List[dict] = []
    if workers == 1:
        for t in tasks:
            results.append(_worker(t))
    else:
        with Pool(processes=workers) as pool:
            for r in pool.imap_unordered(_worker, tasks):
                print(f"  [done] {r['participant']}: {r['n']} 样本")
                results.append(r)
    elapsed = time.time() - t0
    print(f"[isw1000-parallel] 抽取完成，用时 {elapsed/60:.1f} 分钟，"
          f"合计 {sum(r['n'] for r in results)} 样本")

    info = merge_shards(str(shard_dir), DEFAULT_SPLIT_BY_PARTICIPANT, args.out, args.T)
    info["elapsed_sec"] = round(elapsed, 1)
    print("[isw1000-parallel] 各切分：",
          {k: v["num_samples"] for k, v in info["splits"].items()})
    print("[isw1000-parallel] 完成：", json.dumps(
        {k: v for k, v in info.items() if k != "skipped"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
