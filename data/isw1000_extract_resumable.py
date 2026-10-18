"""ISW-1000 **可断点续跑**的骨架抽取（沙箱环境专用）。

为什么需要 resumable：
    本机沙箱会杀掉长时间运行的后台进程（实测 multiprocessing.Pool 报 WinError 5；
    10 个 Start-Process 并发也全部被杀；仅**前台短时运行**可存活，约 3s/视频）。
    全量 9914 视频约需 8 小时，无法在一次调用内跑完。
    故改为：每次调用只跑一个「时间预算」内的片段，边跑边存盘；下次调用自动跳过
    已完成的 gloss，反复运行即可逐步收敛到全量。

存盘格式（每个参与者一个 shard）：
    data/isw1000_shards/<Participant>/npz/train.npz
      - skeletons: (n, T, 27, 3) float32
      - glosses:   (n,)  object —— 每个样本的中文 gloss（文件名）
    （与 data/isw1000_merge.py 对接：合并时用 glosses 建立全局类表并映射标签）

用法（每次前台跑约 90 秒，避免超时被杀）：
    python -m data.isw1000_extract_resumable --budget-sec 90
    python -m data.isw1000_extract_resumable --budget-sec 90 --max-glosses 50
    python -m data.isw1000_extract_resumable --status
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from data.isw1000_extract import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_RESIZE_WIDTH,
    DEFAULT_FRAME_STRIDE,
    VIDEO_EXTS,
    discover_samples,
)
from data.wlasl_pose_extract import extract_one_video, KeypointExtractor, _MP_OK, FIXED_T
from tests.contract import check_skeleton_shape

DEFAULT_SHARD_DIR = os.path.join("data", "isw1000_shards")


def shard_path(shard_dir: str, participant: str) -> Path:
    return Path(shard_dir) / participant / "npz" / "train.npz"


def load_shard(p: Path) -> Tuple[List[np.ndarray], List[str]]:
    """读取已有分片，兼容两种历史格式：
    - 新格式（本脚本写）：skeletons + glosses
    - 旧格式（data/isw1000_extract.py 写）：skeletons + labels + classes（本地类表）
    """
    if not p.exists():
        return [], []
    d = np.load(p, allow_pickle=True)
    sk = list(np.asarray(d["skeletons"], dtype=np.float32))
    if "glosses" in d.files:
        gl = [str(g) for g in d["glosses"]]
    else:
        # 旧格式：把本地 labels 反查成本地 classes，再当作 glosses 列表
        classes = [str(c) for c in d["classes"]]
        labels = np.asarray(d["labels"]).astype(int)
        gl = [classes[i] for i in labels]
    return sk, gl


def save_shard(p: Path, skeletons: List[np.ndarray], glosses: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p,
        skeletons=np.stack(skeletons, axis=0).astype(np.float32) if skeletons
        else np.zeros((0, FIXED_T, 27, 3), dtype=np.float32),
        glosses=np.array(glosses, dtype=object),
    )


def run_budget(dataset_root: str = DEFAULT_DATASET_ROOT,
               shard_dir: str = DEFAULT_SHARD_DIR,
               budget_sec: float = 90.0,
               max_glosses: Optional[int] = None,
               participants: Optional[List[str]] = None,
               T: int = FIXED_T,
               resize_width: int = DEFAULT_RESIZE_WIDTH,
               frame_stride: int = DEFAULT_FRAME_STRIDE,
               save_every: int = 10) -> dict:
    """在时间预算内尽可能多地抽取，未完成的留待下次调用。"""
    if not _MP_OK:
        raise RuntimeError("MediaPipe 不可用（需 mediapipe==0.10.14）")

    samples = discover_samples(dataset_root, participants=participants)
    if max_glosses:
        # 只取前 max_glosses 个中文 gloss（按 Unicode 排序），让样本集中在少数类上，
        # 便于在有限视频数下形成可训练的每类样本数
        zh = sorted({s["gloss"] for s in samples if any('\u4e00' <= c <= '\u9fff' for c in s["gloss"])})
        keep = set(zh[:max_glosses])
        samples = [s for s in samples if s["gloss"] in keep]

    # 按参与者分组
    by_p: dict = {}
    for s in samples:
        by_p.setdefault(s["participant"], []).append(s)

    t0 = time.time()
    done_new = 0
    stats: dict = {}

    with KeypointExtractor() as ext:  # type: ignore[arg-type]
        for p, items in sorted(by_p.items()):
            sp = shard_path(shard_dir, p)
            skeletons, glosses = load_shard(sp)
            done = set(glosses)
            pending = [s for s in items if s["gloss"] not in done]
            if not pending:
                stats[p] = {"total": len(glosses), "new": 0, "pending": 0}
                continue

            added = 0
            for s in pending:
                if time.time() - t0 > budget_sec:
                    break
                try:
                    seq = extract_one_video(
                        s["path"], ext, frame_start=None, frame_end=None, T=T,
                        resize_width=resize_width, frame_stride=frame_stride,
                    )
                except Exception:
                    continue
                if seq is None or not check_skeleton_shape(seq):
                    continue
                skeletons.append(seq)
                glosses.append(s["gloss"])
                added += 1
                done_new += 1
                if added % save_every == 0:
                    save_shard(sp, skeletons, glosses)

            save_shard(sp, skeletons, glosses)   # 预算到点也存盘（关键：不丢进度）
            stats[p] = {"total": len(glosses), "new": added,
                        "pending": len(pending) - added}

    return {
        "elapsed_sec": round(time.time() - t0, 1),
        "new_samples": done_new,
        "per_participant": stats,
    }


def status(shard_dir: str = DEFAULT_SHARD_DIR) -> dict:
    out = {}
    for d in sorted(Path(shard_dir).iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        sp = d / "npz" / "train.npz"
        if sp.exists():
            dd = np.load(sp, allow_pickle=True)
            out[d.name] = int(np.asarray(dd["skeletons"]).shape[0])
        else:
            out[d.name] = 0
    out["TOTAL"] = sum(v for k, v in out.items() if k != "TOTAL")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="ISW-1000 可断点续跑抽取")
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--shard-dir", default=DEFAULT_SHARD_DIR)
    ap.add_argument("--budget-sec", type=float, default=90.0)
    ap.add_argument("--max-glosses", type=int, default=None,
                    help="只取前 N 个中文 gloss（让样本集中，便于训练）")
    ap.add_argument("--participants", nargs="*", default=None)
    ap.add_argument("--T", type=int, default=FIXED_T)
    ap.add_argument("--resize-width", type=int, default=DEFAULT_RESIZE_WIDTH)
    ap.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    ap.add_argument("--status", action="store_true", help="只打印各分片进度")
    args = ap.parse_args()

    if args.status:
        print("[status]", status(args.shard_dir))
        return 0

    info = run_budget(
        dataset_root=args.dataset_root,
        shard_dir=args.shard_dir,
        budget_sec=args.budget_sec,
        max_glosses=args.max_glosses,
        participants=args.participants,
        T=args.T,
        resize_width=args.resize_width,
        frame_stride=args.frame_stride,
    )
    print("[resumable]", info)
    print("[status]", status(args.shard_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
