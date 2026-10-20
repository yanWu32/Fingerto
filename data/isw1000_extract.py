"""ISW-1000（中文孤立手语词）→ 27 点骨架提取驱动。

数据集结构（本机实测 E:\Program\datasets\ISW-1000）：
    Participant_01..10/
        <gloss>.mp4 | .MOV        # 文件名即 gloss 标签（中文词，如 一/上海/不会）
    - 共 9914 个视频：1000 个 gloss × 10 位参与者（Participant_06 缺 86 个）
    - Participant_09 全为 .MOV，其余为 .mp4；OpenCV 均可解码
    - 竖屏手机拍摄（常见 1080x1920），fps 25~60，帧数约 60~200

与 WLASL 抽取（data/wlasl_pose_extract.py）的差异：
    - WLASL 需要 meta json 提供 frame_start/frame_end（本数据集未提供精确片段）；
      ISW-1000 每个视频就是完整的一个词，无需裁剪，直接整段抽。
    - 标签来自文件名（中文 gloss），无需 meta。

切分策略（避免签名者泄漏）：
    **按参与者划分** train/val/test，而不是随机划分——同一签名者的同一词
    若同时出现在 train 与 test，会造成虚高的"见过该手语者"偏差。
    默认 train=P01-P07, val=P08, test=P09-P10（留 P09 全 MOV 以覆盖 .MOV 解码路径）。

复用：
    - extract_one_video(...) 来自 data/wlasl_pose_extract.py（已修 frame_end=-1 哨兵 bug）
    - 产出格式与 build_dataset 一致：npz/{train,val,test}.npz（skeletons/labels/classes）
      因此 recognition.dataset.WlaslDataset / build_wlasl_dataloader 可直接读。

用法：
    python data/isw1000_extract.py                       # 全量抽取（约 9914 个视频）
    python data/isw1000_extract.py --limit 60            # 冒烟：每参与者最多 60 个
    python data/isw1000_extract.py --participants Participant_01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

# 复用 WLASL 抽取管线中已验证的单视频抽取函数与契约常量
from data.wlasl_pose_extract import (
    extract_one_video,
    KeypointExtractor,
    _MP_OK,
    _CV_OK,
    FIXED_T,
)
from tests.contract import check_skeleton_shape, NUM_POINTS, SKELETON_RANK

VIDEO_EXTS = (".mp4", ".mov", ".MOV", ".MP4")

# 默认按参与者切分（避免签名者泄漏）
DEFAULT_SPLIT_BY_PARTICIPANT: Dict[str, List[str]] = {
    "train": ["Participant_01", "Participant_02", "Participant_03", "Participant_04",
              "Participant_05", "Participant_06", "Participant_07"],
    "val":   ["Participant_08"],
    "test":  ["Participant_09", "Participant_10"],
}

DEFAULT_DATASET_ROOT = r"E:\Program\datasets\ISW-1000"
DEFAULT_OUT_DIR = os.path.join("data", "isw1000_real")

# 性能：ISW-1000 多为竖屏 1080x1920 手机视频，逐帧跑 MediaPipe Holistic 极慢
# （实测 100 个视频 >5min）。默认先等比缩到 640 宽 + 每 2 帧抽 1 帧，可大幅提速，
# 且骨架为归一化坐标，缩放/抽帧不影响 (T,27,3) 契约与后续训练。
DEFAULT_RESIZE_WIDTH = 640
DEFAULT_FRAME_STRIDE = 2


def discover_samples(dataset_root: str,
                     participants: Optional[List[str]] = None,
                     limit_per_participant: Optional[int] = None) -> List[dict]:
    """扫描数据集，返回样本列表 [{"path", "gloss", "participant"}]。

    gloss 直接取文件名（不含扩展名）—— ISW-1000 的命名约定。
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"未找到数据集目录：{root}")

    pdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if participants:
        pdirs = [d for d in pdirs if d.name in participants]

    samples: List[dict] = []
    for pdir in pdirs:
        vids = [p for p in pdir.iterdir() if p.is_file() and p.suffix in VIDEO_EXTS]
        vids.sort(key=lambda x: x.name)
        if limit_per_participant:
            vids = vids[:limit_per_participant]
        for v in vids:
            samples.append({
                "path": str(v),
                "gloss": v.stem,          # 文件名即中文 gloss
                "participant": pdir.name,
            })
    return samples


def build_isw1000(dataset_root: str = DEFAULT_DATASET_ROOT,
                  out_dir: str = DEFAULT_OUT_DIR,
                  T: int = FIXED_T,
                  split_map: Optional[Dict[str, List[str]]] = None,
                  limit_per_participant: Optional[int] = None,
                  skip_bad: bool = True,
                  resize_width: Optional[int] = DEFAULT_RESIZE_WIDTH,
                  frame_stride: int = DEFAULT_FRAME_STRIDE) -> dict:
    """抽取 ISW-1000 骨架并按参与者切分落盘。

    Returns:
        info: dict，含各切分样本数 / 类数 / 跳过清单
    """
    if not _MP_OK:
        raise RuntimeError("MediaPipe 不可用，无法抽取骨架（需 mediapipe==0.10.14）")
    if not _CV_OK:
        raise RuntimeError("未安装 opencv-python，无法读帧")

    split_map = split_map or DEFAULT_SPLIT_BY_PARTICIPANT
    samples = discover_samples(dataset_root,
                               participants=None,
                               limit_per_participant=limit_per_participant)
    if not samples:
        raise RuntimeError(f"未在 {dataset_root} 发现任何视频文件")

    # 全局类表（按所有样本排序，保证切分间标签一致）
    classes = sorted({s["gloss"] for s in samples})
    class_to_idx = {g: i for i, g in enumerate(classes)}

    out = Path(out_dir)
    (out / "npz").mkdir(parents=True, exist_ok=True)

    info: dict = {"dataset": "ISW-1000", "num_classes": len(classes), "splits": {}, "skipped": []}
    skipped: List[str] = []

    with KeypointExtractor() as ext:  # type: ignore[arg-type]
        for split, plist in split_map.items():
            split_samples = [s for s in samples if s["participant"] in plist]
            skeletons, labels = [], []
            for s in split_samples:
                try:
                    # ISW-1000 每个视频即完整的词，直接整段抽（frame_start/end 均 None）
                    # 用 resize_width + frame_stride 提速（骨架为归一化坐标，不影响契约）
                    seq = extract_one_video(
                        s["path"], ext,
                        frame_start=None, frame_end=None, T=T,
                        resize_width=resize_width, frame_stride=frame_stride,
                    )
                except Exception as e:  # 单样本失败不阻塞整批
                    if skip_bad:
                        skipped.append(f"{s['path']}:{type(e).__name__}:{e}")
                        continue
                    raise
                if seq is None or not check_skeleton_shape(seq):
                    if skip_bad:
                        skipped.append(f"{s['path']}:bad_or_contract_fail")
                        continue
                    raise RuntimeError(f"样本抽取失败或未通过契约：{s['path']}")

                skeletons.append(seq)
                labels.append(class_to_idx[s["gloss"]])

            if not skeletons:
                info["splits"][split] = {"num_samples": 0, "shape": None}
                continue
            np.savez(
                out / "npz" / f"{split}.npz",
                skeletons=np.stack(skeletons, axis=0).astype(np.float32),
                labels=np.array(labels, dtype=np.int64),
                classes=np.array(classes, dtype=object),
            )
            info["splits"][split] = {
                "num_samples": len(skeletons),
                "shape": (len(skeletons), T, NUM_POINTS, SKELETON_RANK),
                "participants": plist,
            }
            print(f"[isw1000] {split}: {len(skeletons)} 样本 ({plist})")

    # gloss 列表独立保存（中文，与英文 wlasl glosses.txt 严格隔离）
    (out / "glosses.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    info["skipped"] = skipped
    info["glosses_file"] = str(out / "glosses.txt")
    print(f"[isw1000] 类数={len(classes)}  "
          f"各切分={ {k: v['num_samples'] for k, v in info['splits'].items()} }  "
          f"跳过={len(skipped)}")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="ISW-1000 → 27 点骨架提取")
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    ap.add_argument("--T", type=int, default=FIXED_T)
    ap.add_argument("--limit", type=int, default=None,
                    help="每个参与者最多抽取 N 个视频（冒烟用）")
    ap.add_argument("--participants", nargs="*", default=None,
                    help="只处理指定参与者（默认全部）")
    ap.add_argument("--resize-width", type=int, default=DEFAULT_RESIZE_WIDTH,
                    help="送 MediaPipe 前把帧等比缩到该宽度（0=不缩放）")
    ap.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE,
                    help="帧采样步长（1=每帧处理）")
    args = ap.parse_args()

    split_map = DEFAULT_SPLIT_BY_PARTICIPANT
    if args.participants:
        # 单参与者模式：全部归入 train，便于冒烟
        split_map = {"train": args.participants}

    info = build_isw1000(
        dataset_root=args.dataset_root,
        out_dir=args.out,
        T=args.T,
        split_map=split_map,
        limit_per_participant=args.limit,
        resize_width=(args.resize_width or None),
        frame_stride=args.frame_stride,
    )
    print("[isw1000] 完成：", json.dumps(
        {k: v for k, v in info.items() if k != "skipped"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
