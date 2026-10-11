"""CSL（Chinese Sign Language）真实替代数据集加载器骨架。

CSL 是中文孤立词手语数据集，最接近 ISW-1000，可作为真实训练数据替代。
数据由用户后续提供目录即可一键转换，本文件在**无数据时也能 import 无错**。

依赖 perception.keypoint_extractor（MediaPipe Holistic）抽取 27 点骨架，
落 npz 时接口与合成集完全一致：
    npz 内容: skeletons (N, T, 27, 3) / labels (N,) / classes (K,)

隐私说明：本加载器仅读取 RGB 帧用于关键点提取，不保存任何原始像素，
             仅落 27 点骨架与标注文本，符合隐私最小化。

CSL csv 标注格式假设（请按实际目录调整）：
    - 每行一个孤立词样本：`video_name,gloss`
      video_name 对应 raw_dir 下的 `<video_name>.<ext>`（如 mp4/avi）
      gloss 为该样本的中文手语词（与 classes.txt 域一致）
    - 例：
         0001_你.mp4,你
         0002_我.mp4,我
    - 若某词未出现在 classes.txt，会自动并入类表（不影响合成集契约）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from perception.keypoint_extractor import (
    mediapipe_available,
    KeypointExtractor,
)
from perception.skeleton import normalize_sequence
from tests.contract import NUM_POINTS, SKELETON_RANK

try:
    import cv2
    _CV_OK = True
except Exception:
    cv2 = None
    _CV_OK = False


def _read_annotation_csv(csv_path: str):
    """读取 CSL 标注 csv，返回 [(video_name, gloss), ...]。"""
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, gloss = line.split(",")[:2]
            samples.append((name.strip(), gloss.strip()))
    return samples


def _extract_skeleton_from_video(video_path: str, extractor, fps: int = 30):
    """逐帧抽 27 点骨架并归一化，返回 (T, 27, 3) float32。"""
    if not _CV_OK:
        raise RuntimeError("未安装 opencv-python，无法读取 RGB 帧；请先 pip install opencv-contrib-python")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频：{video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        skel, vis = extractor.extract_frame(frame)
        frames.append(skel)
    cap.release()

    if not frames:
        raise ValueError(f"视频无帧：{video_path}")
    seq = np.stack(frames, axis=0).astype(np.float32)
    norm, _, _ = normalize_sequence(seq)
    return norm.astype(np.float32)


def convert_csl(raw_dir: str, out_dir: str, fps: int = 30,
                annotation_csv: Optional[str] = None) -> dict:
    """把 CSL 原始数据转换为与合成集一致的 npz。

    Args:
        raw_dir:        RGB 帧 / 视频目录
        out_dir:        输出 npz 目录
        fps:            抽帧目标 fps（用于节流）
        annotation_csv: 标注 csv 路径；默认 raw_dir/annotations.csv

    Returns:
        info: dict，含样本数、类数、每切分形状
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"请先提供 CSL 目录：{raw_dir} 不存在。"
            "请将 CSL 原始数据（RGB 视频 + 标注 csv）放入该目录后重试。"
        )
    if not mediapipe_available():
        raise RuntimeError(
            "MediaPipe 不可用，无法抽取骨架。请先 `pip install mediapipe==0.10.14` 再调用。"
        )

    csv_path = Path(annotation_csv) if annotation_csv else raw_dir / "annotations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"请先提供 CSL 标注 csv：{csv_path} 不存在。"
            "标注格式：每行 `video_name,gloss`（详见本文件顶部注释）。"
        )

    samples = _read_annotation_csv(str(csv_path))
    if not samples:
        raise ValueError("标注 csv 为空。")

    classes = sorted({g for _, g in samples})
    class_to_idx = {g: i for i, g in enumerate(classes)}

    skeletons, labels = [], []
    with KeypointExtractor() as ext:
        for name, gloss in samples:
            # 兼容：name 可能带/不带扩展名
            video_path = raw_dir / name
            if not video_path.exists():
                cand = list(raw_dir.glob(f"{name}.*"))
                if not cand:
                    print(f"[csl_loader] 跳过缺失样本：{name}")
                    continue
                video_path = cand[0]
            try:
                seq = _extract_skeleton_from_video(str(video_path), ext, fps=fps)
            except Exception as e:
                print(f"[csl_loader] 抽取失败 {name}：{e}")
                continue
            skeletons.append(seq)
            labels.append(class_to_idx[gloss])

    if not skeletons:
        raise RuntimeError("没有任何样本成功转换。请检查 raw_dir / csv 路径与格式。")

    Tmax = max(s.shape[0] for s in skeletons)
    padded = []
    for s in skeletons:
        if s.shape[0] < Tmax:
            pad = np.broadcast_to(s[-1], (Tmax - s.shape[0], NUM_POINTS, SKELETON_RANK))
            padded.append(np.concatenate([s, pad], axis=0))
        else:
            padded.append(s)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "csl.npz",
        skeletons=np.stack(padded, axis=0).astype(np.float32),
        labels=np.array(labels, dtype=np.int64),
        classes=np.array(classes, dtype=object),
    )
    info = {
        "num_samples": len(padded),
        "num_classes": len(classes),
        "shape": (len(padded), Tmax, NUM_POINTS, SKELETON_RANK),
    }
    print(f"[csl_loader] 已转换 {info['num_samples']} 样本 / {info['num_classes']} 类 -> {out_dir / 'csl.npz'}")
    return info


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python data/csl_loader.py <raw_dir> <out_dir> [fps]")
        print("说明: 需先提供 CSL 原始 rgb 目录 + annotations.csv")
        sys.exit(1)
    raw = sys.argv[1]
    out = sys.argv[2]
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    if not Path(raw).exists():
        print(f"请先提供 CSL 目录：{raw} 不存在。")
        sys.exit(2)
    convert_csl(raw, out, fps=fps)
