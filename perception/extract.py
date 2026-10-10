"""骨架序列提取管线。

支持：摄像头实时 / 本地视频文件 -> 27 点骨架序列 (.npy)。

输出结构（npz）：
    skeleton : (T, 27, 3) float32  归一化后坐标
    vis      : (T, 27)    float32  可见性
    raw      : (T, 27, 3) float32  原始归一化前坐标（可选保留）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cv2

from configs import load_config
from .keypoint_extractor import KeypointExtractor, mediapipe_available
from .skeleton import normalize_sequence, smooth_sequence


def extract_video(source, config, smooth=True, max_frames=None,
                   display=False, out_path=None):
    """从摄像头或视频文件提取骨架序列。

    Args:
        source: int（摄像头索引）或视频文件路径
        config: Config 对象
        smooth: 是否做移动平均平滑
        max_frames: 最多处理帧数（None=全部）
        display: 是否实时显示
        out_path: 输出 .npz 路径（None=不保存）

    Returns:
        dict: {"skeleton":(T,27,3), "vis":(T,27), "raw":(T,27,3), "meta":dict}
    """
    if not mediapipe_available():
        raise RuntimeError("MediaPipe 不可用，无法提取骨架。")

    pcfg = config.to_dict().get("perception", {})
    cap = cv2.VideoCapture(int(source) if isinstance(source, int) else str(source))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")

    try:
        with KeypointExtractor(config.to_dict().get("perception")) as ex:
            skeletons, viss, raws = [], [], []
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                sk, v = ex.extract_frame(frame)
                skeletons.append(sk)
                viss.append(v)
                raws.append(sk.copy())
                if display:
                    vis_frame = frame.copy()
                    from .visualize import draw_skeleton
                    draw_skeleton(vis_frame, sk, v)
                    cv2.imshow("sign skeleton", vis_frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break
                idx += 1
                if max_frames and idx >= max_frames:
                    break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

    if not skeletons:
        raise RuntimeError("未提取到任何帧。")

    raw_arr = np.stack(raws, axis=0)
    vis_arr = np.stack(viss, axis=0)
    norm, vis_out, kept = normalize_sequence(raw_arr, vis_arr,
                                             config.get("perception.skeleton.max_missing_ratio", 0.3))
    if smooth:
        norm = smooth_sequence(norm, vis_out)

    meta = {
        "num_points": int(norm.shape[1]),
        "frames": int(norm.shape[0]),
        "source": str(source),
        "mediapipe_mode": pcfg.get("mediapipe", {}).get("mode", "holistic"),
    }
    result = {"skeleton": norm, "vis": vis_out, "raw": raw_arr[kept], "meta": meta}

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            skeleton=norm.astype(np.float32),
            vis=vis_out.astype(np.float32),
            raw=raw_arr[kept].astype(np.float32),
        )
        out.with_suffix(".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[perception] 已保存 {norm.shape} -> {out}")
    return result


def main():
    ap = argparse.ArgumentParser(description="提取 27 点骨架序列")
    ap.add_argument("--source", default=0,
                    help="摄像头索引(整数) 或 视频文件路径")
    ap.add_argument("--out", default="data/processed/sample.npz",
                    help="输出 .npz 路径")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--config", default="perception")
    args = ap.parse_args()

    cfg = load_config(args.config)
    src = int(args.source) if str(args.source).isdigit() else args.source
    extract_video(
        src, cfg,
        smooth=not args.no_smooth,
        max_frames=args.max_frames,
        display=args.display,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
