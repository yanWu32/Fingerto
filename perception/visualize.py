"""骨架可视化 —— 把 27 点骨架画到 BGR 帧上（调试用）。"""
from __future__ import annotations

import numpy as np

from .skeleton import HAND_EDGES, POSE_EDGES, HAND_OFFSET, POSE_OFFSET


def _to_xy(seq_frame, w, h):
    return (
        (seq_frame[:, 0] * w).astype(int),
        (seq_frame[:, 1] * h).astype(int),
    )


def draw_skeleton(image_bgr: np.ndarray, seq_frame: np.ndarray,
                  vis=None, color=(0, 255, 0)):
    """在图像上绘制 27 点骨架。

    seq_frame: (27, 3) 归一化后的骨架（已还原到像素需传入未归一化坐标）；
               这里约定传入的是归一化坐标时会按比例还原到帧尺寸。
    """
    h, w = image_bgr.shape[:2]
    pts = seq_frame
    xs, ys = (pts[:, 0] * w).astype(int), (pts[:, 1] * h).astype(int)

    if vis is not None:
        mask = vis > 0.5
    else:
        mask = np.abs(pts).sum(axis=-1) > 1e-6

    # 手骨连接
    for a, b in HAND_EDGES:
        if mask[a] and mask[b]:
            cv_line(image_bgr, xs[a], ys[a], xs[b], ys[b], color, 2)
    # 上身连接
    for a, b in POSE_EDGES:
        pa, pb = a + POSE_OFFSET, b + POSE_OFFSET
        if mask[pa] and mask[pb]:
            cv_line(image_bgr, xs[pa], ys[pa], xs[pb], ys[pb], (255, 0, 0), 2)
    # 关节点
    for i in range(pts.shape[0]):
        if mask[i]:
            cv_circle(image_bgr, xs[i], ys[i], 3, (0, 0, 255), -1)
    return image_bgr


def cv_line(img, x1, y1, x2, y2, color, thick):
    import cv2
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)


def cv_circle(img, x, y, r, color, fill):
    import cv2
    cv2.circle(img, (int(x), int(y)), r, color, fill)
