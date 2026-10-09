"""MediaPipe 关键点提取 —— 27 点骨架。

使用 MediaPipe Holistic 一次推理同时得到 左手 / 右手 / 上身姿态，
再组装成 27 点骨架：(T, 27, 3) + (T, 27) 可见性掩码。

依赖：mediapipe>=0.10（mp.solutions.holistic）。
"""
from __future__ import annotations

import numpy as np

try:
    import mediapipe as mp
    from mediapipe.python.solutions import holistic as mp_holistic
    _MP_OK = True
except Exception as _e:  # pragma: no cover
    mp = None
    mp_holistic = None
    _MP_OK = False
    _MP_ERR = _e

from .skeleton import POSE_LANDMARK_IDX, NUM_POINTS, HAND_POINTS, POSE_POINTS


def mediapipe_available() -> bool:
    return _MP_OK


class KeypointExtractor:
    """MediaPipe Holistic 封装，逐帧返回 27 点骨架。"""

    def __init__(self, config: dict | None = None, model_complexity: int = 1):
        if not _MP_OK:
            raise RuntimeError(
                f"MediaPipe 不可用：{_MP_ERR}。请安装 mediapipe>=0.10 后再调用。"
            )
        cfg = config or {}
        mp_cfg = cfg.get("mediapipe", {}) if isinstance(cfg, dict) else {}
        self.min_det = mp_cfg.get("min_detection_confidence", 0.5)
        self.min_track = mp_cfg.get("min_tracking_confidence", 0.5)
        self.max_hands = mp_cfg.get("max_num_hands", 2)
        self.swap = mp_cfg.get("swap_handedness", True)
        self.complexity = mp_cfg.get("model_complexity", model_complexity)

        self._mp = mp
        self._holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.complexity,
            enable_segmentation=False,
            min_detection_confidence=self.min_det,
            min_tracking_confidence=self.min_track,
            refine_face_landmarks=False,
        )

    def extract_frame(self, image_bgr: np.ndarray):
        """处理单帧 BGR 图像。

        Returns:
            skeleton: (27, 3) float32，未检测到的点置 0
            vis:      (27,) float32，0/1 可见性
        """
        mp = self._mp
        img_rgb = image_bgr[..., ::-1]  # BGR -> RGB
        res = self._holistic.process(img_rgb)

        skeleton = np.zeros((NUM_POINTS, 3), dtype=np.float32)
        vis = np.zeros((NUM_POINTS,), dtype=np.float32)

        # ---- 手部 21 点（主手优先左手，否则右手）----
        hand = None
        if res.left_hand_landmarks is not None:
            hand = res.left_hand_landmarks
            # 镜像采集下 MediaPipe 的 "Left" 实为画面右侧，即用户右手
            if self.swap:
                hand = res.right_hand_landmarks or hand
        elif res.right_hand_landmarks is not None:
            hand = res.right_hand_landmarks

        if hand is not None:
            for i in range(HAND_POINTS):
                lm = hand.landmark[i]
                skeleton[i, 0] = lm.x
                skeleton[i, 1] = lm.y
                skeleton[i, 2] = lm.z
                vis[i] = 1.0

        # ---- 上身 6 点（Pose 子集）----
        if res.pose_landmarks is not None:
            for j, idx in enumerate(POSE_LANDMARK_IDX):
                lm = res.pose_landmarks.landmark[idx]
                o = HAND_POINTS + j
                skeleton[o, 0] = lm.x
                skeleton[o, 1] = lm.y
                skeleton[o, 2] = lm.z
                vis[o] = 1.0

        return skeleton, vis

    def close(self):
        try:
            self._holistic.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
