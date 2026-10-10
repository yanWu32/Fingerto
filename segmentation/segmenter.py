"""运动能量粗切分 + 滑动窗口细识别。

输入：感知层归一化后的骨骼序列 ``(T, 27, 3)``
  - 索引 0-20 : 主手 21 点（MediaPipe Hands 标准，wrist = 0）
  - 索引 21-26: 上半身 6 点（Pose 子集：鼻/左肩/右肩/左肘/右肘/左腕）

输出：
  - ``segment()``：粗切分得到的手势单元列表（每段含时间/帧边界/子序列）。
  - 滑动窗口：在活跃段上以 (window, stride) 滑窗，产出细识别窗口 ``(w, 27, 3)``。

所有输入/输出的骨骼张量均通过 ``tests.contract.check_skeleton_shape`` 校验
（形状正确、数值有限、无 NaN/Inf）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 契约校验：唯一事实源 tests/contract.py
# ---------------------------------------------------------------------------
try:  # 标准运行环境（Fingerto 为根，tests 为包）
    from tests.contract import check_skeleton_shape
except ModuleNotFoundError:  # 防御：保证契约可被导入
    import os
    import sys

    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from tests.contract import check_skeleton_shape


# ---------------------------------------------------------------------------
# 配置加载（缺失给合理默认）
# ---------------------------------------------------------------------------
def _load_cfg(name: str = "segmentation", overrides: Optional[Dict[str, Any]] = None):
    """加载 configs/segmentation.yaml；失败则用内建默认。"""
    try:
        from configs import load_config

        cfg = load_config(name, overrides)
    except Exception:
        cfg = _DefaultConfig()
    return cfg


class _DefaultConfig:
    """load_config 不可用时的兜底容器，API 与 Config 对齐。"""

    def __init__(self) -> None:
        self._data = DEFAULTS

    def get(self, path: str, default: Any = None) -> Any:
        cur: Any = self._data
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default
        return cur


DEFAULTS: Dict[str, Any] = {
    "energy": {
        # 关键关节（手腕）用于运动能量；映射见 _REFERENCE_NAME_TO_IDX
        "reference_points": ["left_wrist", "right_wrist"],
        "velocity_threshold": 0.02,    # 归一化坐标/帧 的绝对下限
        "min_static_frames": 5,        # 静默持续帧数 -> 判定为单元边界
        "min_segment_frames": 8,       # 短于此的片段视为噪声丢弃
        "max_segment_frames": 120,     # 长于此强制截断
        "smoothing_window": 3,         # 速度序列平滑窗口
        "adaptive_k": 3.0,             # 自适应阈值倍数
        "adaptive_method": "mad",      # mad(鲁棒) | std
    },
    "window": {
        "size": 30,
        "stride": 10,
        "smoothing": "vote",
        "vote_threshold": 0.5,
    },
    "output": {
        "min_confidence": 0.3,
        "merge_gap": 3,
    },
}


# (T,27,3) 布局下的关键点名 -> 索引。
# 由于 27 点仅含单手 21 点 + 上半身 6 点，"双手腕"映射为：
#   手部腕(索引 0) 与 上半身左腕(索引 26，POSE 子集里的左腕)。
_REFERENCE_NAME_TO_IDX = {
    "hand_wrist": 0, "wrist": 0, "left_wrist": 0, "right_wrist": 26,
    "nose": 21, "left_shoulder": 22, "right_shoulder": 23,
    "left_elbow": 24, "right_elbow": 25, "pose_left_wrist": 26,
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """一个粗切分得到的手势单元。"""

    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    sub_sequence: np.ndarray            # (t', 27, 3)，有限且形状合法
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.sub_sequence.shape[0])

    def windows(self, size: int, stride: int) -> Iterator[np.ndarray]:
        """在该子序列上滑窗，产出 ``(w, 27, 3)`` 窗口。

        仅产出完整窗口；若子序列短于窗口，则把整段作为单个窗口产出，
        保证每个活跃段至少产生一个待识别窗口。
        """
        seq = self.sub_sequence
        L = seq.shape[0]
        if L <= size:
            yield seq
            return
        start = 0
        while start + size <= L:
            yield seq[start:start + size]
            start += stride


# ---------------------------------------------------------------------------
# 切分器
# ---------------------------------------------------------------------------
class SkeletonSegmenter:
    """运动能量粗切分 + 滑动窗口细识别。

    用法::

        seg = SkeletonSegmenter()                 # 自动读 configs/segmentation.yaml
        segments = seg.segment(skeleton, fps=30)  # List[Segment]
        for s in segments:
            for w in seg.sliding_windows(s):      # 细识别窗口 (w,27,3)
                ...
    """

    def __init__(
        self,
        config: Any = None,
        cfg_name: str = "segmentation",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """``config`` 可为 Config 对象 / 配置名字符串 / None（加载默认名）。"""
        if config is None:
            cfg = _load_cfg(cfg_name, overrides)
        elif isinstance(config, str):
            cfg = _load_cfg(config, overrides)
        else:
            cfg = config

        self.cfg = cfg

        e = cfg.get("energy", {})
        self.reference_points = cfg.get("energy.reference_points", ["left_wrist", "right_wrist"])
        self.velocity_threshold = float(cfg.get("energy.velocity_threshold", 0.02))
        self.min_static_frames = int(cfg.get("energy.min_static_frames", 5))
        self.min_segment_frames = int(cfg.get("energy.min_segment_frames", 8))
        self.max_segment_frames = int(cfg.get("energy.max_segment_frames", 120))
        self.smoothing_window = int(cfg.get("energy.smoothing_window", 3))
        self.adaptive_k = float(cfg.get("energy.adaptive_k", 3.0))
        self.adaptive_method = str(cfg.get("energy.adaptive_method", "mad"))

        w = cfg.get("window", {})
        self.window_size = int(cfg.get("window.size", 30))
        self.window_stride = int(cfg.get("window.stride", 10))
        self.window_smoothing = str(cfg.get("window.smoothing", "vote"))
        self.vote_threshold = float(cfg.get("window.vote_threshold", 0.5))

        o = cfg.get("output", {})
        self.min_confidence = float(cfg.get("output.min_confidence", 0.3))
        self.merge_gap = int(cfg.get("output.merge_gap", 3))

        # 解析参考关节索引
        self.reference_indices = self._resolve_reference_indices(self.reference_points)

    # ----------------------------- 工具 -----------------------------
    @staticmethod
    def _resolve_reference_indices(points) -> List[int]:
        """把配置里的 reference_points（名称或整数索引）解析成 0-26 的整数列表。"""
        idx: List[int] = []
        for p in points:
            if isinstance(p, int):
                if 0 <= p < 27:
                    idx.append(p)
            else:
                name = str(p).strip().lower()
                if name in _REFERENCE_NAME_TO_IDX:
                    idx.append(_REFERENCE_NAME_TO_IDX[name])
        # 去重并保持顺序；若解析为空则退回 [0, 26]
        seen = set()
        uniq = [i for i in idx if not (i in seen or seen.add(i))]
        return uniq if uniq else [0, 26]

    # ----------------------------- 主流程 -----------------------------
    def segment(self, skeleton: Any, fps: float = 30.0) -> List[Segment]:
        """对骨骼序列做粗切分，返回手势单元列表。

        Args:
            skeleton: ``(T, 27, 3)`` 有限数值（list / np.ndarray 均可）。
            fps: 采样率，用于换算时间戳。

        Returns:
            List[Segment]，按时间顺序排列；无运动时返回空列表。
        """
        seq = np.asarray(skeleton, dtype=np.float32)
        if not check_skeleton_shape(seq):
            raise ValueError(
                "[segmentation] 输入骨骼序列不符合 (T,27,3) 契约：须为有限数值且形状正确"
            )
        if seq.shape[0] < 2:
            return []

        fps = float(fps)
        if fps <= 0:
            fps = 30.0

        energy = self._motion_energy(seq)
        active = self._detect_active(energy)

        runs = self._merge_runs(active)
        segments: List[Segment] = []
        for (s, e) in runs:
            sub = seq[s:e + 1]
            if not check_skeleton_shape(sub):
                # 子序列也应有限且形状合法（理论上恒成立，这里做防御性校验）
                continue
            segments.append(
                Segment(
                    start_frame=int(s),
                    end_frame=int(e),
                    start_time=float(s) / fps,
                    end_time=float(e) / fps,
                    sub_sequence=sub,
                    meta={"energy_peak": float(energy[s:e + 1].max())},
                )
            )
        return segments

    # ----------------------------- 能量 -----------------------------
    def _motion_energy(self, seq: np.ndarray) -> np.ndarray:
        """逐帧运动能量 = 参考关节点速度范数之和。返回长度 T 的数组。"""
        idx = self.reference_indices
        pts = seq[:, idx, :]                       # (T, K, 3)
        vel = np.linalg.norm(np.diff(pts, axis=0), axis=-1)  # (T-1, K)
        per_frame = vel.sum(axis=1)                # (T-1,)

        # 平滑（移动平均）
        if self.smoothing_window and self.smoothing_window > 1:
            per_frame = self._moving_average(per_frame, self.smoothing_window)

        # 对齐到帧：第 0 帧能量设为第 1 帧值（无前置帧）
        energy = np.empty(seq.shape[0], dtype=np.float32)
        energy[0] = per_frame[0] if per_frame.size else 0.0
        energy[1:] = per_frame
        return energy

    @staticmethod
    def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
        window = int(window)
        if window <= 1 or x.size == 0:
            return x
        k = window // 2
        out = np.empty_like(x, dtype=np.float32)
        for i in range(x.size):
            lo, hi = max(0, i - k), min(x.size, i + k + 1)
            out[i] = x[lo:hi].mean()
        return out

    def _detect_active(self, energy: np.ndarray) -> np.ndarray:
        """自适应阈值检测活跃帧。"""
        med = float(np.median(energy))
        if self.adaptive_method == "std":
            scale = float(np.std(energy))
        else:  # mad（对稀疏运动鲁棒）
            scale = 1.4826 * float(np.median(np.abs(energy - med)))
        thr = max(self.velocity_threshold, med + self.adaptive_k * scale)
        return energy > thr

    # ----------------------------- 合并 -----------------------------
    def _merge_runs(self, active: np.ndarray) -> List[Sequence[int]]:
        """把活跃帧合并成片段：

        - 相邻活跃段若间隔（静默）小于 ``min_static_frames`` 则合并；
        - 长度小于 ``min_segment_frames`` 的丢弃（噪声）；
        - 长度超过 ``max_segment_frames`` 强制截断为多段。
        """
        n = active.shape[0]
        # 1) 原始活跃段
        raw: List[List[int]] = []
        s = None
        for i in range(n):
            if active[i] and s is None:
                s = i
            elif not active[i] and s is not None:
                raw.append([s, i - 1])
                s = None
        if s is not None:
            raw.append([s, n - 1])

        if not raw:
            return []

        # 2) 短静默间隙合并
        merged: List[List[int]] = [raw[0]]
        for cur in raw[1:]:
            prev = merged[-1]
            gap = cur[0] - prev[1] - 1          # 之间的静默帧数
            if 0 <= gap < self.min_static_frames:
                prev[1] = cur[1]                # 合并
            else:
                merged.append(cur)

        # 3) 丢弃过短片段 + 4) 超长截断
        out: List[Sequence[int]] = []
        for (a, b) in merged:
            length = b - a + 1
            if length < self.min_segment_frames:
                continue
            if length <= self.max_segment_frames:
                out.append((a, b))
            else:
                # 以 max_segment_frames 为步长强制切分
                start = a
                while start <= b:
                    end = min(start + self.max_segment_frames - 1, b)
                    out.append((start, end))
                    start = end + 1
        return out

    # ----------------------------- 滑动窗口 -----------------------------
    def sliding_windows(self, segment: Segment) -> Iterator[np.ndarray]:
        """产出某活跃段的细识别窗口 ``(w, 27, 3)``。"""
        yield from segment.windows(self.window_size, self.window_stride)

    def iter_windows(self, segments: Sequence[Segment]) -> Iterator[np.ndarray]:
        """叠加所有活跃段的滑动窗口。"""
        for seg in segments:
            yield from self.sliding_windows(seg)


__all__ = ["SkeletonSegmenter", "Segment"]
