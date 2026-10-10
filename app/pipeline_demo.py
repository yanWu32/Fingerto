"""应用层：无 GUI 的端到端串联入口。

数据流（按契约调用，不复制各层逻辑）：
    感知(可选 extract) -> 切分(segmentation.SkeletonSegmenter)
        -> 识别(predictor / recognition.predict.predict)
        -> 智能体(agent.run_agent)

设计要点：
- ``run_demo`` 是唯一的串联入口，返回同时含 ``gloss_json`` 与 ``agent_json``。
- 识别器做成**可注入**的（``predictor=`` 回调），默认才用
  ``recognition.predict.predict``。本机无 torch，因此真实模型路径只在
  真正需要时才懒加载 ``recognition`` 包；测试注入假识别器即可走通全链路。
- 所有层输出都经 ``tests/contract`` 校验（合法返回 True，失败抛 ValueError）。
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 契约校验（唯一事实源）：合法返回 True，失败抛 ValueError
# ---------------------------------------------------------------------------
from tests.contract import validate_agent_output, validate_gloss_output

# 切分层（仅依赖 numpy，无重依赖）
from segmentation.segmenter import SkeletonSegmenter

# 智能体层（无 torch 依赖）
from agent import run_agent


# ---------------------------------------------------------------------------
# 默认识别器（懒加载，仅在真实推理时需要 torch）
# ---------------------------------------------------------------------------
def _default_recognition_predictor(window: np.ndarray, fps: float = 30.0, **_kw) -> Dict[str, Any]:
    """默认识别器：委托 ``recognition.predict.predict`` 对单窗口 (t',27,3) 推理。

    注意：``recognition`` 包在 ``__init__`` 里会导入 torch，因此这里必须懒加载，
    保证无 torch 环境下 ``import app.pipeline_demo`` 与注入式测试不会崩。
    """
    from recognition.predict import predict

    return predict(np.asarray(window, dtype=np.float32), fps=float(fps))


# ---------------------------------------------------------------------------
# 窗口聚合：把多个窗口的 gloss JSON 合并成一段 gloss 序列
# ---------------------------------------------------------------------------
def _aggregate_window_results(window_results: Sequence[Dict[str, Any]],
                              seg_start_times: Sequence[float]) -> Dict[str, Any]:
    """把若干单窗口 gloss JSON 拼接成完整 gloss JSON。

    每个窗口的时间戳相对窗口起点，这里按所属段的起点 ``seg_start_times`` 整体平移。
    返回经契约合法的 gloss JSON。
    """
    gloss_sequence: List[str] = []
    confidence: List[float] = []
    timestamps: List[List[float]] = []
    for res, t0 in zip(window_results, seg_start_times):
        gs = list(res.get("gloss_sequence", []))
        cf = list(res.get("confidence", []))
        ts = list(res.get("timestamps", []))
        for g, c, (s, e) in zip(gs, cf, ts):
            gloss_sequence.append(g)
            confidence.append(float(c))
            timestamps.append([float(s) + t0, float(e) + t0])
    out = {
        "gloss_sequence": gloss_sequence,
        "confidence": confidence,
        "timestamps": timestamps,
    }
    validate_gloss_output(out)  # 契约闸门
    return out


# ---------------------------------------------------------------------------
# 串联入口
# ---------------------------------------------------------------------------
def run_demo(
    skeleton: Any = None,
    gloss_json: Optional[Dict[str, Any]] = None,
    predictor: Optional[Callable[..., Dict[str, Any]]] = None,
    segmenter: Optional[SkeletonSegmenter] = None,
    fps: float = 30.0,
    cfg: Optional[Dict[str, Any]] = None,
    state: Any = None,
    fake_completion: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """端到端串联：感知(可选) -> 切分 -> 识别 -> 智能体。

    两种驱动方式（二选一）：
    1) 传 ``skeleton`` ``(T,27,3)``：先做切分，再对每个窗口调 ``predictor``，
       聚合得到 gloss JSON。
    2) 直接传 ``gloss_json``：跳过切分/识别，直达智能体。

    Args:
        skeleton: ``(T,27,3)`` 骨骼序列（list / np.ndarray）。
        gloss_json: 直接提供的识别层 gloss JSON（优先于 skeleton 的切分路径）。
        predictor: 识别器回调，签名 ``predictor(window, fps=...) -> gloss JSON``。
            默认使用 ``recognition.predict.predict``（需 torch）。
        segmenter: 自定义切分器；默认 ``SkeletonSegmenter()``。
        fps: 采样率，仅用于时间戳换算与窗口产出。
        cfg: 智能体配置（Config 或 dict）；None 时由 run_agent 自动 load_config。
        state: 对话状态对象；None 时由 run_agent 新建。
        fake_completion: 无 API Key 时必须传入的假 LLM 回调。

    Returns:
        dict: {
          "gloss_json": ..., "agent_json": ...,
          "num_segments": int, "num_windows": int,
          "processing_fps": float, "timing": {...}, "ok": True
        }
    """
    t0 = time.perf_counter()

    # ---- 1) 得到 gloss JSON ----
    if gloss_json is not None:
        # 直接提供识别结果，跳过切分/识别
        candidate_gloss = dict(gloss_json)
        num_segments = 0
        num_windows = len(candidate_gloss.get("gloss_sequence", []))
    elif skeleton is not None:
        seq = np.asarray(skeleton, dtype=np.float32)
        if segmenter is None:
            segmenter = SkeletonSegmenter()

        segments = segmenter.segment(seq, fps=fps)

        pred = predictor if predictor is not None else _default_recognition_predictor

        window_results: List[Dict[str, Any]] = []
        start_times: List[float] = []
        for seg in segments:
            for w in segmenter.sliding_windows(seg):
                window_results.append(pred(w, fps=fps))
                start_times.append(seg.start_time)

        if not window_results:
            # 切分未产出活跃段（如完全静止）：把整段当作单个窗口兜底
            window_results.append(pred(seq, fps=fps))
            start_times.append(0.0)

        candidate_gloss = _aggregate_window_results(window_results, start_times)
        num_segments = len(segments)
        num_windows = len(window_results)
    else:
        raise ValueError("run_demo 必须提供 skeleton 或 gloss_json 之一")

    # 契约闸门：识别层输出必须合法
    validate_gloss_output(candidate_gloss)

    # ---- 2) 智能体 ----
    agent_json = run_agent(
        candidate_gloss, state=state, cfg=cfg, fake_completion=fake_completion
    )
    validate_agent_output(agent_json)  # 防御性再校验

    elapsed = time.perf_counter() - t0
    n_frames = int(seq.shape[0]) if skeleton is not None else 0
    processing_fps = (n_frames / elapsed) if (elapsed > 0 and n_frames > 0) else 0.0

    return {
        "gloss_json": candidate_gloss,
        "agent_json": agent_json,
        "num_segments": num_segments,
        "num_windows": num_windows,
        "processing_fps": float(processing_fps),
        "timing": {
            "total_seconds": float(elapsed),
            "num_frames": n_frames,
        },
        "ok": True,
    }


# ---------------------------------------------------------------------------
# 便捷：从可选感知源（摄像头/视频）抽骨架后跑全链路
# ---------------------------------------------------------------------------
def run_demo_from_source(
    source: Any,
    predictor: Optional[Callable[..., Dict[str, Any]]] = None,
    fps: float = 30.0,
    cfg: Optional[Dict[str, Any]] = None,
    fake_completion: Optional[Callable[[str], str]] = None,
    max_frames: Optional[int] = None,
    smooth: bool = True,
):
    """从摄像头/视频抽 27 点骨架，再走 run_demo。

    依赖 mediapipe + opencv，缺二者时抛 RuntimeError（由调用方捕获降级）。
    """
    from perception.extract import extract_video
    from configs import load_config

    pcfg = load_config("perception")
    result = extract_video(
        source, pcfg, smooth=smooth, max_frames=max_frames, display=False
    )
    skeleton = result["skeleton"]
    return run_demo(
        skeleton=skeleton, predictor=predictor, fps=fps, cfg=cfg,
        fake_completion=fake_completion,
    )


__all__ = ["run_demo", "run_demo_from_source", "_default_recognition_predictor"]
