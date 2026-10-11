"""应用层：Gradio 端到端界面（懒加载 / 可选）。

展示：摄像头/视频输入、关键点可视化、识别 gloss、翻译文本、对话历史、FPS。
场景：信息查询（天气/时间）+ 模拟智能家居控制。

重要：本机可能未安装 gradio / mediapipe / opencv。这些依赖一律**懒加载**，
``import app.app`` 不会因缺失而崩；调用 ``build_ui()`` / ``launch()`` 时才真正 import，
缺失时给出清晰报错。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

# 核心串联逻辑（无 GUI、无重依赖）
from .pipeline_demo import run_demo, run_demo_from_source


def gradio_available() -> bool:
    """检测 gradio 是否可用（不抛异常）。"""
    try:
        import gradio  # noqa: F401
        return True
    except Exception:
        return False


def _draw_skeleton(skeleton: Any) -> Optional[Any]:
    """把 (T,27,3) 序列的最后一帧画成关键点图（需要 opencv/numpy）。

    缺失 opencv 时返回 None，由界面降级为纯文本。
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    seq = np.asarray(skeleton, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[0] == 0:
        return None
    pts = seq[-1]  # (27,3)
    # 归一化坐标 -> 像素（简单映射，仅用于可视化）
    h, w = 480, 640
    vis = np.full((h, w, 3), 255, dtype=np.uint8)
    xs = (pts[:, 0] * 0.5 + 0.5) * w
    ys = (1.0 - (pts[:, 1] * 0.5 + 0.5)) * h
    for x, y in zip(xs, ys):
        cv2.circle(vis, (int(x), int(y)), 4, (0, 120, 255), -1)
    # 简易骨架连线（主手 21 点 + 上身 6 点）
    hand_edges = [(i, i + 1) for i in range(0, 20)]
    pose_edges = [(21, 22), (22, 23), (22, 24), (23, 25), (22, 26)]
    for a, b in hand_edges + pose_edges:
        cv2.line(vis, (int(xs[a]), int(ys[a])), (int(xs[b]), int(ys[b])),
                 (40, 40, 200), 2)
    return vis


def _process_source(
    video,
    predictor: Optional[Callable[..., Dict[str, Any]]] = None,
    fake_completion: Optional[Callable[[str], str]] = None,
    history: Optional[List[List[str]]] = None,
):
    """处理一段摄像头/视频输入，返回 (可视化图, gloss文本, 翻译文本, 历史, FPS)。"""
    if video is None:
        return None, "", "", history or [], 0.0

    try:
        out = run_demo_from_source(
            video, predictor=predictor, fake_completion=fake_completion
        )
    except Exception as e:  # 缺 mediapipe/opencv 等依赖时降级
        msg = f"无法处理视频源（可能缺少 mediapipe/opencv）：{e}"
        return None, "", msg, history or [], 0.0

    gloss = out["gloss_json"]
    agent = out["agent_json"]
    gloss_text = " ".join(gloss.get("gloss_sequence", []))
    trans_text = agent.get("natural_language", "")
    fps = out.get("processing_fps", 0.0)

    # 关键点可视化（如可用）
    try:
        from perception.extract import extract_video  # 仅用于取最后一帧骨架
        from configs import load_config
        res = extract_video(video, load_config("perception"), display=False)
        vis = _draw_skeleton(res["skeleton"])
    except Exception:
        vis = None

    new_history = list(history or [])
    new_history.append([gloss_text or "(空)", trans_text])
    return vis, gloss_text, trans_text, new_history, float(fps)


def build_ui(predictor: Optional[Callable[..., Dict[str, Any]]] = None,
             fake_completion: Optional[Callable[[str], str]] = None):
    """构建 Gradio Blocks 界面。未安装 gradio 时抛 RuntimeError。"""
    if not gradio_available():
        raise RuntimeError(
            "gradio 未安装，无法构建界面。请先 `pip install gradio` 后重试。"
        )
    import gradio as gr

    with gr.Blocks(title="手语对话系统") as demo:
        gr.Markdown("# 基于 LLM 智能体的手语对话系统\n"
                    "摄像头/视频输入 → 关键点 → 识别 gloss → 翻译 → 对话")
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="摄像头/视频输入",
                                    sources=["webcam", "upload"])
                run_btn = gr.Button("识别并对话", variant="primary")
                fps_out = gr.Number(label="处理 FPS", value=0.0)
            with gr.Column():
                vis_out = gr.Image(label="关键点可视化")
                gloss_out = gr.Textbox(label="识别 gloss", lines=2)
                trans_out = gr.Textbox(label="翻译文本", lines=3)
                hist_out = gr.Chatbot(label="对话历史")

        state_history = gr.State([])

        def on_run(video, history):
            return _process_source(video, predictor=predictor,
                                   fake_completion=fake_completion, history=history)

        run_btn.click(
            on_run,
            inputs=[video_in, state_history],
            outputs=[vis_out, gloss_out, trans_out, hist_out, fps_out],
        ).then(lambda h: h, inputs=[hist_out], outputs=[state_history])

    return demo


def launch(*args, **kwargs):
    """构建并启动 Gradio 界面。"""
    demo = build_ui(
        predictor=kwargs.pop("predictor", None),
        fake_completion=kwargs.pop("fake_completion", None),
    )
    return demo.launch(*args, **kwargs)


__all__ = ["build_ui", "launch", "gradio_available", "run_demo", "run_demo_from_source"]
