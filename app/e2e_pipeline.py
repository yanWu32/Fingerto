"""四层端到端联调胶水（切分 → 识别 → 智能体），全程过 tests/contract 校验。

背景：感知/切分/识别/智能体各层早已实现并通过 mock/合成测试，但从未在
「一个连续骨架序列」上把它们串起来。本脚本是胶水：
    1. Segmenter.segment() 把 (T,27,3) 切出若干手势单元 Segment；
    2. 对每个 Segment.sub_sequence 调 Predictor.predict_window() 得单词 gloss JSON；
    3. 合并成整条 gloss JSON（gloss_sequence/confidence/timestamps 等长）；
    4. run_pipeline() 把 gloss JSON 送智能体三步管线得 agent JSON。

全程用 tests/contract.validate_* 严校验，任一环节不合法即抛错（不静默降级）。
不修改任何层内部逻辑——只组装数据流。

用法：
    python -m app.e2e_pipeline                 # 合成连续序列端到端跑通验证
    python -m app.e2e_pipeline --weights <pt> --classes data/isw1000_real/glosses.txt \
        --model cnn_lstm                       # 接真实 ISW-1000 权重（需 fingerto env）
    python -m app.e2e_pipeline --video my.mp4 --weights <pt> --classes <txt> \
        --model stgcn --device cuda --out out.json   # 视频 → 骨架 → 切分 → 识别 → 智能体
    python -m app.e2e_pipeline --npz sk.npz --weights <pt> --classes <txt>  # 已抽好的骨架
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from tests.contract import (  # noqa: E402
    validate_gloss_output,
    validate_agent_output,
    check_skeleton_shape,
)
from segmentation.segmenter import SkeletonSegmenter  # noqa: E402
from recognition.predict import Predictor  # noqa: E402
from recognition.models import build_model  # noqa: E402
from agent.pipeline import AgentPipeline  # noqa: E402
from agent.dialogue_state import DialogueState  # noqa: E402
from agent.llm_client import LLMClient  # noqa: E402


def merge_segment_glosses(segments, predictor, fps: float, fallback_seq=None,
                          ctx_frames: int = 20):
    """把每个 segment 的单词 gloss JSON 合并成整条 gloss JSON。

    时间戳用 segment 的真实起止时间（秒），不再用窗口内 [0, t/fps]。
    为缓解「切分段过短（8-16 帧）与模型训练长度 90 帧不匹配」问题，识别前对
    每个 segment 前后各扩展 ctx_frames 帧的真实运动上下文（非补零/循环复制），
    让输入更接近训练分布（训练样本是 90 帧连续手势）。
    返回经的 validate_gloss_output 校验的 dict。
    """
    gloss_sequence: list[str] = []
    confidence: list[float] = []
    timestamps: list[list[float]] = []
    margins: list[float] = []
    low_conf: list[bool] = []

    # 整段上下文（fallback / 上下文扩展用），避免越界读取
    full = np.asarray(fallback_seq, dtype=np.float32) if fallback_seq is not None else None

    for seg in segments:
        sub = np.asarray(seg.sub_sequence, dtype=np.float32)
        if sub.shape[0] == 0:
            continue
        # 上下文扩展：从整段序列里取 [start-CTX, end+1+CTX] 的真实帧
        if full is not None and full.ndim == 3 and full.shape[0] > 0:
            s = max(0, int(seg.start_frame) - ctx_frames)
            e = min(full.shape[0], int(seg.end_frame) + 1 + ctx_frames)
            sub = np.asarray(full[s:e], dtype=np.float32)
        single = predictor.predict_window(sub)  # 返回单元素 gloss JSON
        g = single["gloss_sequence"][0]
        c = single["confidence"][0]
        gloss_sequence.append(g)
        confidence.append(c)
        timestamps.append([seg.start_time, seg.end_time])
        margins.append(float(single.get("margin_top1_top2", [0.0])[0]))
        low_conf.append(bool(single.get("low_confidence", [False])[0]))

    out = {
        "gloss_sequence": gloss_sequence,
        "confidence": confidence,
        "timestamps": timestamps,
        # 可选判别力字段（下游/前端据此判断可信度；契约校验为可选）
        "margin_top1_top2": margins,
        "low_confidence": low_conf,
    }
    # 兜底：切分未产出任何活跃段（视频近乎静止 / 手未入镜）时，
    # 把整段当作单个手势单元识别一次，避免产出空 gloss 序列误导下游。
    if not gloss_sequence and fallback_seq is not None:
        seq = np.asarray(fallback_seq, dtype=np.float32)
        if seq.ndim == 3 and seq.shape[0] > 0:
            single = predictor.predict_window(seq)
            out["gloss_sequence"] = list(single["gloss_sequence"])
            out["confidence"] = [float(c) for c in single["confidence"]]
            out["timestamps"] = [[0.0, float(seq.shape[0]) / max(fps, 1e-6)]]
            out["margin_top1_top2"] = [float(m) for m in single.get("margin_top1_top2", [0.0])]
            out["low_confidence"] = [bool(b) for b in single.get("low_confidence", [False])]
    validate_gloss_output(out)  # 不合法即抛 ValueError
    return out


def run_e2e(skeleton_seq: np.ndarray, predictor, fps: float = 30.0,
            use_agent: bool = True, agent_state=None, ctx_frames: int = 20):
    """跑完整四层链路，返回 (gloss_json, agent_json)。

    skeleton_seq: (T, 27, 3) 连续骨架序列。
    agent_state: 可选 DialogueState，连续对话时跨调用持有。
    ctx_frames: 识别前对每个切分单元前后各扩展的上下文帧数（缓解短段问题）。
    """
    seq = np.asarray(skeleton_seq, dtype=np.float32)
    if not check_skeleton_shape(seq):
        raise ValueError("输入骨架序列不符合 (T,27,3) 契约")

    # 1) 切分
    segments = SkeletonSegmenter().segment(seq, fps=fps)
    # 2)+3) 识别 → 合并 gloss JSON（无活跃段时回退为整段单窗）
    gloss_json = merge_segment_glosses(segments, predictor, fps=fps,
                                       fallback_seq=seq, ctx_frames=ctx_frames)

    # 4) 智能体
    agent_json = None
    if use_agent:
        state = agent_state or DialogueState({})
        # 注入 mock LLM（无 API Key 也能跑通整条链路；真实 LLM 时去掉 fake_completion）
        client = LLMClient({}, fake_completion=_fake_llm)
        pipe = AgentPipeline({}, client=client)
        agent_json = pipe.run(gloss_json, state)
        validate_agent_output(agent_json)

    return gloss_json, agent_json


def _fake_llm(prompt: str) -> str:
    """mock LLM：根据 prompt 关键词做极简确定性回复，仅用于链路自测。"""
    if "翻译" in prompt or "gloss" in prompt.lower():
        return "（mock 翻译）用户用手语表达了一段内容。"
    if "意图" in prompt or "任务" in prompt or "JSON" in prompt:
        return '{"intent": "chat", "task": "none", "natural_language": "（mock）已理解。", "need_clarify": false}'
    return "（mock）ok"


# ---------------------------------------------------------------------------
# 合成连续序列（用于无真实数据时验证整条链路）
# ---------------------------------------------------------------------------
def _make_synthetic_continuous(rng: np.random.Generator, fps: float = 30.0,
                               num_words: int = 4, frames_per_word: int = 40,
                               gap_frames: int = 80):
    """构造一段「静止-运动-静止」交替的连续骨架，模拟词间有停顿的连续手语。

    每个词 = 一段主动运动 + 一段静默间隔。骨架本身随机（仅用于驱动切分/链路，
    识别层用随机权重时 gloss 无意义，但链路与契约全部可验证）。
    """
    seqs = []
    for _ in range(num_words):
        # 运动段：手腕(0)做明显正弦位移；静止段占多数（贴近真实手语：
        # 短手势 + 长停顿），确保自适应阈值落在静止段、手势稳触发
        t = np.arange(frames_per_word)
        base = rng.standard_normal((frames_per_word, 27, 3)).astype(np.float32) * 0.005
        motion = 1.5 * np.sin(2 * np.pi * t / frames_per_word)[:, None, None]
        base[:, 0:2, :] += motion
        seqs.append(base)
        # 静默段：完全静止（零运动，模拟词间自然停顿的干净背景）
        gap = np.zeros((gap_frames, 27, 3), dtype=np.float32)
        seqs.append(gap)
    full = np.concatenate(seqs, axis=0).astype(np.float32)
    return full


def _build_predictor(model_name: str, weights_path: str | None, classes: list[str],
                     device: str, window_frames: int | None, fps: float):
    if weights_path and os.path.isfile(weights_path):
        model = build_model(model_name, num_classes=len(classes))
        sd = __import__("torch").load(weights_path, map_location=device,
                                   weights_only=True)
        model.load_state_dict(sd)
        print(f"[e2e] 加载真实权重：{weights_path}（{len(classes)} 类）")
    else:
        model = build_model(model_name, num_classes=max(len(classes), 2))
        print(f"[e2e] 未提供权重，使用随机初始化模型（仅验证链路/契约，gloss 无意义）")
    return Predictor(model, classes, device=device, window_frames=window_frames, fps=fps)


def main() -> int:
    ap = argparse.ArgumentParser(description="四层端到端联调")
    ap.add_argument("--weights", default=None, help="识别模型权重 .pt（不提供则随机初始化）")
    ap.add_argument("--classes", default=None,
                    help="类别文件（每行一词，如 data/isw1000_real/glosses.txt）")
    ap.add_argument("--model", default="stgcn", choices=["stgcn", "transformer_gcn", "cnn_lstm"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window-frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--ctx-frames", type=int, default=20,
                    help="识别前对每个切分单元前后各扩展的上下文帧数（缓解短段与90帧不匹配）")
    ap.add_argument("--no-agent", action="store_true", help="跳过智能体层，仅验到 gloss JSON")
    ap.add_argument("--num-words", type=int, default=4)
    ap.add_argument("--video", default=None,
                    help="手语视频文件路径（mp4/avi/mov）；提供则走感知层抽骨架")
    ap.add_argument("--npz", default=None,
                    help="已抽好的骨架 .npz（含 skeleton 键）；与 --video 二选一")
    ap.add_argument("--max-frames", type=int, default=None, help="最多处理帧数")
    ap.add_argument("--out", default=None, help="把 gloss/agent JSON 存成 .json")
    args = ap.parse_args()

    if args.classes and os.path.isfile(args.classes):
        classes = [l.strip() for l in open(args.classes, encoding="utf-8") if l.strip()]
    else:
        # 默认合成 60 类占位（与识别层 mock 对齐）
        classes = [f"gloss_{i}" for i in range(60)]

    predictor = _build_predictor(args.model, args.weights, classes,
                                 args.device, args.window_frames, args.fps)

    # ---- 输入源：真实视频 / 已抽骨架 npz / 合成序列（三选一）----
    if args.video:
        from configs import load_config
        from perception.extract import extract_video

        # --fps 自动探测：用 OpenCV 读视频真实 fps，与用户给定不符时采用真实值
        if args.video and os.path.isfile(args.video):
            try:
                import cv2
                _cap = cv2.VideoCapture(args.video)
                real_fps = _cap.get(cv2.CAP_PROP_FPS)
                _cap.release()
                if real_fps and real_fps > 1 and abs(real_fps - args.fps) > 0.5:
                    print(f"[e2e] 警告：视频真实 fps={real_fps:.2f}，与 --fps "
                          f"{args.fps} 不符，已自动采用真实 fps")
                    args.fps = float(real_fps)
            except Exception as _e:  # 探测失败不阻塞，沿用 --fps
                print(f"[e2e] fps 自动探测失败（{_e}），沿用 --fps={args.fps}")

        result = extract_video(
            args.video, load_config("perception"),
            smooth=True, max_frames=args.max_frames, display=False,
            out_path=None,
        )
        seq = np.asarray(result["skeleton"], dtype=np.float32)
        print(f"[e2e] 视频 {args.video} -> 骨架 shape={seq.shape}")
    elif args.npz:
        data = np.load(args.npz, allow_pickle=True)
        seq = np.asarray(data["skeleton"], dtype=np.float32)
        print(f"[e2e] 载入骨架 {args.npz} shape={seq.shape}")
    else:
        rng = np.random.default_rng(20261010)
        seq = _make_synthetic_continuous(rng, fps=args.fps, num_words=args.num_words)
        print(f"[e2e] 合成连续序列 shape={seq.shape}")

    # 切分可能产不出段（如视频几乎静止），兜底：整段当一个窗口
    if seq.ndim != 3 or seq.shape[1:] != (27, 3):
        raise ValueError(f"骨架形状非法 {seq.shape}，应为 (T,27,3)")

    gloss_json, agent_json = run_e2e(
        seq, predictor, fps=args.fps, use_agent=not args.no_agent,
        ctx_frames=args.ctx_frames,
    )
    print("[e2e] gloss JSON =", gloss_json)
    if agent_json is not None:
        print("[e2e] agent JSON =", agent_json)

    if args.out:
        import json
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"gloss_json": gloss_json, "agent_json": agent_json}
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"[e2e] 结果已保存 -> {out_path}")

    print("[e2e] 全链路契约校验通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
