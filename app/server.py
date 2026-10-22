"""FastAPI 后端服务：把四层链路（感知→切分→识别→智能体）包成 REST + SSE。

端点（契约见 docs/openapi.v3.json）：
    GET  /health                 依赖就绪探针（前端据 ready=false 降级到 mock）
    POST /recognition            上传视频 → gloss 序列（同步）
    POST /pipeline               上传视频 → 识别 + 智能体（同步，全量）
    POST /pipeline/stream        上传视频 → 分阶段流式（SSE）
    GET  /skeleton/{jobId}       按 jobId 二次拉取 (T,27,3) 骨架
    POST /pipeline/{jobId}/cancel 取消正在进行的流式任务

设计要点：
- 所有重依赖（torch / mediapipe / cv2 / openai）一律**懒加载**，本模块可被
  `import app.server` 而不崩；真正调用推理时才 import。
- Predictor（识别模型）按 (model, weights, device) 缓存为单例，避免每次请求重加载权重。
- 视频上传做 MIME + 后缀 + 魔数二次校验，仅放行 mp4/avi/mov/webm，≤50MB。
- 缺感知/推理依赖时接口返回 503（而非 500 崩），前端据此回退 mock。
- 唯一数据流来源：`pipeline_events()` 生成器，同步/流式端点共用，保证契约一致。

启动（在仓库根目录）：
    python -m app.server                 # 默认 127.0.0.1:8000
    python -m app.server --port 8000 --device cpu
或通过 serve.py：
    python serve.py --backend-fastapi    # 拉起本服务并把 /api 反代到它
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

# 仓库根（Fingerto/）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse, JSONResponse  # noqa: E402

from tests.contract import (  # noqa: E402
    validate_gloss_output,
    validate_agent_output,
    check_skeleton_shape,
)

# ---------------------------------------------------------------------------
# 默认配置（相对仓库根）
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "cnn_lstm"
DEFAULT_WEIGHTS = _ROOT / "data" / "isw1000_real" / "cnn_lstm_100g_epoch80.pt"
DEFAULT_CLASSES = _ROOT / "data" / "isw1000_real" / "glosses.txt"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB
ALLOWED_KINDS = {"mp4", "avi", "mov", "webm"}

# ---------------------------------------------------------------------------
# 进程内状态：任务表（骨架缓存 + 取消标志）
# ---------------------------------------------------------------------------
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_MAX_JOBS = 200


def _register_job(job_id: str) -> Dict[str, Any]:
    with _JOBS_LOCK:
        if len(_JOBS) >= _MAX_JOBS:
            # 丢弃最旧的
            oldest = next(iter(_JOBS))
            _JOBS.pop(oldest, None)
        rec = {"cancel": False, "skeleton": None, "fps": None,
               "num_frames": None, "shape": None}
        _JOBS[job_id] = rec
        return rec


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def _cancel_job(job_id: str) -> bool:
    with _JOBS_LOCK:
        rec = _JOBS.get(job_id)
        if rec is None:
            return False
        rec["cancel"] = True
        return True


# ---------------------------------------------------------------------------
# Predictor 缓存（单例，按 model+weights+device）
# ---------------------------------------------------------------------------
_PREDICTORS: Dict[tuple, Any] = {}
_PRED_LOCK = threading.Lock()


def get_predictor(model_name: str, weights_path: Optional[str],
                  classes_path: Optional[str], device: str,
                  window_frames: Optional[int] = None, fps: float = 30.0):
    """按 (model, weights, device) 返回缓存的 Predictor，首次构建并加载权重。"""
    key = (model_name, weights_path or "", device)
    with _PRED_LOCK:
        if key in _PREDICTORS:
            return _PREDICTORS[key]

    # 懒加载重依赖
    from recognition.predict import Predictor
    from recognition.models import build_model

    if classes_path and os.path.isfile(classes_path):
        classes = [l.strip() for l in open(classes_path, encoding="utf-8") if l.strip()]
    else:
        classes = [f"gloss_{i}" for i in range(100)]

    model = build_model(model_name, num_classes=len(classes))
    if weights_path and os.path.isfile(weights_path):
        import torch
        sd = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(sd)
    predictor = Predictor(model, classes, device=device,
                          window_frames=window_frames, fps=fps)
    with _PRED_LOCK:
        _PREDICTORS[key] = predictor
    return predictor


# ---------------------------------------------------------------------------
# 依赖探针（/health 用，返回哪些依赖就绪）
# ---------------------------------------------------------------------------
# 依赖探针结果会被缓存：__import__ torch/mediapipe/cv2 很慢（冷启动数秒），
# 若每次 /health 都重跑会让前端首次探测超时误判「后端不可达」。
# 进程内只探测一次，状态变化需重启后端（依赖安装属于环境变更，本就需重启）。
_DEPS_CACHE: Dict[str, bool] | None = None


def probe_deps() -> Dict[str, bool]:
    global _DEPS_CACHE
    if _DEPS_CACHE is not None:
        return _DEPS_CACHE
    deps = {}
    for name, mod in [("torch", "torch"), ("mediapipe", "mediapipe"),
                      ("opencv", "cv2"), ("fastapi", "fastapi")]:
        try:
            __import__(mod)
            deps[name] = True
        except Exception:
            deps[name] = False
    _DEPS_CACHE = deps
    return deps


def probe_models_loaded() -> List[str]:
    out = []
    if DEFAULT_WEIGHTS.is_file():
        out.append(DEFAULT_MODEL)
    return out


# ---------------------------------------------------------------------------
# 视频校验（MIME/后缀/魔数二次校验）
# ---------------------------------------------------------------------------
def _sniff_video(head: bytes, filename: str) -> Optional[str]:
    name = (filename or "").lower()
    if len(head) >= 12:
        if head[4:8] == b"ftyp":
            brand = head[8:12]
            return "mov" if brand == b"qt  " else "mp4"
        if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
            return "avi"
        if head[:4] == b"\x1a\x45\xdf\xa3":
            return "webm"
    if name.endswith(".mp4"):
        return "mp4"
    if name.endswith(".avi"):
        return "avi"
    if name.endswith(".mov"):
        return "mov"
    if name.endswith(".webm"):
        return "webm"
    return None


def _validate_upload(upload: UploadFile) -> str:
    """校验上传文件，返回 kind（mp4/avi/mov/webm）。非法则抛 HTTPException。"""
    # 大小（流末尾 seek 取长度）
    try:
        upload.file.seek(0, 2)
        size = upload.file.tell()
        upload.file.seek(0)
    except Exception:
        size = 0
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="视频超过 50MB 上限")
    if size == 0:
        raise HTTPException(status_code=400, detail="空文件或无法读取")

    head = upload.file.read(4096)
    upload.file.seek(0)
    kind = _sniff_video(head, upload.filename or "")
    if kind is None:
        raise HTTPException(status_code=400,
                            detail="仅支持 mp4/avi/mov/webm 视频")
    return kind


# ---------------------------------------------------------------------------
# 感知：视频 → 骨架（懒加载 mediapipe/cv2）
# ---------------------------------------------------------------------------
def extract_skeleton(video_path: str, max_frames: Optional[int] = None):
    """返回 (seq: np.ndarray(T,27,3), fps: float, num_frames: int)。"""
    from perception.extract import extract_video
    from configs import load_config

    cfg = load_config("perception")
    result = extract_video(video_path, cfg, smooth=True,
                           max_frames=max_frames, display=False, out_path=None)
    seq = np.asarray(result["skeleton"], dtype=np.float32)

    # fps 自动探测
    fps = 30.0
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        real = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if real and real > 1:
            fps = float(real)
    except Exception:
        pass
    return seq, fps, int(seq.shape[0])


# ---------------------------------------------------------------------------
# 智能体：3 步（真实 AgentPipeline，可注入 fake LLM）
# ---------------------------------------------------------------------------
def _fake_llm(prompt: str) -> str:
    if "翻译" in prompt or "gloss" in prompt.lower():
        return "（mock 翻译）用户用手语表达了一段内容。"
    if "意图" in prompt or "任务" in prompt or "JSON" in prompt:
        return '{"intent": "chat", "task": "none", "natural_language": "（mock）已理解。", "need_clarify": false}'
    if "低置信" in prompt or "澄清" in prompt:
        return ""
    return "（mock）ok"


def _use_fake_llm() -> bool:
    if os.environ.get("FINGERTO_AGENT_FAKE") == "1":
        return True
    if os.environ.get("DEEPSEEK_API_KEY"):
        return False
    return True


def _run_agent(gloss_json: Dict[str, Any], agent_cfg: Dict[str, Any],
               fake) -> Dict[str, Any]:
    """跑智能体三步，返回经契约校验的 agent JSON。"""
    from agent import build_agent
    from agent.dialogue_state import DialogueState

    state = DialogueState(agent_cfg)
    agent = build_agent(agent_cfg, fake_completion=fake)
    gs = list(gloss_json.get("gloss_sequence", []))
    cf = list(gloss_json.get("confidence", []))
    low = agent.low_confidence_words(gs, cf)
    need = agent.needs_clarify(cf)

    confirmed = agent.step1_confirm(gs, cf)
    nl = agent.step2_translate(confirmed, state)
    dec = agent.step3_decide(nl, state)
    state.add_turn(role="user", content=nl,
                   intent=dec.get("intent", "chat"),
                   task=dec.get("task", "none"), gloss=gs)
    if need:
        dec["need_clarify"] = True
    result = {
        "natural_language": nl,
        "intent": dec.get("intent", "chat"),
        "task": dec.get("task", "none"),
        "clarification": bool(dec.get("need_clarify", False) or need),
        "low_confidence_words": low,
    }
    validate_agent_output(result)
    return result


# ---------------------------------------------------------------------------
# 识别：切分 → 上下文扩展 → 逐窗 predict（与 e2e_pipeline.merge_segment_glosses 对齐）
# ---------------------------------------------------------------------------
def _recognize(seq: np.ndarray, segments, predictor, fps: float,
               ctx_frames: int, fallback_seq) -> Dict[str, Any]:
    gloss_sequence: List[str] = []
    confidence: List[float] = []
    timestamps: List[List[float]] = []
    margins: List[float] = []
    low_conf: List[bool] = []

    full = np.asarray(fallback_seq, dtype=np.float32) if fallback_seq is not None else None

    for seg in segments:
        sub = np.asarray(seg.sub_sequence, dtype=np.float32)
        if sub.shape[0] == 0:
            continue
        if full is not None and full.ndim == 3 and full.shape[0] > 0:
            s = max(0, int(seg.start_frame) - ctx_frames)
            e = min(full.shape[0], int(seg.end_frame) + 1 + ctx_frames)
            sub = np.asarray(full[s:e], dtype=np.float32)
        single = predictor.predict_window(sub)
        gloss_sequence.append(single["gloss_sequence"][0])
        confidence.append(float(single["confidence"][0]))
        timestamps.append([seg.start_time, seg.end_time])
        margins.append(float(single.get("margin_top1_top2", [0.0])[0]))
        low_conf.append(bool(single.get("low_confidence", [False])[0]))

    out = {
        "gloss_sequence": gloss_sequence,
        "confidence": confidence,
        "timestamps": timestamps,
        "margin_top1_top2": margins,
        "low_confidence": low_conf,
    }
    if not gloss_sequence and fallback_seq is not None:
        s2 = np.asarray(fallback_seq, dtype=np.float32)
        if s2.ndim == 3 and s2.shape[0] > 0:
            single = predictor.predict_window(s2)
            out["gloss_sequence"] = list(single["gloss_sequence"])
            out["confidence"] = [float(c) for c in single["confidence"]]
            out["timestamps"] = [[0.0, float(s2.shape[0]) / max(fps, 1e-6)]]
            out["margin_top1_top2"] = [float(m) for m in single.get("margin_top1_top2", [0.0])]
            out["low_confidence"] = [bool(b) for b in single.get("low_confidence", [False])]
    validate_gloss_output(out)
    return out


def _count_windows(segmenter, segments) -> int:
    total = 0
    for seg in segments:
        L = seg.num_frames
        if L <= segmenter.window_size:
            total += 1
        else:
            total += ((L - segmenter.window_size) // segmenter.window_stride) + 1
    return total


# ---------------------------------------------------------------------------
# 唯一数据流来源：生成器，逐阶段 yield SSE 事件
# ---------------------------------------------------------------------------
def pipeline_events(video_path: str, model_name: str, device: str,
                    use_agent: bool, ctx_frames: int, job_id: str,
                    weights_path: Optional[str] = None,
                    classes_path: Optional[str] = None,
                    max_frames: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    rec = _get_job(job_id)
    if rec is None:
        rec = _register_job(job_id)

    def _cancelled() -> bool:
        return bool(rec.get("cancel"))

    # ---- 1) 感知 ----
    yield {"type": "stage", "stage": "perception", "label": "正在抽取骨架"}
    try:
        seq, fps, n_frames = extract_skeleton(video_path, max_frames=max_frames)
    except Exception as e:  # 缺 mediapipe/cv2 或视频损坏
        yield {"type": "error", "code": 503,
               "msg": f"感知依赖未就绪或视频无法解析：{e}"}
        return
    if not check_skeleton_shape(seq):
        yield {"type": "error", "code": 503, "msg": "抽取到的骨架不符合 (T,27,3) 契约"}
        return

    rec["skeleton"] = seq
    rec["fps"] = fps
    rec["num_frames"] = n_frames
    rec["shape"] = list(seq.shape)
    yield {
        "type": "perception_done",
        "shape": str(list(seq.shape)),
        "numFrames": n_frames,
        "fps": fps,
        "durationSec": round(n_frames / max(fps, 1e-6), 2),
        "skeletonUrl": f"/api/skeleton/{job_id}",
    }
    if _cancelled():
        yield {"type": "error", "code": 499, "msg": "已取消"}
        return

    # ---- 2) 切分 ----
    yield {"type": "stage", "stage": "segmentation", "label": "正在切分手势单元"}
    try:
        from segmentation.segmenter import SkeletonSegmenter
        segmenter = SkeletonSegmenter()
        segments = segmenter.segment(seq, fps=fps)
    except Exception as e:
        yield {"type": "error", "code": 503, "msg": f"切分失败：{e}"}
        return
    num_windows = _count_windows(segmenter, segments)
    yield {
        "type": "segmentation_done",
        "segments": [{"startTime": seg.start_time, "endTime": seg.end_time}
                     for seg in segments],
        "numSegments": len(segments),
        "numWindows": num_windows,
    }
    if _cancelled():
        yield {"type": "error", "code": 499, "msg": "已取消"}
        return

    # ---- 3) 识别（逐窗，发 recognition_window，最后聚合）----
    yield {"type": "stage", "stage": "recognition", "label": "正在识别 gloss"}
    try:
        predictor = get_predictor(model_name, weights_path, classes_path,
                                  device, window_frames=None, fps=fps)
    except Exception as e:
        yield {"type": "error", "code": 503, "msg": f"识别模型加载失败：{e}"}
        return

    # 逐窗发事件（与 e2e 同逻辑：上下文扩展后再 predict）
    full = seq
    for seg in segments:
        if _cancelled():
            yield {"type": "error", "code": 499, "msg": "已取消"}
            return
        sub = np.asarray(seg.sub_sequence, dtype=np.float32)
        if sub.shape[0] == 0:
            continue
        s = max(0, int(seg.start_frame) - ctx_frames)
        e = min(full.shape[0], int(seg.end_frame) + 1 + ctx_frames)
        sub = np.asarray(full[s:e], dtype=np.float32)
        try:
            single = predictor.predict_window(sub)
            g = single["gloss_sequence"][0]
            c = float(single["confidence"][0])
        except Exception:
            continue
        yield {
            "type": "recognition_window",
            "window": {
                "startTime": seg.start_time,
                "endTime": seg.end_time,
                "gloss": g,
                "confidence": c,
            },
        }

    gloss_json = _recognize(seq, segments, predictor, fps, ctx_frames, fallback_seq=seq)
    yield {"type": "recognition_done", "recognition": gloss_json}

    # ---- 4) 智能体 ----
    agent_json = None
    if use_agent:
        yield {"type": "stage", "stage": "agent_step1", "label": "智能体：低置信消歧"}
        try:
            from configs import load_config
            agent_cfg = load_config("agent").to_dict()
            fake = _fake_llm if _use_fake_llm() else None
            agent_json = _run_agent(gloss_json, agent_cfg, fake)
        except Exception as e:
            yield {"type": "error", "code": 503, "msg": f"智能体执行失败：{e}"}
            return
        yield {"type": "agent_done", "agent": agent_json}
        if _cancelled():
            yield {"type": "error", "code": 499, "msg": "已取消"}
            return

    # ---- 5) 应用层（无真实 IoT 执行器，仅映射任务，诚实标注 pending）----
    task = agent_json.get("task", "none") if agent_json else "none"
    task_name = task.replace("_", " ") if task != "none" else "无"
    yield {
        "type": "application",
        "apiPath": f"/api/device/{task}" if task != "none" else None,
        "taskName": task_name,
        "task": task,
        "controllerStatus": "pending",
        "changes": [],
    }

    # ---- 6) 完成 ----
    yield {
        "type": "done",
        "recognition": gloss_json,
        "agent": agent_json,
    }


# ---------------------------------------------------------------------------
# SSE 格式化 + 异步包装（避免阻塞事件循环）
# ---------------------------------------------------------------------------
def _sse(event: Dict[str, Any]) -> str:
    t = event.get("type", "")
    return f"event: {t}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _stream(gen: Iterator[Dict[str, Any]]):
    loop = asyncio.get_event_loop()
    while True:
        try:
            ev = await loop.run_in_executor(None, lambda: next(gen))
        except StopIteration:
            break
        yield _sse(ev)


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(title="Fingerto 手语识别后端 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


def _envelope(code: int, msg: str, data: Any = None) -> Dict[str, Any]:
    return {"code": code, "msg": msg, "data": data}


@app.get("/health")
def health():
    deps = probe_deps()
    ready = bool(deps.get("mediapipe") and deps.get("opencv") and deps.get("torch"))
    return _envelope(0, "success", {
        "ready": ready,
        "deps": {
            "torch": deps.get("torch", False),
            "mediapipe": deps.get("mediapipe", False),
            "opencv": deps.get("opencv", False),
        },
        "models_loaded": probe_models_loaded(),
    })


@app.post("/recognition")
def recognition(
    video: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    device: str = Form("cpu"),
    ctx_frames: int = Form(20),
):
    kind = _validate_upload(video)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{kind}", delete=False) as f:
            while True:
                chunk = video.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
            tmp = f.name
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        _register_job(job_id)
        gen = pipeline_events(tmp, model, device, use_agent=False,
                              ctx_frames=ctx_frames, job_id=job_id,
                              weights_path=str(DEFAULT_WEIGHTS),
                              classes_path=str(DEFAULT_CLASSES))
        # 收集 recognition_done 事件
        gloss = None
        for ev in gen:
            if ev.get("type") == "recognition_done":
                gloss = ev["recognition"]
            elif ev.get("type") == "error":
                code = ev.get("code", 503)
                raise HTTPException(status_code=code, detail=ev.get("msg", "处理失败"))
        if gloss is None:
            raise HTTPException(status_code=503, detail="识别未产出结果")
        return _envelope(0, "success", gloss)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"识别失败：{e}")
    finally:
        if tmp and os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


@app.post("/pipeline")
def pipeline(
    video: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    device: str = Form("cpu"),
    use_agent: bool = Form(True),
    ctx_frames: int = Form(20),
):
    kind = _validate_upload(video)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{kind}", delete=False) as f:
            while True:
                chunk = video.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
            tmp = f.name
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        _register_job(job_id)
        gen = pipeline_events(tmp, model, device, use_agent=use_agent,
                              ctx_frames=ctx_frames, job_id=job_id,
                              weights_path=str(DEFAULT_WEIGHTS),
                              classes_path=str(DEFAULT_CLASSES))
        done = None
        for ev in gen:
            if ev.get("type") == "done":
                done = ev
            elif ev.get("type") == "error":
                code = ev.get("code", 503)
                raise HTTPException(status_code=code, detail=ev.get("msg", "处理失败"))
        if done is None:
            raise HTTPException(status_code=503, detail="流水线未产出结果")
        return _envelope(0, "success", {
            "recognition": done["recognition"],
            "agent": done["agent"],
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"流水线失败：{e}")
    finally:
        if tmp and os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


@app.post("/pipeline/stream")
async def pipeline_stream(
    video: UploadFile = File(...),
    jobId: Optional[str] = Form(None),
    model: str = Form(DEFAULT_MODEL),
    device: str = Form("cpu"),
    ctx_frames: int = Form(20),
):
    kind = _validate_upload(video)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{kind}", delete=False) as f:
            while True:
                chunk = video.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
            tmp = f.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法接收文件：{e}")

    job_id = jobId or f"job-{uuid.uuid4().hex[:12]}"
    _register_job(job_id)
    gen = pipeline_events(tmp, model, device, use_agent=True,
                          ctx_frames=ctx_frames, job_id=job_id,
                          weights_path=str(DEFAULT_WEIGHTS),
                          classes_path=str(DEFAULT_CLASSES))

    async def _gen():
        try:
            async for chunk in _stream(gen):
                yield chunk
        finally:
            if tmp and os.path.isfile(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/skeleton/{jobId}")
def skeleton(jobId: str):
    rec = _get_job(jobId)
    if rec is None or rec.get("skeleton") is None:
        raise HTTPException(status_code=404, detail="job 不存在或骨架已过期")
    seq = np.asarray(rec["skeleton"], dtype=np.float32)
    return _envelope(0, "success", {
        "shape": str(list(seq.shape)),
        "fps": rec.get("fps"),
        "frames": int(seq.shape[0]),
        "skeleton": seq.tolist(),
    })


@app.post("/pipeline/{jobId}/cancel")
def cancel(jobId: str):
    ok = _cancel_job(jobId)
    if not ok:
        raise HTTPException(status_code=404, detail="job 不存在")
    return _envelope(0, "success", {"jobId": jobId, "cancelled": True})


# ---------------------------------------------------------------------------
# CLI 启动
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Fingerto FastAPI 后端服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cpu", help="推理设备 cpu/cuda")
    args = ap.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
