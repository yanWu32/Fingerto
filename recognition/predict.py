"""识别层推理入口：skeleton_seq -> gloss JSON。

契约输出（tests/contract.validate_gloss_output 校验通过）：
    {
      "gloss_sequence": [str, ...],
      "confidence":     [float, ...],   # 与 gloss_sequence 等长 ∈ [0,1]
      "timestamps":     [[s, e], ...],  # 单位秒，与 gloss_sequence 等长，s<=e
    }

识别层不输出命令，只输出 gloss 序列 + 置信度 + 时间戳。
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch

# 让 prediction 能在 tests/ 下导入 contract
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.contract import validate_gloss_output  # noqa: E402

from .models import build_model  # noqa: E402


class Predictor:
    """封装已加载的识别模型，对单窗口做前向 -> softmax -> top-1 词 + 置信度 + 时间窗。"""

    def __init__(self, model, classes: list[str], device: str = "cpu",
                 window_frames: int | None = None, fps: float = 30.0):
        self.model = model.to(device).eval()
        self.classes = list(classes)
        self.device = device
        self.fps = fps
        self.window_frames = window_frames  # 若为 None，则用输入实际帧数

    @torch.no_grad()
    def predict_window(self, skeleton_seq: np.ndarray) -> dict:
        """对单窗口 (t',27,3) 推理。

        Args:
            skeleton_seq: (t', 27, 3) numpy，必须符合骨骼契约。

        Returns:
            gloss JSON（经 validate_gloss_output 校验）。
        """
        seq = np.asarray(skeleton_seq, dtype=np.float32)
        if seq.ndim != 3 or seq.shape[1] != 27 or seq.shape[2] != 3:
            raise ValueError(f"识别层输入须为 (t',27,3)，收到 {seq.shape}")
        t = seq.shape[0]
        if self.window_frames is not None and t != self.window_frames:
            # 截断或补零到模型期望长度
            w = self.window_frames
            if t >= w:
                seq = seq[:w]
            else:
                if t == 0:
                    seq = np.zeros((w, 27, 3), dtype=np.float32)
                else:
                    pad = np.repeat(seq[-1:], w - t, axis=0)
                    seq = np.concatenate([seq, pad], axis=0)
            t = w

        x = torch.from_numpy(seq.copy()).unsqueeze(0).to(self.device)  # (1, t, 27, 3)
        logits = self.model(x)                                        # (1, num_classes)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        top1 = int(np.argmax(probs))
        conf = float(probs[top1])

        gloss = self.classes[top1] if 0 <= top1 < len(self.classes) else "unknown"
        start = 0.0
        end = float(t) / float(self.fps)

        out = {
            "gloss_sequence": [gloss],
            "confidence": [conf],
            "timestamps": [[start, end]],
        }
        validate_gloss_output(out)  # 契约自校验，失败抛 ValueError
        return out


def predict(skeleton_seq, fps: float = 30.0, classes: list[str] | None = None,
            model_name: str = "stgcn", num_classes: int = 60,
            weights_path: str | None = None, device: str = "cpu",
            window_frames: int | None = None) -> dict:
    """模块级推理入口。

    无权重时用随机初始化模型（仅验证前向/契约链路，mock 场景）。
    有 weights_path 则加载 state_dict。
    """
    if classes is None:
        classes = [f"gloss_{i}" for i in range(num_classes)]
    model = build_model(model_name, num_classes=len(classes))
    if weights_path is not None and os.path.isfile(weights_path):
        sd = torch.load(weights_path, map_location=device)
        model.load_state_dict(sd)
    pred = Predictor(model, classes, device=device, window_frames=window_frames, fps=fps)
    return pred.predict_window(skeleton_seq)
