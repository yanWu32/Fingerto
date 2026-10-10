"""识别层 — 骨骼序列 -> gloss 序列 + 置信度 + 时间戳。

基线模型：ST-GCN
消融对比：Transformer+GCN、CNN+LSTM

契约输出：
    {"gloss_sequence": [...], "confidence": [...], "timestamps": [[s,e], ...]}

对外暴露：
    - predict(skeleton_seq, fps, ...) -> gloss JSON（经 validate_gloss_output 校验）
    - build_model(name, num_classes) -> 模型实例
    - GlossDataset / build_dataloader
"""

from .models import build_model, STGCN, TransformerGCN, CNNLSTM, MODEL_REGISTRY
from .dataset import GlossDataset, build_dataloader, generate_mock_window, generate_mock_batch
from .predict import predict, Predictor

__all__ = [
    "predict", "Predictor",
    "build_model", "STGCN", "TransformerGCN", "CNNLSTM", "MODEL_REGISTRY",
    "GlossDataset", "build_dataloader", "generate_mock_window", "generate_mock_batch",
]
