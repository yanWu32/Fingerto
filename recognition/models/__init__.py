"""识别层模型注册表。"""

from .stgcn import STGCN
from .transformer_gcn import TransformerGCN
from .cnn_lstm import CNNLSTM

MODEL_REGISTRY = {
    "stgcn": STGCN,
    "transformer_gcn": TransformerGCN,
    "cnn_lstm": CNNLSTM,
}


def build_model(name: str, num_classes: int, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型 {name!r}，可选 {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)


__all__ = ["STGCN", "TransformerGCN", "CNNLSTM", "MODEL_REGISTRY", "build_model"]
