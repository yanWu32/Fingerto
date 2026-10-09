"""识别层 — 骨骼序列 -> gloss 序列 + 置信度 + 时间戳。

基线模型：ST-GCN
消融对比：Transformer+GCN、CNN+LSTM

契约输出：
    {"gloss_sequence": [...], "confidence": [...], "timestamps": [[s,e], ...]}
"""
