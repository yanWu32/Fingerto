"""智能体层（核心）— LLM 三步提示管线 + 对话状态管理。

Step1 确认（低置信消歧）
Step2 翻译（gloss -> 自然语言）
Step3 决策（意图识别 / 任务生成）

契约输入：{"gloss_sequence", "confidence", "timestamps"}
契约输出：{"natural_language", "intent", "task", "clarification", "low_confidence_words"}
"""
