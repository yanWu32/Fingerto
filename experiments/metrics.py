"""实验层指标（简化实现，不依赖 nltk / rouge 外部包）。

提供消融实验所需指标：
- BLEU（字符级 n-gram，含 brevity penalty 的简化实现）
- ROUGE-1 / ROUGE-L（字符级，F1）
- 意图准确率 intent_accuracy
- 对话完成率 dialogue_completion_rate
- 澄清触发率 clarification_trigger_rate
- 端到端延迟 end_to_end_latency（由调用方计时后聚合）

所有比例类指标返回值均 ∈ [0, 1]（延迟为毫秒，>=0）。

设计原则：对中文按「字符」切词（中文无空格），因此 BLEU/ROUGE 的
n-gram 以字符为单位。真实 LLM 接入后无需改本模块，仅需替换数据/completion。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence


# ---------------------------------------------------------------------------
# 通用分词：中文按字符，保留字母/数字整体
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    """把文本切成 token：中文逐字符，连续 ASCII 字母/数字作为一个 token。"""
    if not text:
        return []
    toks: List[str] = []
    buf = ""
    for ch in text:
        if ("一" <= ch <= "鿿") or ch in "，。！？、；：""''（）《》":
            # 中文（含标点）逐个字符
            if buf:
                toks.append(buf)
                buf = ""
            toks.append(ch)
        elif ch.isalnum():
            buf += ch
        else:
            if buf:
                toks.append(buf)
                buf = ""
            # 其它空白/符号丢弃
    if buf:
        toks.append(buf)
    return toks


def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# BLEU（简化，字符级，n=1..4，含 brevity penalty）
# ---------------------------------------------------------------------------
def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """简化 BLEU ∈ [0,1]。

    - 字符级 n-gram 修正精度
    - brevity penalty = min(1, len(hyp)/len(ref))
    当任一阶 precision 为 0 时 BLEU=0（标准做法）。
    """
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    if not hyp:
        return 0.0
    if not ref:
        return 0.0
    cap = max(1, min(max_n, len(ref), len(hyp)))
    precisions: List[float] = []
    for n in range(1, cap + 1):
        hyp_ng = _ngrams(hyp, n)
        ref_ng = _ngrams(ref, n)
        if not hyp_ng or not ref_ng:
            precisions.append(0.0)
            continue
        ref_counts = Counter(ref_ng)
        hyp_counts = Counter(hyp_ng)
        clipped = sum(min(c, ref_counts.get(g, 0)) for g, c in hyp_counts.items())
        precisions.append(clipped / len(hyp_ng))
    if any(p == 0.0 for p in precisions):
        return 0.0
    bp = min(1.0, len(hyp) / len(ref))
    log_mean = sum(math.log(p) for p in precisions) / len(precisions)
    return bp * math.exp(log_mean)


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------
def rouge_n(reference: str, hypothesis: str, n: int = 1) -> float:
    """ROUGE-N（字符级 n-gram）F1 ∈ [0,1]。"""
    ref = _ngrams(_tokenize(reference), n)
    hyp = _ngrams(_tokenize(hypothesis), n)
    if not ref:
        return 1.0 if not hyp else 0.0
    if not hyp:
        return 0.0
    ref_counts = Counter(ref)
    hyp_counts = Counter(hyp)
    overlap = sum(min(c, ref_counts.get(g, 0)) for g, c in hyp_counts.items())
    recall = overlap / len(ref)
    precision = overlap / len(hyp)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L（最长公共子序列）F1 ∈ [0,1]。"""
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    lcs = _lcs_length(ref, hyp)
    recall = lcs / len(ref)
    precision = lcs / len(hyp)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# 分类类指标
# ---------------------------------------------------------------------------
def intent_accuracy(preds: Sequence[str], refs: Sequence[str]) -> float:
    """意图准确率 ∈ [0,1]：逐条比较 pred intent 与 ref intent。"""
    if not preds:
        return 0.0
    assert len(preds) == len(refs), "preds 与 refs 长度必须一致"
    correct = sum(1 for p, r in zip(preds, refs) if p == r)
    return correct / len(preds)


def clarification_trigger_rate(records: Sequence[Dict]) -> float:
    """澄清触发率 ∈ [0,1]：系统输出 clarification==True 的占比（系统行为，非对照 ref）。"""
    if not records:
        return 0.0
    trig = sum(1 for r in records if bool(r.get("clarification")))
    return trig / len(records)


def dialogue_completion_rate(records: Sequence[Dict]) -> float:
    """对话完成率 ∈ [0,1]：系统澄清决策与期望一致（clarification == ref_clarification）的占比。

    含义：该轮对话是否「以正确方式收口」——要么干净给出回复（无需澄清），
    要么正确地识别到不确定并发起澄清。既未正确决策、又漏触发澄清，记为未完成。
    """
    if not records:
        return 0.0
    ok = sum(1 for r in records
             if bool(r.get("clarification")) == bool(r.get("ref_clarification")))
    return ok / len(records)


def end_to_end_latency(records: Sequence[Dict]) -> float:
    """端到端延迟（毫秒）：单轮 run 调用耗时的均值，>=0。

    records 需含 'latency_ms'（由调用方计时）。mock 模式下数值很小但保持单调性：
    C/D（含完整三步管线）> B（单步）> A（纯规则，最快）。
    """
    vals = [float(r["latency_ms"]) for r in records if "latency_ms" in r]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# 聚合：一组 per-turn 记录 -> 指标字典
# ---------------------------------------------------------------------------
def aggregate(records: Sequence[Dict]) -> Dict[str, float]:
    """把同一 (场景, 系统) 的 per-turn 记录聚合为指标字典。

    每条 record 字段：
        hyp_nl, ref_nl, pred_intent, ref_intent,
        clarification(bool), ref_clarification(bool), latency_ms(float)
    """
    recs = list(records)
    if not recs:
        return {
            "bleu": 0.0, "rouge1": 0.0, "rougeL": 0.0,
            "intent_accuracy": 0.0, "clarification_rate": 0.0,
            "completion_rate": 0.0, "eot_latency_ms": 0.0,
        }
    bleu = sum(bleu_score(r["ref_nl"], r["hyp_nl"]) for r in recs) / len(recs)
    rouge1 = sum(rouge_n(r["ref_nl"], r["hyp_nl"], 1) for r in recs) / len(recs)
    rougeL = sum(rouge_l(r["ref_nl"], r["hyp_nl"]) for r in recs) / len(recs)
    return {
        "bleu": bleu,
        "rouge1": rouge1,
        "rougeL": rougeL,
        "intent_accuracy": intent_accuracy(
            [r["pred_intent"] for r in recs], [r["ref_intent"] for r in recs]),
        "clarification_rate": clarification_trigger_rate(recs),
        "completion_rate": dialogue_completion_rate(recs),
        "eot_latency_ms": end_to_end_latency(recs),
    }


def assert_valid_metrics(m: Dict[str, float]) -> None:
    """断言比例类指标 ∈ [0,1]、延迟 >= 0。用于测试闸门。"""
    for k in ("bleu", "rouge1", "rougeL", "intent_accuracy",
              "clarification_rate", "completion_rate"):
        v = m[k]
        assert 0.0 <= v <= 1.0, f"{k}={v} 超出 [0,1]"
    assert m["eot_latency_ms"] >= 0.0, "延迟应为非负"
