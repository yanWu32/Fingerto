"""实验层测试（R-experiment 交付，仅新增文件，不改动其它层）。

覆盖：
- metrics 各函数返回值 ∈ [0,1]，聚合结果合法
- 四套系统均可实例化（build_system 全 SYSTEM_NAMES）
- 在 fake_completion 下 run_all 产出 5 场景 × 4 系统
- 消融信号合理：System C 在低置信/指代场景上澄清触发率、意图准确率
  不低于（且多轮指代场景上高于）规则基线 A
- run_ablation 落盘 csv/json 可写可读
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs import load_config
from agent import build_system, SYSTEM_NAMES
from experiments.metrics import (
    bleu_score, rouge_n, rouge_l,
    intent_accuracy, clarification_trigger_rate,
    dialogue_completion_rate, end_to_end_latency,
    aggregate, assert_valid_metrics,
)
from experiments.ablation import (
    ablation_fake_completion, build_scenarios, build_systems, run_all,
)

SCENARIO_IDS = [s["id"] for s in build_scenarios()]


@pytest.fixture(scope="module")
def cfg():
    return load_config("agent")


# ---------------------------------------------------------------------------
# metrics 单测
# ---------------------------------------------------------------------------
def test_bleu_identical_is_one():
    assert bleu_score("我想喝水", "我想喝水") == 1.0


def test_bleu_disjoint_is_zero():
    assert bleu_score("我想喝水", "他关窗户") == 0.0


def test_bleu_range():
    v = bleu_score("开灯", "开灯吗")
    assert 0.0 <= v <= 1.0


def test_rouge_range_and_identical():
    assert 0.0 <= rouge_n("开灯", "关灯", 1) <= 1.0
    assert rouge_l("我想喝水", "我想喝水") == 1.0
    assert rouge_l("", "") == 0.0  # 空输入安全


def test_intent_accuracy():
    assert intent_accuracy(["chat", "command"], ["chat", "command"]) == 1.0
    assert intent_accuracy(["chat", "chat"], ["chat", "command"]) == 0.5
    assert intent_accuracy([], []) == 0.0


def test_clarification_and_completion_rates():
    recs = [
        {"clarification": True, "ref_clarification": True},
        {"clarification": False, "ref_clarification": False},
        {"clarification": True, "ref_clarification": False},
    ]
    assert clarification_trigger_rate(recs) == pytest.approx(2 / 3)
    assert dialogue_completion_rate(recs) == pytest.approx(2 / 3)
    assert end_to_end_latency([{"latency_ms": 1.0}, {"latency_ms": 3.0}]) == 2.0


def test_aggregate_valid_and_ranges():
    recs = [
        {"hyp_nl": "我想喝水", "ref_nl": "我想喝水",
         "pred_intent": "chat", "ref_intent": "chat",
         "clarification": False, "ref_clarification": False, "latency_ms": 0.5},
        {"hyp_nl": "开灯", "ref_nl": "开灯",
         "pred_intent": "command", "ref_intent": "command",
         "clarification": False, "ref_clarification": False, "latency_ms": 0.7},
    ]
    m = aggregate(recs)
    assert_valid_metrics(m)
    assert m["bleu"] == pytest.approx(1.0)
    assert m["intent_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# 系统实例化
# ---------------------------------------------------------------------------
def test_system_names_mapping():
    # 与 agent/systems.SYSTEM_NAMES 一致（A/B/C/D）
    assert SYSTEM_NAMES == ["A", "B", "C", "D"]


@pytest.mark.parametrize("name", SYSTEM_NAMES)
def test_build_system_instantiable(cfg, name):
    s = build_system(name, cfg, fake_completion=ablation_fake_completion)
    assert s is not None
    # 能跑出契约合法的 agent 输出
    from tests.contract import validate_agent_output
    out = s.run(build_scenarios()[0]["turns"][0])
    assert validate_agent_output(out) is True


# ---------------------------------------------------------------------------
# 端到端消融
# ---------------------------------------------------------------------------
def test_run_all_shape_and_ranges(cfg):
    results, flat = run_all(cfg, fake_completion=ablation_fake_completion)
    assert set(results.keys()) == set(SCENARIO_IDS)
    assert len(results) == 5
    for sc_id, by_sys in results.items():
        assert set(by_sys.keys()) == set(SYSTEM_NAMES)
        for name, m in by_sys.items():
            assert_valid_metrics(m)  # 全部比例类 ∈[0,1]，延迟≥0
    assert len(flat) == 5 * 4


def test_coreference_shows_agent_contribution(cfg):
    """多轮指代场景：System C（含对话状态）意图准确率严格高于规则基线 A 与无状态 B。

    证明「智能体层 + 对话状态」相对纯规则基线 / 单轮无状态对多轮上下文有贡献。
    （A 在 turn1 开灯可命中规则，turn2 指代「它开」误判 chat；B 无状态更差。）
    """
    results, _ = run_all(cfg, fake_completion=ablation_fake_completion)
    c_acc = results["coreference"]["C"]["intent_accuracy"]
    a_acc = results["coreference"]["A"]["intent_accuracy"]
    b_acc = results["coreference"]["B"]["intent_accuracy"]
    assert c_acc == 1.0
    assert c_acc > a_acc
    assert c_acc > b_acc


def test_low_conf_clarification_triggered_by_full_pipeline(cfg):
    """低置信消歧 / 多义手势：C、D 触发澄清，A、B 不触发（验证 Step1 价值）。"""
    results, _ = run_all(cfg, fake_completion=ablation_fake_completion)
    for sc in ("low_conf_disambig", "polysemous"):
        assert results[sc]["C"]["clarification_rate"] == 1.0
        assert results[sc]["A"]["clarification_rate"] == 0.0


def test_d_fallback_equals_c(cfg):
    """System D 无权重时退化为 C，指标应与 C 一致（消融中作为本地对照）。"""
    results, _ = run_all(cfg, fake_completion=ablation_fake_completion)
    for sc_id in SCENARIO_IDS:
        mc, md = results[sc_id]["C"], results[sc_id]["D"]
        assert mc["intent_accuracy"] == md["intent_accuracy"]
        assert mc["clarification_rate"] == md["clarification_rate"]


def test_run_ablation_artifacts_written(cfg, tmp_path):
    """run_ablation 主流程可在临时目录产出 csv/json（不污染仓库 results）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_ablation_tmp", os.path.join(ROOT, "experiments", "run_ablation.py"))
    mod = importlib.util.module_from_spec(spec)
    # 重定向落盘目录到 tmp
    spec.loader.exec_module(mod)
    mod.RESULTS_DIR = str(tmp_path)
    mod.CSV_PATH = os.path.join(str(tmp_path), "ablation_metrics.csv")
    mod.JSON_PATH = os.path.join(str(tmp_path), "ablation_metrics.json")
    rc = mod.main()
    assert rc == 0
    assert os.path.exists(mod.CSV_PATH)
    assert os.path.exists(mod.JSON_PATH)
    # csv 行数 = 表头 + 5×4
    with open(mod.CSV_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert len(lines) == 1 + 5 * 4
