"""识别层验收总入口：一次性跑完 5 项验证门（不依赖 pytest）。

    python recognition/verify.py

覆盖：
    1. 三模型 forward (1,3,90,27) -> (1,60)
    2. train.py --mock 至少一个训练步
    3. predict() 输出通过 tests/contract.validate_gloss_output
    4. tests/test_recognition.py 独立脚本全量断言
    5. 三模型在合成集上的学习曲线（ST-GCN 为重点）
"""

from __future__ import annotations

import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tests.contract import validate_gloss_output  # noqa: E402
from recognition.models import build_model  # noqa: E402
from recognition.predict import predict  # noqa: E402

NUM_CLASSES = 60
NUM_POINTS = 27
PY = sys.executable
OK = []


def gate1_forward():
    print("\n=== 门 1：三模型 forward (1,3,90,27) -> (1,60) ===")
    ok = True
    for name in ["stgcn", "transformer_gcn", "cnn_lstm"]:
        torch.manual_seed(0)
        m = build_model(name, num_classes=NUM_CLASSES).eval()
        x = torch.randn(1, 3, 90, NUM_POINTS)
        with torch.no_grad():
            y = m(x)
        good = tuple(y.shape) == (1, NUM_CLASSES)
        ok = ok and good
        print(f"  {name:16s} forward -> {tuple(y.shape)}  {'OK' if good else 'FAIL'}")
    return ok


def gate2_mock_train():
    print("\n=== 门 2：train.py --mock 训练步 ===")
    r = subprocess.run([PY, "recognition/train.py", "--mock", "--epochs", "2",
                        "--batch-size", "8"], cwd=_ROOT, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return r.returncode == 0


def gate3_predict_contract():
    print("\n=== 门 3：predict() 通过 validate_gloss_output ===")
    ok = True
    seq = np.random.default_rng(7).standard_normal((90, NUM_POINTS, 3)).astype(np.float32)
    for name in ["stgcn", "transformer_gcn", "cnn_lstm"]:
        out = predict(seq, fps=30.0, model_name=name, num_classes=NUM_CLASSES)
        validate_gloss_output(out)
        good = (len(out["gloss_sequence"]) == len(out["confidence"]) == len(out["timestamps"]) == 1)
        ok = ok and good
        print(f"  {name:16s} gloss={out['gloss_sequence'][0]!r} "
              f"conf={out['confidence'][0]:.4f} ts={out['timestamps'][0]}  PASS")
    return ok


def gate4_standalone_tests():
    print("\n=== 门 4：tests/test_recognition.py 独立运行 ===")
    r = subprocess.run([PY, "tests/test_recognition.py"], cwd=_ROOT,
                       capture_output=True, text=True)
    tail = [ln for ln in r.stdout.splitlines() if ln.startswith(("[PASS]", "[FAIL]", "结果", "全部"))]
    print("\n".join("  " + ln for ln in tail) or r.stderr.strip())
    return r.returncode == 0


def gate5_synthetic_learning():
    print("\n=== 门 5：合成集真实训练（ST-GCN 重点 + 两个对比模型） ===")
    outs = {}
    for name, epochs in [("stgcn", 8), ("cnn_lstm", 6), ("transformer_gcn", 6)]:
        r = subprocess.run([PY, "recognition/train.py", "--synthetic", "--model", name,
                            "--epochs", str(epochs), "--batch-size", "32", "--device", "cuda"],
                           cwd=_ROOT, capture_output=True, text=True)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("[train]")]
        print(f"  -- {name} --")
        for ln in lines:
            if "epoch" in ln or "final" in ln:
                print("   " + ln)
        outs[name] = r.returncode == 0
    return all(outs.values())


def main():
    results = {
        "1 forward (1,3,90,27)->(1,60)": gate1_forward(),
        "2 train.py --mock": gate2_mock_train(),
        "3 predict 契约": gate3_predict_contract(),
        "4 独立测试脚本": gate4_standalone_tests(),
        "5 合成集学习曲线": gate5_synthetic_learning(),
    }
    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"  门 {k:34s} {'PASS' if v else 'FAIL'}")
    print("=" * 60)
    print("验收：" + ("全部通过 ✅" if all(results.values()) else "存在失败 ❌"))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
