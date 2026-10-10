"""TASK-007 WLASL 数据层验收测试（R-data）。

验证：
  - 合成自检产出的 27 点映射 / fix_length / 归一化结果形状为 (T,27,3)、有限、通过 check_skeleton_shape；
  - load_wlasl 接口签名/契约正确（无真实数据时 import 不报错，调用时给清晰提示）；
  - wlasl_pose_extract / wlasl_loader 模块可独立 import 且语法正确；
  - glosses 非空（由合成自检构造的假类表模拟）、可迭代。

本测试**不依赖真实视频 / 网络 / pytest**：既可用 `pytest tests/test_wlasl.py`
收集，也可 `python tests/test_wlasl.py` 直接运行（等价断言）。
"""
import os
import sys
from pathlib import Path

# 把仓库根加入 sys.path，支持 `python tests/test_wlasl.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from tests.contract import (
    check_skeleton_shape,
    NUM_POINTS,
    SKELETON_RANK,
    POSE_LANDMARK_IDX,
)
from data.wlasl_pose_extract import (
    map_mediapipe_to_27,
    fix_length,
    _synthetic_self_check,
    FIXED_T,
)


# ---------------------------------------------------------------------------
# 1. 27 点映射逻辑（用 numpy 模拟 MediaPipe 输出）
# ---------------------------------------------------------------------------
def test_map_mediapipe_to_27_shape_and_pose_mapping():
    left = np.random.default_rng(1).normal(0.5, 0.1, size=(21, 3)).astype(np.float32)
    right = np.random.default_rng(2).normal(0.5, 0.1, size=(21, 3)).astype(np.float32)
    pose = np.random.default_rng(3).normal(0.5, 0.1, size=(33, 3)).astype(np.float32)

    skel, vis = map_mediapipe_to_27(left, right, pose, swap=True)
    assert skel.shape == (NUM_POINTS, SKELETON_RANK)
    assert vis.shape == (NUM_POINTS,)
    assert np.all(np.isfinite(skel))

    # swap=True 时主手取右手（索引 5 应等于右手第 5 点）
    assert np.allclose(skel[5], right[5])
    # Pose 左腕(15) 必须映射到索引 26
    assert POSE_LANDMARK_IDX[5] == 15
    assert np.allclose(skel[26], pose[15])
    # 鼻(0) -> 索引 21
    assert np.allclose(skel[21], pose[0])


def test_map_missing_hand_fills_zero():
    # 左手/右手都 None -> 手点全 0，但 vis 为 0
    pose = np.random.default_rng(4).normal(0.5, 0.1, size=(33, 3)).astype(np.float32)
    skel, vis = map_mediapipe_to_27(None, None, pose, swap=True)
    assert np.all(skel[0:21] == 0.0)
    assert np.all(vis[0:21] == 0.0)
    assert np.allclose(skel[21:], pose[POSE_LANDMARK_IDX])


# ---------------------------------------------------------------------------
# 2. fix_length 截断 / pad
# ---------------------------------------------------------------------------
def test_fix_length_truncate_and_pad():
    seq = np.random.default_rng(5).normal(size=(60, NUM_POINTS, SKELETON_RANK)).astype(np.float32)
    out = fix_length(seq, T=90)
    assert out.shape == (90, NUM_POINTS, SKELETON_RANK)
    # 尾帧重复 pad
    assert np.allclose(out[60:], seq[-1:])

    seq2 = np.random.default_rng(6).normal(size=(120, NUM_POINTS, SKELETON_RANK)).astype(np.float32)
    out2 = fix_length(seq2, T=90)
    assert out2.shape[0] == 90


# ---------------------------------------------------------------------------
# 3. 合成自检（核心契约）
# ---------------------------------------------------------------------------
def test_synthetic_self_check_contract():
    res = _synthetic_self_check(T=FIXED_T)
    assert res["final_shape"] == [FIXED_T, NUM_POINTS, SKELETON_RANK]
    assert res["contract_ok"] is True
    assert res["all_finite"] is True


# ---------------------------------------------------------------------------
# 4. wlasl_loader 接口与契约校验逻辑（用合成 npz 临时文件验证）
# ---------------------------------------------------------------------------
def _make_fake_wlasl_npz(tmp_root: Path):
    """写一个最小合成 WLASL npz + glosses.txt，供 load_wlasl 读取验证。"""
    npz_dir = tmp_root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    T = FIXED_T
    skeletons = rng.normal(size=(4, T, NUM_POINTS, SKELETON_RANK)).astype(np.float32)
    labels = np.array([0, 1, 2, 1], dtype=np.int64)
    glosses = ["hello", "wish", "thankyou"]
    np.savez(
        npz_dir / "train.npz",
        skeletons=skeletons,
        labels=labels,
        classes=np.array(glosses, dtype=object),
    )
    (tmp_root / "glosses.txt").write_text("\n".join(glosses) + "\n", encoding="utf-8")
    return tmp_root


def test_load_wlasl_shape_glosses_iterable(tmp_path=None):
    if tmp_path is None:
        # 不依赖 pytest fixture，用仓库内可写临时目录（避开受限的 AppData 临时区）
        import shutil
        tmp_root = REPO_ROOT / "data" / ".tmp_wlasl_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)
    else:
        tmp_root = Path(tmp_path) / "wlasl"
    _make_fake_wlasl_npz(tmp_root)

    from data.wlasl_loader import load_wlasl, get_glosses
    skeletons, labels, glosses = load_wlasl("train", root=str(tmp_root))

    # 形状 (T,27,3) 与契约
    assert skeletons.ndim == 4
    assert skeletons.shape[2:] == (NUM_POINTS, SKELETON_RANK)
    assert skeletons.dtype == np.float32
    assert labels.shape[0] == skeletons.shape[0]
    # glosses 非空 + 可迭代
    assert isinstance(glosses, list)
    assert len(glosses) > 0
    assert all(isinstance(g, str) for g in glosses)
    assert len(list(glosses)) == len(glosses)
    # 每个样本经 check_skeleton_shape 校验
    for i in range(skeletons.shape[0]):
        assert check_skeleton_shape(skeletons[i]), f"样本 {i} 契约失败"
        assert skeletons[i].shape == (FIXED_T, NUM_POINTS, SKELETON_RANK)
    # labels 在 gloss 范围内
    assert np.all(labels >= 0) and np.all(labels < len(glosses))
    # get_glosses 兜底可读
    assert get_glosses(str(tmp_root)) == glosses


def test_load_wlasl_missing_file_raises():
    from data.wlasl_loader import load_wlasl
    # pytest 可能未安装：用 try/except 手动断言，避免 import 失败
    raised = False
    try:
        load_wlasl("train", root="data/wlasl_nonexistent_xyz")
    except FileNotFoundError:
        raised = True
    assert raised, "缺失 npz 应抛 FileNotFoundError"


# ---------------------------------------------------------------------------
# 5. 直接运行的等价断言（pytest 未装时可用）
# ---------------------------------------------------------------------------
def _run_all() -> int:
    failures = []

    def check(name, fn):
        try:
            fn()
            print(f"  ✅ {name}")
        except Exception as e:  # noqa: BLE001
            failures.append((name, repr(e)))
            print(f"  ❌ {name}: {e}")

    check("map_mediapipe_to_27 形状/姿态映射", test_map_mediapipe_to_27_shape_and_pose_mapping)
    check("map 缺失手填 0", test_map_missing_hand_fills_zero)
    check("fix_length 截断/pad", test_fix_length_truncate_and_pad)
    check("合成自检契约 (T,27,3)/有限/check_skeleton_shape", test_synthetic_self_check_contract)
    check("load_wlasl 形状/glosses非空/可迭代", test_load_wlasl_shape_glosses_iterable)
    check("load_wlasl 缺失文件抛错", test_load_wlasl_missing_file_raises)

    print("\n" + ("全部通过 ✅" if not failures else f"{len(failures)} 项失败 ❌"))
    for name, err in failures:
        print(f"  - {name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
