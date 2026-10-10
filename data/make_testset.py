"""命令行入口：重新生成合成手语骨架数据集（train/val/test 三切分）。

用法:
    cd Fingerto
    conda run -n fingerto python data/make_testset.py

固定种子 SEED=20261010，保证可复现；产物在 data/synthetic/ 下：
    train.npz / val.npz / test.npz / classes.txt
npz 内容: skeletons (N,T,27,3) / labels (N,) / classes (60,)
"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许 `python data/make_testset.py` 直接运行（把仓库根加入 sys.path）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.synthetic import generate_dataset, SEED, SPLITS


def main():
    here = Path(__file__).resolve().parent
    out_root = here / "synthetic"
    info = generate_dataset(out_root, seed=SEED, splits=SPLITS)
    for split, (sh, n) in info.items():
        print(f"[make_testset] {split}: {n} 样本, skeletons{sh}")
    print(f"[make_testset] 完成，已写入 {out_root}")


if __name__ == "__main__":
    main()
