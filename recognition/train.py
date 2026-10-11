"""识别层训练脚本（含 --mock 冒烟训练，不依赖真实 ISW-1000 数据）。

用法：
    python recognition/train.py --mock
    python recognition/train.py --data-root <dir>  # 真实 .npz 目录
    python recognition/train.py --mock --model transformer_gcn --epochs 2
    python recognition/train.py --synthetic --model stgcn --epochs 8  # 合成集真训

参数从 configs/recognition.yaml 读取；--num-classes / --model / --epochs 等可覆盖。
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from recognition.dataset import build_dataloader
from recognition.models import build_model


def _synthetic_loaders(root: str, batch_size: int, num_workers: int):
    """用 data/dataset.py::load_dataset 读合成集（train/val/test 三个切分）。

    返回 (loaders dict, classes list)。不动 data/ 层，仅只读调用。
    """
    from data.dataset import load_dataset

    loaders = {}
    classes = None
    for split in ("train", "val", "test"):
        sk, lb, cls = load_dataset(split, root=root)
        classes = cls
        ds = TensorDataset(torch.from_numpy(sk), torch.from_numpy(lb))
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers,
        )
    return loaders, classes


@torch.no_grad()
def _evaluate(model, loader, device):
    """在给定 loader 上算 top-1 / top-5 准确率与平均 loss。"""
    model.eval()
    crit = nn.CrossEntropyLoss()
    total, correct, correct5, loss_sum = 0, 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_sum += crit(logits, y).item() * x.size(0)
            correct += (logits.argmax(1) == y).float().sum().item()
            _, top5 = logits.topk(5, dim=1)
            correct5 += top5.eq(y.view(-1, 1)).sum().item()
            total += x.size(0)
    model.train()
    return loss_sum / max(total, 1), correct / max(total, 1), correct5 / max(total, 1)


def load_config(path: str) -> dict:
    if yaml is None or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Recognition training (mock-capable)")
    p.add_argument("--config", default=os.path.join(_ROOT, "configs", "recognition.yaml"))
    p.add_argument("--mock", action="store_true", help="在合成数据上冒烟训练")
    p.add_argument("--synthetic", action="store_true",
                   help="在 data/synthetic 合成数据集上做真实训练（含 val/test 评估）")
    p.add_argument("--data-root", default=None, help="真实 .npz 目录（单文件 skeleton/label 格式）")
    p.add_argument("--wlasl-root", default=None,
                   help="WLASL 多样本 npz 目录（含 npz/{train,val,test}.npz 与 glosses.txt）；"
                        "真实英文 ASL 对照实验入口")
    p.add_argument("--model", default=None, choices=["stgcn", "transformer_gcn", "cnn_lstm"])
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save", default=None, help="训练后保存权重路径")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})

    model_name = args.model or model_cfg.get("name", "stgcn")
    num_classes = args.num_classes or model_cfg.get("num_classes", 1000)
    epochs = args.epochs or train_cfg.get("epochs", 100)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    lr = args.lr or train_cfg.get("lr", 1e-3)
    max_frames = args.max_frames or data_cfg.get("max_frames", 120)
    pad_mode = data_cfg.get("pad_mode", "last")
    device = args.device or train_cfg.get("device", "cpu")
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    num_workers = args.num_workers

    # mock 模式：占位 60 类（与 R-data 并行生成的 60 个中文 gloss 对齐）
    if args.mock:
        num_classes = args.num_classes or 60
    elif args.synthetic:
        num_classes = args.num_classes or num_classes  # 真实类数由 loader 覆盖
    else:
        num_classes = args.num_classes or num_classes

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device in (None, "", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("[train] cuda 不可用，回退 cpu")
        device = "cpu"
    print(f"[train] device={device}")
    device = torch.device(device)

    # 构造 dataloader
    eval_loaders = {}
    if args.synthetic:
        syn_root = args.data_root or os.path.join(_ROOT, "data", "synthetic")
        print(f"[train] synthetic 模式：读取 {syn_root}")
        loaders, classes = _synthetic_loaders(syn_root, batch_size, num_workers)
        loader = loaders["train"]
        eval_loaders = {"val": loaders["val"], "test": loaders["test"]}
        num_classes = len(classes)
        print(f"[train] train={len(loaders['train'].dataset)}  "
              f"val={len(loaders['val'].dataset)}  test={len(loaders['test'].dataset)}")
    elif args.wlasl_root is not None:
        from recognition.dataset import build_wlasl_dataloader
        print(f"[train] WLASL 真实数据模式：{args.wlasl_root}")
        loaders, classes = build_wlasl_dataloader(
            args.wlasl_root, batch_size=batch_size, max_frames=max_frames,
            pad_mode=pad_mode, num_workers=num_workers,
        )
        loader = loaders.get("train")
        eval_loaders = {k: v for k, v in loaders.items() if k != "train"}
        num_classes = len(classes)
        print(f"[train] train={len(loader.dataset) if loader else 0}  "
              f"val={len(eval_loaders.get('val', []) and eval_loaders['val'].dataset or [])}  "
              f"test={len(eval_loaders.get('test', []) and eval_loaders['test'].dataset or [])}  "
              f"类数={num_classes}")
    elif args.mock or args.data_root is None:
        print(f"[train] mock 模式：合成数据，num_classes={num_classes}")
        loader, classes = build_dataloader(
            root=None, mock_classes=num_classes, batch_size=batch_size,
            max_frames=max_frames, pad_mode=pad_mode, mock_size=max(64, batch_size * 4),
            num_workers=num_workers, rng_seed=seed,
        )
    else:
        print(f"[train] 从 {args.data_root} 加载真实数据")
        loader, classes = build_dataloader(
            root=args.data_root, batch_size=batch_size,
            max_frames=max_frames, pad_mode=pad_mode, num_workers=num_workers,
        )
        num_classes = len(classes)

    model = build_model(model_name, num_classes=num_classes)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model={model_name}  params={n_params}  classes={num_classes}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=train_cfg.get("weight_decay", 1e-4))

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss, running_acc, n = 0.0, 0.0, 0
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            running_acc += (logits.argmax(1) == y).float().sum().item()
            n += x.size(0)
        train_acc = running_acc / max(n, 1)
        line = (f"[train] epoch {epoch}/{epochs}  loss={running_loss / max(n,1):.4f}  "
                f"acc={train_acc:.3f}")
        for split, el in eval_loaders.items():
            vloss, vacc, vacc5 = _evaluate(model, el, device)
            line += f"  {split}_acc={vacc:.3f}(top5={vacc5:.3f})"
        print(line)

    if eval_loaders:
        for split, el in eval_loaders.items():
            vloss, vacc, vacc5 = _evaluate(model, el, device)
            print(f"[train] final {split}: loss={vloss:.4f}  acc={vacc:.3f}  top5={vacc5:.3f}")

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"[train] 权重已保存：{args.save}")

    print("[train] 冒烟训练完成 ✅")
    return model


if __name__ == "__main__":
    main()
