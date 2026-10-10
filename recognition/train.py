"""识别层训练脚本（含 --mock 冒烟训练，不依赖真实 ISW-1000 数据）。

用法：
    python recognition/train.py --mock
    python recognition/train.py --data-root <dir>  # 真实 .npz 目录
    python recognition/train.py --mock --model transformer_gcn --epochs 2

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

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from recognition.dataset import build_dataloader
from recognition.models import build_model


def load_config(path: str) -> dict:
    if yaml is None or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Recognition training (mock-capable)")
    p.add_argument("--config", default=os.path.join(_ROOT, "configs", "recognition.yaml"))
    p.add_argument("--mock", action="store_true", help="在合成数据上冒烟训练")
    p.add_argument("--data-root", default=None, help="真实 .npz 目录")
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
    else:
        num_classes = args.num_classes or num_classes

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "cuda" and not torch.cuda.is_available():
        print("[train] cuda 不可用，回退 cpu")
        device = "cpu"
    device = torch.device(device)

    # 构造 dataloader
    if args.mock or args.data_root is None:
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
        print(f"[train] epoch {epoch}/{epochs}  loss={running_loss / max(n,1):.4f}  "
              f"acc={running_acc / max(n,1):.3f}")

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"[train] 权重已保存：{args.save}")

    print("[train] 冒烟训练完成 ✅")
    return model


if __name__ == "__main__":
    main()
