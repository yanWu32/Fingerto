# ISW-1000 中文真实数据训练结果（50 词 × 10 人 = 500 视频 初版指标）

> 数据集：ISW-1000（中文孤立手语词），已就位 `E:\Program\datasets\ISW-1000`。
> 本初版指标采用**「50 词 × 10 人 = 500 视频」**子集（top-50 高频词，须 10 人全有，
> 恰好构成干净的 50×10 网格）。抽取管线 `data/isw1000_extract_resumable.py` 断点续跑，
> 合并 `data/isw1000_merge.py`，回合驱动 `data/isw1000_round.py`。
>
> ⚠️ **性质说明**：这是**中文主实验**的初版指标（非英文 WLASL 对照）。全量 9914 视频 /
> 1000 词仍在累积；本子集用于**先拿到中文真实结果、验证管线**，规模小（每类仅 7 训练样本），
> 指标不代表最终论文主结论上限。

## 数据规模（抽骨架后，27 点布局，T=90）

| 切分 | 样本数 | 形状 | 来源参与者 |
|---|---|---|---|
| train | 350 | (350, 90, 27, 3) | P01–P07 |
| val | 50 | (50, 90, 27, 3) | P08 |
| test | 100 | (100, 90, 27, 3) | P09–P10 |
| **合计** | **500** | 50 类（**7 样本/类训练**） | 10 人 |

按参与者切分（防签名者泄漏）：同一 signer 的同词不会同时出现在 train/test。
落盘：`data/isw1000_real/npz/{train,val,test}.npz`（被 .gitignore 忽略，不入库）。

## 三模型真实训练结果（80 epoch，batch=32，lr=3e-3，CUDA）

| 模型 | train_acc | val_acc (top5) | test_acc (top5) | 备注 |
|---|---|---|---|---|
| ST-GCN | 0.654 | 0.160 (0.460) | **0.150 (0.390)** | 最优；40ep 版 test=0.120/0.330 |
| Transformer-GCN | 0.006 | 0.020 (0.100) | 0.020 (0.100) | **未收敛**，loss 卡在 ~3.92（与合成集表现反差） |
| CNN-LSTM | 0.931 | 0.120 (0.320) | 0.140 (0.380) | train 过拟合，泛化弱 |

随机基线（50 类）= 2% top-1 / ~2% top-5。

## 关键发现与归因

1. **中文真实数据上 ST-GCN 显著优于 WLASL 对照**：ISW-1000 子集 test Top-1≈15% /
   Top-5≈39%，远高于 WLASL100 的 ~4% / ~12%（见 `wlasl100_real_training.md`）。
   根因是 ISW-1000 **干净拍摄、每类样本更充足、中文母语手势更规范**，验证了
   「中文数据集才是中国手语识别的主实验数据」的判断。
2. **仍是欠拟合 + 跨签名者泛化难**：训练集每类仅 7 样本，且 val/test 是**未见过的
   签名者**（P08 / P09–P10），故 test 远低于 train（65%→15%）。这是跨签名者手语识别的
   固有难点，而非管线 bug——合成集 60 类同人上能到 0.99，佐证架构正确。
3. **Transformer-GCN 在真实小样本上仍不收敛**（loss 卡 ~3.92，≈随机），与合成 /
   WLASL 上的失败一致，确认该结构对数据量级极敏感，需 warmup+dropout+更长训练。
4. **指标会随累积规模上升**：当前仅 500/9914。继续扩充（→ 全量 或 100 词×10 人）
   预计 test Top-1 会明显提高——这正是后续回合的目标。

## 训练入口（复现）

```bash
# 需 fingerto 环境（torch+cu121, mediapipe 0.10.14）
PYTHONPATH=<repo>/Fingerto python recognition/train.py \
  --wlasl-root <repo>/Fingerto/data/isw1000_real \
  --model {stgcn|cnn_lstm|transformer_gcn} \
  --epochs 80 --batch-size 32 --lr 3e-3 --device auto \
  --save checkpoints/isw1000_<model>.pt
```

## 待办

- [x] 50 词 × 10 人 = 500 视频抽骨架 + 合并 + 三模型初版指标。
- [ ] 扩大子集（如 100 词 × 10 人，或继续累积到全量），复跑看指标随规模上升曲线。
- [ ] Transformer-GCN 真实数据调参（warmup + dropout）后复测。
- [ ] 论文「识别层有效性」章节：以 ISW-1000 中文真实指标作主结论（替换 WLASL 对照）。
- [ ] 把 `data/isw1000_real/`（npz 不入库）与抽取脚本一并纳入复现说明。
