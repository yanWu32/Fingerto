# WLASL100 真实数据训练结果（跨语种泛化对照实验）

> 数据集：WLASL100（英文美式手语 ASL），用户已下载 665/2038 个视频到
> `E:\Program\datasets\WLASL100`。本机现实有 665 个视频对应 665 个 instance，
> 覆盖 100 个 gloss。视频未裁到 frame_start~frame_end（fps=25）精确片段，
> 抽取管线按 frame_end=-1 当「整段」哨兵处理。
>
> ⚠️ **性质说明**：WLASL 是英文 ASL，**不是中文手语**。本实验仅作
> 「方法跨语种泛化」对照基线，**不能替代中文主实验**。论文主实验仍以
> ISW-1000 / CSL（中文、申请制）为准。

## 数据规模（抽骨架后，27 点布局）

| 切分 | 样本数 | 形状 |
|---|---|---|
| train | 538 | (538, 90, 27, 3) |
| val | 58 | (58, 90, 27, 3) |
| test | 69 | (69, 90, 27, 3) |
| **合计** | **665** | 100 类（~5.4 样本/类） |

跳过 = 0。抽取脚本：`data/wlasl_pose_extract.py::build_dataset`。
落盘：`data/wlasl100_real/npz/{train,val,test}.npz`（被 .gitignore 忽略，不入库）。

## 三模型真实训练结果（60 epoch，batch=16，CUDA）

| 模型 | train_acc | val_acc (top5) | test_acc (top5) | 备注 |
|---|---|---|---|---|
| ST-GCN | 0.351 | 0.069 (0.155) | 0.043 (0.116) | 增强后 train↑ 仍强过拟合 |
| Transformer-GCN | 0.020 | 0.017 (0.017) | 0.000 (0.029) | **未收敛**，loss 卡在 ~4.57 |
| CNN-LSTM | 0.907 | 0.086 (0.207) | 0.101 (0.232) | train 收敛但泛化差 |

随机基线（100 类）= 1% top-1 / ~1% top-5。

## 关键发现与归因

1. **严重过拟合，泛化极差**：即便 CNN-LSTM train 冲到 90.7%、ST-GCN 到 35%，
   val/test 仍只在 7~10%（≈ 随机）。根因是**真实样本太稀疏**：
   - 665 个视频 / 100 类 ≈ **5.4 样本/类**；
   - 且 WLASL 以 YouTube 视频为主，单视频含大量非手势帧（已按整段抽，噪声大）；
   - 手语骨架对背景、拍摄角度、光照极敏感。
2. **Transformer-GCN 在真实数据上完全不收敛**（loss 卡在初始化 ~4.57，
   与合成集上 0.828/0.842 形成强烈反差）。说明该结构对数据分布/量级更敏感，
   需更强正则（dropout、学习率 warmup、更长训练）或数据量，当前不可用。
3. **结论对论文的正向支撑**：这恰恰论证了「为什么需要 ISW-1000（~10k 样本、
   干净拍摄、中文）」——稀疏英文视频无法训出可用识别器，**外部效度受数据规模制约**，
   与开题报告「数据集是瓶颈」的判断一致。

## 训练入口（复现）

```bash
# 需 fingerto 环境（torch+cu121, mediapipe 0.10.14）
PYTHONPATH=<repo>/Fingerto python -m recognition.train \
  --wlasl-root <repo>/Fingerto/data/wlasl100_real \
  --model {stgcn|cnn_lstm|transformer_gcn} \
  --epochs 60 --batch-size 16 --device auto \
  --save checkpoints/wlasl_<model>.pt
```

数据增强（`WlaslDataset`）：训练集随机关节抖动(σ=0.01) + 时间循环平移(±8帧)。
验证/测试集关闭增强。

## 待办

- [ ] 申请 ISW-1000 / CSL（中文、申请制）→ 用同管线抽 27 点 → 复跑本表，
      预期泛化指标将显著高于 WLASL100 的 ~10%。
- [ ] Transformer-GCN 真实数据调参（warmup + dropout + 更长训练）后复测。
- [ ] 论文「识别层有效性」章节：以 WLASL100 对照说明数据稀疏瓶颈，
      以 ISW-1000/CSL 真实中文指标作主结论。
