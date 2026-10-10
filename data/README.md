# 数据层（Data Layer）

手语对话系统的数据管家：提供**临时合成骨架数据集**（立即可用）与 **CSL 真实替代加载器骨架**（数据就位即转），二者接口完全对齐 `tests/contract.py` 的 27 点骨架契约。

## 数据状态

| 数据集 | 状态 | 说明 |
| --- | --- | --- |
| ISW-1000 | ⏳ pending | 申请邮件待用户本人发出，暂未入库。此层不依赖它。 |
| 临时合成集（CSL 风格骨架） | ✅ 已生成 | `data/synthetic/`：60 类中文孤立手语词，每类 36 样本（train 24 / val 6 / test 6），形状 `(N, T, 27, 3)` float32。 |
| CSL 真实替代 | 🔧 骨架就绪 | `data/csl_loader.py`：给定原始 RGB 目录 + 标注 csv 即转 27 点 npz，接口与合成集一致；无数据时 import 不报错。 |
| **WLASL（英文 ASL 对照）** | ⏳ 待用户侧下载 | `data/wlasl_loader.py` + `data/wlasl_pose_extract.py`：英文美式手语视频 → MediaPipe 抽 27 点骨架 → `(T,27,3)` npz。⚠️ **仅作跨语种方法泛化对照基线，不是中文主实验**，gloss 独立存 `data/wlasl/glosses.txt`。代码已就绪；骨架 npz 需用户按 `wlasl_download_guide.md` 在本机下载视频后生成（沙箱网络封了 raw.githubusercontent/huggingface）。 |

## 文件用途

- `data/__init__.py` —— 公共符号导出（`load_dataset`、`SyntheticDataset`、`generate_dataset` 等）。
- `data/synthetic.py` —— 合成骨架生成器：按类索引参数化驱动主手 21 点相对上身 6 点的运动（画圆 / 直线 / 上下摆 / 左右摆 / 8 字 等 10 种运动 × 多频率/半径/相位/中心/手势开合组合），复用 `perception.skeleton.normalize_sequence` 做鼻原点 / 肩宽尺度 / 肩线水平归一化。固定种子 `20261010` 可复现。
- `data/dataset.py` —— 统一加载 API：
  - `load_dataset(split="train", root="data/synthetic") -> (skeletons, labels, classes)`，所有样本经 `check_skeleton_shape` 校验。
  - `class SyntheticDataset(Dataset)`：`__getitem__` 返回 `(skeleton (T,27,3), label int)`。
- `data/csl_loader.py` —— CSL 加载器骨架：`convert_csl(raw_dir, out_dir, fps=30)` 经 MediaPipe Holistic 抽 27 点骨架并落 npz；仅保存骨架不存原始像素（隐私最小化）。无数据 / 无 mediapipe 时 import 不报错，调用时给出清晰提示。
- `data/make_testset.py` —— 命令行入口：`python data/make_testset.py` 重新生成三切分（固定 seed）。

## 合成数据集契约

- npz 字段：`skeletons (N,T,27,3) float32`、`labels (N,) int64`、`classes (60,) str`。
- 骨骼布局：27 点 = 主手 21（HAND_OFFSET=0）+ 上身 6（POSE_OFFSET=21，索引 `[0,11,12,13,14,15]`=鼻/左肩/右肩/左肘/右肘/左腕）。
- 归一化：鼻为原点、肩宽为尺度、绕 Y 轴使肩线水平（见 `perception/skeleton.normalize_sequence`）。
- `classes.txt`：60 个中文 gloss（你/我/他/她/好/谢谢/请/是/不/吃/喝/水/饭/书/笔/车/家/学校/老师/学生/朋友/爱/想/知道/时间/今天/明天/天气/冷/热/大/小/上/下/开/关/灯/门/走/坐/站/说/听/看/做/帮/买/卖/问/答/红/绿/白/黑/一/二/三/四/五/六）。

## 快速使用

```python
import data
skeletons, labels, classes = data.load_dataset("train")   # (N,T,27,3), (N,), list[str]

from data.dataset import SyntheticDataset
ds = SyntheticDataset("train")
skel, label = ds[0]
```

## 重新生成

```bash
cd Fingerto
conda run -n fingerto python data/make_testset.py
```

## CSL 转换（待用户提供目录）

标注 csv 格式假设（每行）：`video_name,gloss`（详见 `csl_loader.py` 顶部注释）。

```bash
conda run -n fingerto python data/csl_loader.py <raw_dir> <out_dir> [fps]
```

## WLASL（英文 ASL 对照基线）⚠️ 非中文主实验

> **语种差异硬警告**：WLASL 是 **英文美式手语（ASL）word-level 视频数据集**，
> 与中文手语（CSL 主实验）在**手势体系、gloss 语种、词汇表**上完全不同。
> 它**只**用于论文中「方法在跨语种（英文 ASL）数据上的泛化性」补充证据，
> **不能**当作中文主实验结果，也**不能**把英文 gloss 混入 `data/synthetic/classes.txt`。

### 状态
- **代码已就绪**：`wlasl_loader.py`（加载）、`wlasl_pose_extract.py`（提取 + 合成自检）。
- **数据待生成**：骨架 npz 不入库（`*.npz` 已被 `.gitignore` 忽略），需用户侧先下载视频。
- **获取方式**：见同目录 [`wlasl_download_guide.md`](wlasl_download_guide.md)。
  因为本 DeepSeek Harness 沙箱封禁 `raw.githubusercontent.com` 与 `huggingface.co`，
  **下载动作必须在你自己的机器（有外网）上完成**，再把 `WLASL_v0.3.json` 与
  `videos/*.mp4` 拷回 `data/wlasl/`。

### 27 点映射要点（与中文 CSL 同源契约）
- 主手 21 点：`HAND_OFFSET=0`，MediaPipe Hands 标准 21 点（优先左手，swap 时取右手）。
- 上身 6 点：`POSE_OFFSET=21`，索引 `[0,11,12,13,14,15]` = 鼻/左肩/右肩/左肘/右肘/左腕。
- **英文数据无「上身左腕」专属对应**：MediaPipe Pose 索引 **15 = LEFT_WRIST** 可直接用，
  与中文布局一致；若该点被遮挡/缺失，对应点填 0（`check_skeleton_shape` 仅校验有限性）。
- 固定帧数 `T=90`（尾帧重复 pad / 截断），复用 `normalize_sequence` 做鼻原点/肩宽尺度/肩线水平归一化。

### 使用流程
```bash
# 1) 本机下载（见 wlasl_download_guide.md）后，沙箱内生成骨架：
conda run -n fingerto python data/wlasl_pose_extract.py \
    --meta data/wlasl/WLASL_v0.3.json --videos data/wlasl/videos --out data/wlasl

# 2) 加载（接口同 load_dataset）：
conda run -n fingerto python -c "from data.wlasl_loader import load_wlasl; \
    sk,lb,gl = load_wlasl('train'); print(sk.shape, len(gl))"

# 3) 合成自检（无需视频/网络/mediapipe，仅 numpy，沙箱可跑）：
conda run -n fingerto python data/wlasl_pose_extract.py --self-check
```

### 产物布局
```
data/wlasl/
├── WLASL_v0.3.json          # 元数据（用户侧下载后拷入，可入库）
├── videos/                  # 下载的 mp4（不入库）
├── wlasl_download_guide.md  # 获取指南
├── wlasl_pose_extract.py    # 提取管线
├── wlasl_loader.py          # 加载器
├── npz/{train,val,test}.npz # 27 点骨架（*.npz 不入库）
└── glosses.txt              # 英文 ASL gloss 列表（独立保存，可入库）
```
