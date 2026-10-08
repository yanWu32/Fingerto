# Fingerto — 基于LLM智能体的手语对话系统

本科毕业设计 · 福州理工学院 计算与信息科学学院 · 计算机科学与技术
作者：吴家骏（学号 2303116096）

> 一个以 LLM 智能体为主体、以手语识别为感知输入的手语对话系统。
> 手语识别模块只负责把手部关键点转换为 gloss（手语词汇标注）序列，
> 所有意图推理、跨语法翻译、对话管理与任务执行均由智能体完成。

---

## 一、项目定位与边界

- **系统形态**：学术验证型，不要求真实硬件落地；环境状态由 JSON / 模拟器提供。
- **交互边界**：面向**词级 / 短语级**手语对话，假设手势之间存在自然停顿；
  **不声称**处理任意流畅的连续手语翻译。
- **核心范式转换**：从传统的「手势 → 命令」映射，升级为
  「**gloss 序列 + 上下文 → 意图 → 对话 / 任务**」的推理过程。
- **核心设计**：识别层**不直接输出命令**，只输出 gloss 序列与置信度；
  决策权 100% 交给智能体。

## 二、系统架构（四层解耦）

```
┌──────────────────────────────────────────────────────┐
│ 感知层（非智能体）                                     │
│  OpenCV 采集 → 预处理 → MediaPipe → 27 点骨架 (T,27,3) │
│  → 运动能量粗切分 + 滑动窗口细识别 → 候选手势单元       │
└───────────────────────┬──────────────────────────────┘
                        │ 候选手势单元（骨架片段）
┌───────────────────────▼──────────────────────────────┐
│ 识别层                                                 │
│  骨骼序列 (T,V,3) → 分类模型 → gloss + 置信度 + 时间戳  │
└───────────────────────┬──────────────────────────────┘
                        │ {gloss_sequence, confidence, timestamps}
┌───────────────────────▼──────────────────────────────┐
│ 智能体层（核心）                                        │
│  Step1 确认(低置信消歧) → Step2 翻译 → Step3 决策/任务  │
│  维护对话状态 / 环境状态 / 用户偏好 / 待确认意图          │
└───────────────────────┬──────────────────────────────┘
                        │ {natural_language, intent, task, clarification}
┌───────────────────────▼──────────────────────────────┐
│ 应用层                                                 │
│  Gradio 界面（摄像头 / 可视化 / 对话历史 / FPS）        │
│  场景：信息查询 + 模拟智能家居控制                       │
└──────────────────────────────────────────────────────┘
```

### 层间数据契约

```json
// 识别层 → 智能体层
{"gloss_sequence": ["我", "冷", "窗", "关"],
 "confidence":    [0.92, 0.58, 0.91, 0.85],
 "timestamps":    [[0.0, 0.4], [0.5, 0.9], [1.0, 1.4], [1.5, 1.9]]}

// 智能体层 → 应用层
{"natural_language": "我觉得冷，能关一下窗户吗？",
 "intent": "command",
 "task": {"action": "close_window"},
 "clarification": null,
 "low_confidence_words": ["冷"]}
```

## 三、分层方案要点

| 层 | 关键方案 |
|---|---|
| **感知层** | OpenCV + MediaPipe 提取双手各 21 点 + 上半身 6 点 = **27 点骨架**；缺失帧时序插值 + 运动学约束；随机旋转增强 |
| **手势切分** | **运动能量粗切分**（手腕速度低于阈值持续 N 帧判边界）+ **滑动窗口细识别**（30 帧窗口 + 时序平滑 / Viterbi）组合 |
| **识别层** | 基线锁 **ST-GCN**（时空图卷积）；对比 Transformer+GCN、轻量 CNN+LSTM；输出 Top-1 / Top-5 + 延迟 |
| **智能体层** | 三步提示管线（确认 → 翻译 → 决策）；对话状态管理；**置信度路由消歧**（<0.6 澄清）；云端 LLM API 主用 + 本地降级 + 高频词缓存 |
| **应用层** | Gradio；信息查询与模拟智能家居两类场景，无需真实硬件 |

## 四、技术栈

| 层 | 主选 | 备选 |
|---|---|---|
| 感知 | OpenCV 4.x + MediaPipe | RTMPose |
| 识别 | PyTorch + ST-GCN | Transformer+GCN / CNN+LSTM |
| 智能体 | 云端 LLM API（DeepSeek 主 / Qwen 备） | Qwen2.5-3B 本地 (Ollama) |
| 界面 | Gradio | PyQt |
| 环境模拟 | Python dict / JSON | 智能家居模拟器 |
| 工程 | Git + Anaconda | venv |

## 五、工程结构

```
Fingerto/
├── perception/      # 视频采集 + MediaPipe 关键点提取 + 归一化
├── segmentation/    # 运动能量检测 + 滑动窗口切分
├── recognition/     # datasets / models(stgcn…) / train / eval / infer
├── agent/           # llm_client / prompt_pipeline / dialogue_state / disambiguation / intent
├── app/             # gradio_app.py
├── data/            # 数据集 + processed + testset（不入库）
├── experiments/     # build_testset / run_ablation / metrics
├── configs/         # recognition.yaml / agent.yaml / experiment.yaml
├── docs/            # 数据报告 / 实验记录 / 论文素材
├── tools/           # check_env.py 等辅助脚本
└── requirements.txt
```

## 六、运行环境

推荐使用 Anaconda 环境 `pytorch`：

- 解释器：`E:\Program\anaconda\anaconda3\envs\pytorch\python.exe`（Python 3.8.20）
- 已具备：**torch 2.4.0+cu121**、torchvision 0.19.0、opencv-python 4.13.0、
  numpy / pandas / scikit-learn / matplotlib / tqdm / pyyaml / pillow / requests
- GPU：**NVIDIA GeForce RTX 4060 Laptop GPU**，CUDA 可用
- 需补装：`mediapipe`、`gradio`、`openai`

环境自检：

```bash
python tools/check_env.py
```

## 七、数据集策略

| 预案 | 数据集 | 说明 |
|---|---|---|
| A（首选） | **ISW-1000** | 约 1000 中文手语词，仅用其骨骼模态；正在申请中 |
| B（降级） | **CSL** | 合肥工大中文手语孤立词集，含 RGB / 深度 / 骨骼 |
| C（保底） | 自采最小集 | 本人录制高频词，MediaPipe 提骨架，跑通端到端管线 |

> 数据集待定**不阻塞开发**：感知层、切分、智能体层（可用 mock gloss）、
> 应用层与消融评测脚本均可先行实现。

## 八、实验设计（消融）

| 系统 | 组成 | 验证目标 |
|---|---|---|
| System A | 识别 + 规则模板回复 | 基线 |
| System B | 识别 + LLM 翻译（无对话状态） | LLM 翻译增量 |
| System C | 识别 + LLM 翻译 + 对话状态管理 | 对话管理增量 |
| System D（可选） | 微调小模型替代通用 LLM | 延迟–精度权衡 |

**场景**：孤立词翻译 / 低置信度消歧 / 多轮指代消解 / 用户纠正 / 多义手势消歧
**指标**：BLEU / ROUGE、意图准确率、对话完成率、澄清触发率、端到端延迟

## 九、进度

- [x] 工程脚手架（目录结构 / requirements / .gitignore / 环境自检脚本）
- [x] 方案确定书
- [ ] 感知层：MediaPipe 提取 27 点骨架
- [ ] 智能体层：LLM 三步提示管线（可先用 mock gloss）
- [ ] 手势切分：运动能量 + 滑动窗口
- [ ] 识别层：ST-GCN 基线（依赖数据集）
- [ ] 应用层：Gradio 端到端 Demo
- [ ] 消融实验与论文撰写

---

详细方案见毕设目录下的 `方案确定书.md` 与 `实现方案-全流程.md`。
