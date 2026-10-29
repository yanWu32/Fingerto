# Fingerto 后端 API 接口表

> 机器可读版（Swagger / OpenAPI 3.0.3）：`docs/openapi.v3.json` —— 可直接导入 Swagger UI / Apifox / Postman。
> 实现代码：`app/server.py`（FastAPI）。启动：`python -m app.server --port 8000`，或 `python serve.py --backend-fastapi`。
> 统一 baseURL：`/api`（前端禁止写死 IP，由 Vite / `serve.py` 反代）。

## 一、接口总览

| # | 方法 | 路径 | 说明 | 返回 data |
|---|---|---|---|---|
| 1 | GET | `/health` | 依赖就绪探针（前端据 `ready=false` 自动降级 mock） | `HealthStatus` |
| 2 | POST | `/recognition` | 识别层：上传视频 → gloss 序列（同步） | `GlossSchema` |
| 3 | POST | `/pipeline` | 端到端：上传视频 → 识别 + 智能体（同步全量） | `PipelineResult` |
| 4 | POST | `/pipeline/stream` | 端到端分阶段流式（**SSE**，边算边渲染） | `text/event-stream` |
| 5 | GET | `/skeleton/{jobId}` | 按 jobId 二次拉取 `(T,27,3)` 骨架 | `SkeletonData` |
| 6 | POST | `/pipeline/{jobId}/cancel` | 取消正在进行的流式任务 | `CancelAck` |

所有 JSON 响应统一包：`{ "code": 0, "msg": "success", "data": {...} }`。
`code=0` 成功；非 0 见下方错误码表（`msg` 由后端给，前端直接 toast，不自行编造）。

## 二、各接口详情

### 1. GET /health
无鉴权、无参数。

| 字段 | 类型 | 说明 |
|---|---|---|
| `ready` | bool | 三项依赖全就绪才为 true |
| `deps.torch` | bool | 推理框架 |
| `deps.mediapipe` | bool | 骨架抽取 |
| `deps.opencv` | bool | 视频解码 |
| `models_loaded` | string[] | 已缓存的识别模型，如 `["cnn_lstm"]` |

```json
{"code":0,"msg":"success","data":{"ready":true,"deps":{"torch":true,"mediapipe":true,"opencv":true},"models_loaded":["cnn_lstm"]}}
```

### 2. POST /recognition
`multipart/form-data`

| 参数 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `video` | ✅ | file | 仅 mp4/avi/mov/webm，≤50MB（服务端二次校验魔数，不信任前端） |
| `model` | ❌ | string | `stgcn` / `cnn_lstm` / `transformer_gcn`，默认 `cnn_lstm` |
| `device` | ❌ | string | `cpu` / `cuda`，默认 `cpu` |
| `ctx_frames` | ❌ | int | 识别前对切分单元前后扩展的上下文帧数，默认 `20` |

返回 `data`（`GlossSchema`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `gloss_sequence` | string[] | 手语词序列 |
| `confidence` | number[] | 与 gloss 等长，∈[0,1] |
| `timestamps` | [number,number][] | 单位秒，`[start,end]`，`start<=end` |
| `margin_top1_top2` | number[] | **可选**：top1-top2 概率差，判别力指标 |
| `low_confidence` | bool[] | **可选**：`margin<0.2` 标记低判别力 |

```json
{"code":0,"msg":"success","data":{"gloss_sequence":["书记","20","三十"],"confidence":[0.758,0.66,0.738],"timestamps":[[1.0,1.3],[2.2,2.47],[10.27,10.5]],"margin_top1_top2":[0.67,0.51,0.6],"low_confidence":[false,false,false]}}
```

### 3. POST /pipeline
同 `/recognition` 入参，多一个 `useAgent`（bool，默认 true，是否跑智能体层）。
返回 `data` = `{ recognition: GlossSchema, agent: AgentSchema }`。

`AgentSchema`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `natural_language` | string | 翻译后的自然语言 |
| `intent` | string | `chat` / `query` / `command` / `clarify` |
| `task` | string | 任务标识，无任务为 `"none"` |
| `clarification` | bool | 是否需要澄清 |
| `low_confidence_words` | string[] | 低置信词，供前端高亮 |

### 4. POST /pipeline/stream（SSE，动态演示用）
入参同上（含可选 `jobId`，幂等/断点用）。响应 `text/event-stream`，每事件一行 `data: {json}`。
⚠️ axios 不支持 SSE，前端须用 `fetch` + `ReadableStream`。

| type | 时机 | 关键字段 |
|---|---|---|
| `stage` | 阶段开始 | `stage`, `label`（中文） |
| `perception_done` | 骨架就绪 | `shape` `"[670,27,3]"`, `numFrames`, `fps`, `durationSec`, `skeletonUrl` |
| `segmentation_done` | 切分完成 | `segments[{startTime,endTime}]`, `numSegments`, `numWindows` |
| `recognition_window` | 每窗识别（多次） | `window{startTime,endTime,gloss,confidence}` |
| `recognition_done` | 聚合完成 | `recognition`（GlossSchema 全量） |
| `agent_done` | 智能体完成 | `agent`（AgentSchema 全量） |
| `application` | 应用层执行 | `apiPath`, `taskName`, `task`, `controllerStatus`, `changes[]` |
| `done` | 全部完成 | `recognition` + `agent` |
| `error` | 出错 | `code`, `msg` |

实测事件序列（`video_01.mp4`）：
`stage → perception_done → stage → segmentation_done → stage → 15×recognition_window → recognition_done → stage → agent_done → application → done`

### 5. GET /skeleton/{jobId}
返回 `data`：`{ shape, fps, frames, skeleton }`，`skeleton` 为 `(T,27,3)` 行优先数组（`SkeletonCanvas` 可直接渲染）。job 不存在/过期 → 404。

### 6. POST /pipeline/{jobId}/cancel
返回 `data`：`{ jobId, cancelled }`。job 不存在 → 404。

## 三、错误码

| HTTP | code | msg 示例 | 触发 |
|---|---|---|---|
| 400 | 400 | `仅支持 mp4/avi/mov/webm 视频` / `空文件或无法读取` | 文件类型非法或为空 |
| 401 | 401 | `未登录` | 缺/错 token（接口已预留，当前放行） |
| 404 | 404 | `job 不存在` / `job 不存在或骨架已过期` | skeleton / cancel 查不到 job |
| 413 | 413 | `视频超过 50MB 上限` | 超限 |
| 499 | 499 | `已取消` | 流式任务被 cancel |
| 503 | 503 | `后端推理依赖未就绪（缺 torch/mediapipe）` / `感知依赖未就绪或视频无法解析：…` / `识别模型加载失败：…` / `智能体执行失败：…` | 缺依赖、视频损坏、模型/LLM 异常 |

> 设计原则：依赖缺失时返回 **503 + error 事件**，绝不 500 崩；前端据此自动回退 mock 演示。

## 四、实测状态（真实视频 + 真实权重）

| 端点 | 结果 |
|---|---|
| `/health` | ✅ `ready:true`，deps 全 true |
| `/recognition` | ✅ HTTP 200，15 词 gloss + margin + low_confidence（video_01.mp4，cnn_lstm_100g） |
| `/pipeline/stream` | ✅ SSE 25 事件，顺序正确 |
| `/skeleton/{jobId}` | ✅ 返回 `(T,27,3)`；缺失 → 404 |
| `/pipeline/{jobId}/cancel` | ✅ 取消标志位；缺失 → 404 |

性能提示：CPU 上单视频约 **100s**（MediaPipe 抽 670 帧 + 15 窗 CNN-LSTM），前端须带 loading；SSE 流式正是为此设计（边算边渲染）。可加 `--device cuda` 加速。
