# 手语对话系统 · 前端 SPA

基于 LLM 智能体的手语对话系统的前端界面，采用**前后端分离**架构（Vue 3 + Vite + TypeScript）。
后端仅提供 HTTP JSON 接口，前端负责页面渲染、交互与前端路由。

## 技术栈

- **Vue 3**（组合式 API）+ **TypeScript**
- **Vite** 构建 / 开发服务器（代理 `/api` 规避跨域）
- **Pinia** 全局状态、**Vue Router** 前端路由（懒加载）
- **Axios** 统一请求封装（拦截器处理 token / 全局错误）

## 目录结构（对齐前端开发规范 §1.4）

```
src
├── api/          # 接口请求封装
│                 #   request.ts   统一 axios 实例（拦截器处理 token / 全局错误）
│                 #   types.ts     契约类型（对齐 tests/contract.py）
│                 #   mock.ts      mock 占位数据（USE_MOCK 时生效）
│                 #   recognition.ts 识别层接口   POST /recognition
│                 #   agent.ts       智能体层接口 POST /agent
│                 #   pipeline.ts    端到端串联   （真实环境走 /pipeline）
│                 #   demoData.ts   全链路演示的 mock 场景数据
│                 #   demoPipeline.ts 演示编排：逐阶段产出流水线快照
├── components/   # 公共复用组件
│                 #   SkeletonCanvas  27 点骨架 SVG 播放器（渲染真实输入张量）
│                 #   PipelineFlow    流水线阶段进度时间轴
├── views/        # 页面级组件（HomeView / DemoView / AboutView / NotFoundView），对应路由
├── router/       # 前端路由配置（路由懒加载 + 404 catch-all）
├── store/        # 全局状态（Pinia）
├── utils/        # 工具函数
│                 #   env.ts      环境开关（USE_MOCK / API_BASE）
│                 #   format.ts   置信度 / 时间区间格式化
│                 #   skeleton.ts 27 点骨架模型（骨连接、位置映射、姿态插值）
│                 #   skeletonSeq.ts 由 gloss 时间戳合成 (T, 27, 3) 序列
│                 #   sanitize.ts v-html 高亮渲染（先转义再包 <mark>，防 XSS）
├── assets/       # 静态资源（图片、字体）
└── styles/       # 全局样式、CSS 主题变量
```

## 开发

```bash
npm install      # 安装依赖
npm run dev      # 启动开发服务器（默认 http://localhost:5173）
npm run build    # 类型检查 + 生产构建（产物 dist/）
npm run preview  # 预览构建产物
npm run lint     # ESLint 检查 + 自动修复
npm run format   # Prettier 格式化
npm run test     # 单元测试（Vitest，非必须构建流程）
npm run test:watch  # 测试 watch 模式
```

## mock / 真实后端切换

通过环境变量 `VITE_USE_MOCK` 切换（见 `.env.example`，默认 `true`）：

- `VITE_USE_MOCK=true`：使用 `src/api/mock.ts` 内置数据，无需后端即可跑通界面。
- `VITE_USE_MOCK=false`：调用真实后端 `/api/recognition`、`/api/agent`、`/api/pipeline`。

调用方只依赖 `src/api/pipeline.ts` 的 `runPipeline()`，切换时**无需改动页面代码**。

## 接口约定（对齐方案 §4）

- 统一响应结构：`{ "code": 200, "msg": "success", "data": {} }`，由 `request.ts` 统一解包，
  业务代码拿到的泛型 `T` 即 `data` 的类型。
- 识别层 → 智能体层契约：gloss 序列 + 置信度 + 时间戳（类型见 `src/api/types.ts`，
  **唯一事实源为 `../tests/contract.py`**：`task` 必为 string、`clarification` 必为 boolean）。
- 开发环境 `/api/*` 由 Vite 代理到 `http://localhost:8000`；生产由 Nginx 反向代理。

## 页面

| 路由 | 页面 | 说明 |
| --- | --- | --- |
| `/` | HomeView | 系统概览 + 「识别 → 智能体」结果速览 |
| `/demo` | DemoView | **全链路演示**：手语视频 → 文本 → 任务执行的完整流程 |
| `/about` | AboutView | 系统介绍 |
| 其它 | NotFoundView | 404 页 |

### 全链路演示页（`/demo`）

按真实分层依次演示，每一层都展示该层的输入、处理与产出：

1. **感知层**：读取视频 → 提取骨架，渲染形状为 `(T, 27, 3)` 的真实输入张量
   （27 点 = 主手 21 点 + 上半身 6 点，可播放/暂停，点击 gloss 可跳转定位）
2. **切分层**：运动能量粗切分 → 滑动窗口（窗口 30 帧 / 步长 10），时间轴可视化活跃段
3. **识别层**：逐窗口推理 → vote 投票聚合成 gloss 序列，置信度分三档着色；
   可展开查看每个窗口的原始产出
4. **智能体层**：完整展示 LLM 三步提示管线的 System/User Prompt 与输出
   - Step1 低置信消歧 → Step2 Gloss→自然语言 → Step3 意图/任务决策
   - 关键 token（`<lowconf:词:分>`、任务标识）高亮
5. **应用层**：任务 API、执行前后的环境状态变化表

> mock 数据与真实后端行为严格对齐（这是正确性而非美观问题）：
> 低置信词由 `clarify_threshold=0.4` 决定、`<lowconf>` 标注由 `high_threshold=0.75` 触发、
> Step3 的 task 只能取自 `configs/agent.yaml` 的 8 个动作或 `none`。
> 这些约束都有单元测试覆盖（`src/utils/skeletonSeq.test.ts`）。

## 当前进度

工程化目录 + 语义化占位首页（`HomeView`）+ **全链路演示页（`DemoView`）** + 关于页 +
404 页。API 层具备「接后端」能力（mock / real 环境变量切换），配有 23 个单元测试
（契约一致性、27 点骨架张量形状、XSS 转义）。

## 规范遵循

遵循 `../前端页面开发规范.md`：语义化 HTML、结构/样式/行为分离、ES Module 组件化、
`===` 严格相等、接口 `try/catch`、路由懒加载、敏感信息不写死前端、键盘可访问等。
