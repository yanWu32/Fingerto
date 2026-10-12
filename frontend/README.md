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
├── api/          # 接口请求封装（request.ts 统一实例 + mock.ts 占位数据）
├── components/   # 公共复用组件（AppCard.vue）
├── views/        # 页面级组件（HomeView / AboutView），对应路由
├── router/       # 前端路由配置（路由懒加载）
├── store/        # 全局状态（Pinia）
├── utils/        # 工具函数（日期、置信度、时间格式化）
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
```

## 接口约定（对齐方案 §4）

- 统一响应结构：`{ "code": 200, "msg": "success", "data": {} }`
- 识别层 → 智能体层契约：gloss 序列 + 置信度 + 时间戳（见 `src/api/mock.ts`）
- 开发环境 `/api/*` 由 Vite 代理到 `http://localhost:8000`；生产由 Nginx 反向代理。

## 当前进度

首版骨架：工程化目录 + 语义化占位首页（首页用 mock 数据演示「识别 → 智能体」链路）+
关于页。后续按需求细化对话演示、摄像头可视化、模拟智能家居等页面。

## 规范遵循

遵循 `../前端页面开发规范.md`：语义化 HTML、结构/样式/行为分离、ES Module 组件化、
`===` 严格相等、接口 `try/catch`、路由懒加载、敏感信息不写死前端、键盘可访问等。
