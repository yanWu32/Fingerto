/// <reference types="vite/client" />

// 让 TS 识别 .vue 单文件组件
declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<Record<string, unknown>, Record<string, unknown>, unknown>
  export default component
}

// 自定义环境变量类型（Vite 注入）
interface ImportMetaEnv {
  /** 是否使用 mock 数据（默认 true）。置为 'false' 即切换为真实后端接口。 */
  readonly VITE_USE_MOCK?: string
  /** 后端服务地址前缀（开发态由 Vite 代理 /api，生产由 Nginx 代理）。 */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
