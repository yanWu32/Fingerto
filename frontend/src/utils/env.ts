// 运行环境开关：通过 VITE_USE_MOCK 切换 mock / 真实后端（规范 1.3 数据无关先行）。
// 默认开发态使用 mock 数据；接后端时把 .env 的 VITE_USE_MOCK 置为 false 即可。
export const USE_MOCK: boolean = (import.meta.env.VITE_USE_MOCK ?? 'true') !== 'false'

/** 后端服务基址（生产由 Nginx 反向代理，见 vite.config.ts /api） */
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? '/api'
