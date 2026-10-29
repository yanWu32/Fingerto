import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

// Vite 配置：开发代理 /api 到后端服务（前后端分离，规避跨域）
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    host: true,
    port: 5173,
    proxy: {
      // 开发环境将 /api 反向代理到后端服务；生产由 Nginx 反向代理。
      // 后端 FastAPI(server.py) 注册的路由本身不带 /api 前缀（/recognition、/pipeline、/health…），
      // 这里把 /api 前缀剥掉再转发（rewrite），保证前端统一以 /api 调用、后端无需重复前缀。
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api/, ''),
      },
    },
  },
})
