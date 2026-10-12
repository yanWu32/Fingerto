import { defineStore } from 'pinia'

// 全局状态示例：应用级配置（登录态、全局配置后续在此扩展）。
// 局部状态放在组件内，只有跨页面共享状态才放这里（规范 4.3）。
interface AppState {
  appName: string
  // 模拟环境状态（对应方案中的智能家居 JSON 模拟）
  environment: Record<string, boolean>
}

export const useAppStore = defineStore('app', {
  state: (): AppState => ({
    appName: '手语对话系统',
    environment: {
      light: false,
      air_conditioner: false,
    },
  }),
  actions: {
    toggleDevice(key: keyof AppState['environment']) {
      this.environment[key] = !this.environment[key]
    },
  },
})
