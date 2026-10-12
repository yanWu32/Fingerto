import axios, { type AxiosInstance, type AxiosResponse } from 'axios'

// 统一响应结构（规范 1.2/1.3）：后端返回 { code, msg, data }
export interface ApiResponse<T = unknown> {
  code: number
  msg: string
  data: T
}

// 创建统一请求实例：集中处理 baseURL（/api 前缀）、超时、token、全局错误
const instance: AxiosInstance = axios.create({
  baseURL: '/api',
  timeout: 15000,
})

// 请求拦截：统一携带鉴权 token（敏感信息不写死在前端）
instance.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 响应拦截：解包统一结构 + 统一错误处理（规范 1.3）
instance.interceptors.response.use(
  (response: AxiosResponse<ApiResponse>) => {
    const body = response.data
    if (body.code === 200) {
      return body.data as never
    }
    // 业务错误码：抛出供调用方 catch
    return Promise.reject(new Error(body.msg || '请求失败'))
  },
  (error) => {
    const status = error.response?.status
    if (status === 401) {
      // 未登录：清除 token（真实权限必须由后端判定）
      localStorage.removeItem('token')
    }
    return Promise.reject(error)
  },
)

export default instance
