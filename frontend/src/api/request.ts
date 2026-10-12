import axios, {
  type AxiosInstance,
  type AxiosResponse,
  type InternalAxiosRequestConfig,
} from 'axios'

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
instance.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 响应拦截：只做统一错误处理，不做解包。
// 解包放在下面的 http 包装里做一次，调用处拿到的泛型 T 即业务 data 类型。
// （此前曾在此处解包一次、又在 http 里解包第二次，导致类型与运行时不一致。）
instance.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: unknown) => {
    const status = (error as { response?: { status?: number } }).response?.status
    if (status === 401) {
      // 未登录：清除 token（真实权限必须由后端判定）
      localStorage.removeItem('token')
    }
    return Promise.reject(error)
  }
)

// 对 axios 方法做一层包装：泛型 T 直接表示「业务 data 的类型」，
// 而非整包 ApiResponse，从而去掉 as never 这类强制转换。
export const http = {
  get: <T>(url: string, config?: InternalAxiosRequestConfig): Promise<T> =>
    instance.get<ApiResponse<T>>(url, config).then((res) => res.data.data),
  post: <T>(url: string, data?: unknown, config?: InternalAxiosRequestConfig): Promise<T> =>
    instance.post<ApiResponse<T>>(url, data, config).then((res) => res.data.data),
  put: <T>(url: string, data?: unknown, config?: InternalAxiosRequestConfig): Promise<T> =>
    instance.put<ApiResponse<T>>(url, data, config).then((res) => res.data.data),
  delete: <T>(url: string, config?: InternalAxiosRequestConfig): Promise<T> =>
    instance.delete<ApiResponse<T>>(url, config).then((res) => res.data.data),
}

export default http
