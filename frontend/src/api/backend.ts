// 后端就绪探针（规范：统一的后端健康探测 + 自动降级到 mock）。
// 后端 server.py 提供 GET /health 返回 { code, msg, data: { ready, dependencies } }；
// ready=false 表示推理依赖（torch/mediapipe/cv2）未就绪，前端应自动回退 mock（规范 1.3 数据无关先行）。
//
// 设计：运行时探测一次并缓存结果（负结果缓存 30s，避免反复打后端）；
// 网络失败 / 超时一律视为「不可用」，不阻塞 UI，由调用方决定降级。
import { ref } from 'vue'
import http from './request'

/** /health 返回的 data 部分 */
export interface HealthData {
  ready: boolean
  dependencies: Record<string, boolean>
}

/** 骨架二次拉取返回结构：data.{ shape, fps, num_frames, data }（最内层 data 为 (T,27,3) 或 null） */
export interface SkeletonResponse {
  shape: string
  fps: number
  num_frames: number
  data: number[][][] | null
}

/** 后端不可用（探测失败 / ready=false）时抛出，调用方据此回退 mock。 */
export class BackendUnavailableError extends Error {
  /** ready=false（依赖未就绪）还是网络/探测失败 */
  readonly reason: 'not_ready' | 'unreachable'
  constructor(reason: 'not_ready' | 'unreachable', message: string) {
    super(message)
    this.name = 'BackendUnavailableError'
    this.reason = reason
  }
}

// 模块级单例：就绪态 + 阴性缓存
const ready = ref<boolean | null>(null)
const checking = ref(false)
let lastCheckedAt = 0
const NEG_CACHE_MS = 30_000

/**
 * 探测后端是否可用。成功返回 true；ready=false 抛 BackendUnavailableError('not_ready')；
 * 探测失败/超时抛 BackendUnavailableError('unreachable')。结果按 NEG_CACHE_MS 缓存。
 */
export async function ensureBackendReady(force = false): Promise<boolean> {
  const now = Date.now()
  if (!force && ready.value === true) return true
  if (!force && ready.value === false && now - lastCheckedAt < NEG_CACHE_MS) {
    throw new BackendUnavailableError('not_ready', '后端推理依赖未就绪（已缓存）')
  }
  if (checking.value) {
    // 已有探测在跑，复用其结果
    while (checking.value) {
      await new Promise((r) => setTimeout(r, 50))
    }
    if (ready.value === true) return true
    throw new BackendUnavailableError(
      ready.value === false ? 'not_ready' : 'unreachable',
      '后端不可用'
    )
  }
  checking.value = true
  try {
    const data = await http.get<HealthData>('/health', {
      timeout: 4000,
    } as never)
    ready.value = !!data?.ready
    lastCheckedAt = now
    if (!ready.value) {
      throw new BackendUnavailableError('not_ready', '后端推理依赖未就绪（torch/mediapipe/cv2）')
    }
    return true
  } catch (e) {
    if (e instanceof BackendUnavailableError) throw e
    // 网络错误 / 超时 / 非契约响应 → 视为不可达
    lastCheckedAt = now
    ready.value = false
    throw new BackendUnavailableError('unreachable', '无法连接后端服务（请确认 serve.py --backend-fastapi 已启动）')
  } finally {
    checking.value = false
  }
}

/** 当前就绪态（用于 UI 展示「真实后端 / mock」徽标，不触发探测）。 */
export function backendReady(): typeof ready {
  return ready
}

/** 按 jobId 二次拉取 (T,27,3) 骨架（/skeleton/{jobId}）。 */
export async function fetchSkeleton(jobId: string): Promise<SkeletonResponse> {
  return http.get<SkeletonResponse>(`/skeleton/${encodeURIComponent(jobId)}`)
}
