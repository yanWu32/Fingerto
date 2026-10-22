// 后端就绪探针（规范：统一的后端健康探测 + 自动降级到 mock）。
// 后端 server.py 提供 GET /health 返回 { code, msg, data: { ready, deps, models_loaded } }
// （字段名以 API接口表.md 为准：deps.torch / deps.mediapipe / deps.opencv，非 dependencies）。
// ready=false 表示推理依赖（torch/mediapipe/cv2）未就绪，前端应自动回退 mock（规范 1.3 数据无关先行）。
//
// 设计：运行时探测一次并缓存结果（负结果缓存 30s，避免反复打后端）；
// 网络失败 / 超时一律视为「不可用」，不阻塞 UI，由调用方决定降级。
import { ref } from 'vue'
import http from './request'

/** /health 返回的 data 部分（对齐 API接口表.md §1） */
export interface HealthData {
  ready: boolean
  /** 三项依赖就绪情况 */
  deps: {
    torch: boolean
    mediapipe: boolean
    opencv: boolean
  }
  /** 已缓存的识别模型，如 ["cnn_lstm"] */
  models_loaded: string[]
}

/** 骨架二次拉取返回结构：对齐后端 /skeleton/{jobId} 的 inner data。
 * 后端返回 { shape, fps, frames, skeleton }（skeleton 为 (T,27,3) 或 null）。 */
export interface SkeletonResponse {
  shape: string
  fps: number
  /** 帧数（后端字段名 frames，非 num_frames） */
  frames: number
  /** (T,27,3) 骨架张量（后端字段名 skeleton，非 data） */
  skeleton: number[][][] | null
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
/** 最近一次成功的探测明细（用于在 UI 上说明「缺哪个依赖」/「已加载哪个模型」） */
const health = ref<HealthData | null>(null)
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
      timeout: 15000,
    } as never)
    // 防御：serve.py 未带 --backend 时 /api/health 会回退成 SPA 的 HTML（200），
    // axios 仍能「解析」为字符串/对象。必须校验返回的确实是契约结构，
    // 否则会误判「真实后端」并展示乱码依赖。非契约响应一律视为不可达。
    const d = data as unknown
    if (
      !d ||
      typeof d !== 'object' ||
      typeof (d as HealthData).ready !== 'boolean' ||
      typeof (d as HealthData).deps !== 'object' ||
      (d as HealthData).deps === null
    ) {
      lastCheckedAt = now
      ready.value = false
      throw new BackendUnavailableError(
        'unreachable',
        '未检测到真实后端响应（请确认已用 python serve.py --backend-fastapi 启动）'
      )
    }
    ready.value = !!data.ready
    health.value = data
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

/** 最近一次探测明细（deps / models_loaded），未探测过则为 null。 */
export function backendHealth(): typeof health {
  return health
}

/** 把未就绪的依赖列成中文短语，供 UI 提示「缺什么」。 */
export function missingDeps(h: HealthData | null): string[] {
  if (!h) return []
  const names: Record<string, string> = {
    torch: 'torch（推理框架）',
    mediapipe: 'mediapipe（骨架抽取）',
    opencv: 'opencv（视频解码）',
  }
  return Object.entries(h.deps ?? {})
    .filter(([, ok]) => !ok)
    .map(([k]) => names[k] ?? k)
}

/** 按 jobId 二次拉取 (T,27,3) 骨架（/skeleton/{jobId}）。 */
export async function fetchSkeleton(jobId: string): Promise<SkeletonResponse> {
  return http.get<SkeletonResponse>(`/skeleton/${encodeURIComponent(jobId)}`)
}
