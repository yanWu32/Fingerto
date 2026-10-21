// SSE 流式读取：POST /api/pipeline/stream（multipart 上传视频 → 分阶段事件流）。
// 后端 server.py 的 _sse() 把每个事件字典直接序列化成 `data: {事件体}\n\n`，
// 即 data 行内就是 SseEvent 本身（无 {code,msg,data} 包裹）。详见下方解析处。
// 与统一 request.ts 不冲突：SSE 是流式特例，axios 不支持，单独用 fetch + ReadableStream 读取。
//
// 配套规范：组件内禁止裸写 fetch（这条是基础设施，集中在 api/ 层）；
// 取消用 AbortController（规范 2.3 取消 pending 请求）。
import type { AgentResult, GlossResult } from './types'

// 阶段名严格对齐后端 server.py pipeline_events() 实际下发的 `stage` 字段。
// 注意：后端 wire 用 snake_case（perception/segmentation/recognition/agent_step1..3），
// 与 PipelineFlow.vue 的 camelCase（sensing/skeleton/...）不同；用下方 toUiStage() 映射，
// 本模块只忠实表达契约。
export type StreamStageName =
  | 'idle'
  | 'perception'
  | 'segmentation'
  | 'recognition'
  | 'agent_step1'
  | 'agent_step2'
  | 'agent_step3'

/** UI 阶段名（PipelineFlow.vue / demoPipeline.ts 使用的 camelCase 枚举） */
export type UiStageName =
  | 'idle'
  | 'sensing'
  | 'skeleton'
  | 'segmenting'
  | 'recognizing'
  | 'agentStep1'
  | 'agentStep2'
  | 'agentStep3'
  | 'executing'
  | 'done'

/** wire 阶段 → UI 阶段映射表（后端 snake_case → 前端 camelCase） */
const STAGE_TO_UI: Record<StreamStageName, UiStageName> = {
  idle: 'idle',
  perception: 'sensing',
  segmentation: 'segmenting',
  recognition: 'recognizing',
  agent_step1: 'agentStep1',
  agent_step2: 'agentStep2',
  agent_step3: 'agentStep3',
}

/** 把后端 wire 阶段名转成 PipelineFlow 能识别的 UI 阶段名。 */
export function toUiStage(s: string): UiStageName {
  return STAGE_TO_UI[s as StreamStageName] ?? 'idle'
}

export interface StreamWindow {
  startTime: number
  endTime: number
  gloss: string
  confidence: number
}

export interface StreamEnvChange {
  label: string
  from: string
  to: string
}

export interface StreamApplication {
  apiPath: string
  taskName: string
  task: string
  controllerStatus: string
  changes: StreamEnvChange[]
  noop?: boolean
}

/** 后端逐个下发的事件联合类型（data 内的 type 字段区分） */
export type SseEvent =
  | { type: 'stage'; stage: StreamStageName; label: string }
  | {
      type: 'perception_done'
      shape: string
      numFrames: number
      fps: number
      durationSec: number
      /** 骨架本体较大，后端只下发这个 URL，前端按 jobId 二次拉取 */
      skeletonUrl?: string
    }
  | {
      type: 'segmentation_done'
      segments: { startTime: number; endTime: number }[]
      numSegments: number
      numWindows: number
    }
  | { type: 'recognition_window'; window: StreamWindow }
  | { type: 'recognition_done'; recognition: GlossResult }
  | { type: 'agent_done'; agent: AgentResult }
  | {
      type: 'application'
      apiPath: string
      taskName: string
      task: string
      controllerStatus: string
      changes: StreamEnvChange[]
      noop?: boolean
    }
  | { type: 'done'; recognition: GlossResult; agent: AgentResult }
  | { type: 'error'; code: number; msg: string }

/** 给 fetch 加上 Authorization（与 request.ts 拦截器保持一致） */
function authHeaders(): Record<string, string> {
  try {
    const token = localStorage.getItem('token')
    if (token) return { Authorization: `Bearer ${token}` }
  } catch {
    /* localStorage 不可用时忽略 */
  }
  return {}
}

export interface StreamHandle {
  /** 流转完（或出错/取消）后 resolve */
  promise: Promise<void>
  /** 取消：中断 fetch（AbortController） */
  cancel: () => void
  /** 本次任务的 jobId（用于取消接口 / 拉骨架） */
  jobId: string
}

/**
 * 流式识别：上传视频并逐事件回调。
 * @param file 视频文件（前端仅做类型提示，真实校验在后端）
 * @param onEvent 每收到一个事件回调
 * @param onError 网络错误 / 后端 error 事件回调（取消不算错误）
 */
export function streamPipeline(
  file: File,
  onEvent: (e: SseEvent) => void,
  onError?: (err: Error) => void
): StreamHandle {
  const jobId =
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : `job-${Date.now()}-${Math.random().toString(16).slice(2)}`
  const ctrl = new AbortController()
  const form = new FormData()
  form.append('video', file)
  form.append('jobId', jobId)

  const promise = (async () => {
    let resp: Response
    try {
      resp = await fetch('/api/pipeline/stream', {
        method: 'POST',
        body: form,
        headers: authHeaders(),
        signal: ctrl.signal,
      })
    } catch (e) {
      if ((e as Error).name === 'AbortError') return
      onError?.(e as Error)
      return
    }
    if (!resp.ok || !resp.body) {
      onError?.(new Error(`后端返回 HTTP ${resp.status}`))
      return
    }
    const reader = resp.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buf = ''
    try {
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        // SSE 以空行分隔事件块；逐块解析，避免半包
        let sep: number
        while ((sep = buf.indexOf('\n\n')) !== -1) {
          const raw = buf.slice(0, sep)
          buf = buf.slice(sep + 2)
          for (const line of raw.split('\n')) {
            if (!line.startsWith('data:')) continue
            const payload = line.slice(5).trim()
            if (!payload) continue
            try {
              const evt = JSON.parse(payload) as SseEvent
              if (evt && typeof evt.type === 'string') {
                onEvent(evt)
                if (evt.type === 'error') {
                  onError?.(new Error(evt.msg || '后端处理出错'))
                }
              }
            } catch {
              // 跳过不完整/非法 JSON，不中断整条流
            }
          }
        }
      }
    } catch (e) {
      if ((e as Error).name === 'AbortError') return
      onError?.(e as Error)
    } finally {
      try {
        reader.releaseLock()
      } catch {
        /* noop */
      }
    }
  })()

  return {
    promise,
    cancel: () => ctrl.abort(),
    jobId,
  }
}

/** 取消正在进行的流式任务：前端中断 + 通知后端停止（best-effort） */
export async function cancelStream(handle: StreamHandle): Promise<void> {
  handle.cancel()
  try {
    await fetch(`/api/pipeline/${encodeURIComponent(handle.jobId)}/cancel`, {
      method: 'POST',
      headers: authHeaders(),
    })
  } catch {
    /* 后端不可达也不影响前端取消 */
  }
}
