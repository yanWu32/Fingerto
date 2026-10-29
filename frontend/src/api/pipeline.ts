// 端到端 pipeline 接口：/api/pipeline（上传视频 → 识别 + 智能体，同步全量）
// 契约见 API接口表.md §3：multipart（video + useAgent），返回 { recognition, agent }。
// mock 环境本地拼接，真实环境一次请求拿全链路结果，返回类型一致。
import { USE_MOCK } from '@/utils/env'
import { mockPipeline } from './mock'
import request from './request'
import type { PipelineResult } from './types'

/** /pipeline 的可选参数 */
export interface PipelineOptions {
  /** 是否跑智能体层，默认 true */
  useAgent?: boolean
  /** 识别模型：stgcn / cnn_lstm / transformer_gcn */
  model?: string
  /** 推理设备：cpu / cuda */
  device?: string
  /** 上下文扩展帧数 */
  ctxFrames?: number
}

/**
 * 端到端：上传视频 → 识别 + 智能体。
 * mock 模式下忽略入参直接返回内置数据（便于无后端跑通界面）。
 */
export async function runPipeline(
  video?: File,
  opts: PipelineOptions = {}
): Promise<PipelineResult> {
  if (USE_MOCK) {
    return mockPipeline()
  }
  if (!video) {
    throw new Error('真实模式需要上传视频文件')
  }
  const form = new FormData()
  form.append('video', video)
  form.append('useAgent', String(opts.useAgent ?? true))
  if (opts.model) form.append('model', opts.model)
  if (opts.device) form.append('device', opts.device)
  if (opts.ctxFrames !== undefined) form.append('ctx_frames', String(opts.ctxFrames))
  return request.post<PipelineResult>('/pipeline', form)
}
