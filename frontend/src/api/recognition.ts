// 识别层接口：/api/recognition
// 后端契约见 API接口表.md §2：multipart/form-data 上传视频（video 必填，
// model / device / ctx_frames 可选），返回 data 为 GLOSS_SCHEMA。
// 通过 VITE_USE_MOCK 切换 mock / 真实请求（返回类型一致，调用方无感）。
import request from './request'
import { USE_MOCK } from '@/utils/env'
import { mockRecognition } from './mock'
import type { GlossResult } from './types'

/** /recognition 的可选参数（与后端 Form 默认值一致，故均可选） */
export interface RecognizeOptions {
  /** 识别模型：stgcn / cnn_lstm / transformer_gcn，默认 cnn_lstm */
  model?: string
  /** 推理设备：cpu / cuda，默认 cpu */
  device?: string
  /** 上下文扩展帧数，默认 20 */
  ctxFrames?: number
}

/** 上传视频做识别，返回识别层 gloss 结果（同步端点，SSE 流式请用 streamPipeline）。 */
export async function recognize(
  video: File,
  opts: RecognizeOptions = {}
): Promise<GlossResult> {
  if (USE_MOCK) {
    return mockRecognition()
  }
  // multipart 上传：content-type 交给浏览器自动带 boundary，不手写
  const form = new FormData()
  form.append('video', video)
  if (opts.model) form.append('model', opts.model)
  if (opts.device) form.append('device', opts.device)
  if (opts.ctxFrames !== undefined) form.append('ctx_frames', String(opts.ctxFrames))
  return request.post<GlossResult>('/recognition', form)
}
