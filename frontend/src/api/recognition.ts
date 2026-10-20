// 识别层接口：/api/recognition
// 后端契约见 tests/contract.py GLOSS_SCHEMA（gloss_sequence / confidence / timestamps）。
// 通过 VITE_USE_MOCK 切换 mock / 真实请求（types 一致，调用方无感）。
import request from './request'
import { USE_MOCK } from '@/utils/env'
import { mockRecognition } from './mock'
import type { GlossResult, RecognitionRequest } from './types'

/** 提交视频/骨架，返回识别层 gloss 结果。 */
export async function recognize(req?: RecognitionRequest): Promise<GlossResult> {
  if (USE_MOCK) {
    return mockRecognition()
  }
  // 真实接口：后端返回 { code, msg, data: GlossResult }
  return request.post<GlossResult>('/recognition', req ?? {})
}
