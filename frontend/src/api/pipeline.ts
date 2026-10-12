// 端到端 pipeline 接口：/api/pipeline（识别 + 智能体 串联）
// 前端先具备「接后端」能力：真实环境一次请求拿全链路结果；mock 环境本地拼接。
import { USE_MOCK } from '@/utils/env'
import { mockPipeline } from './mock'
import { recognize } from './recognition'
import { runAgent } from './agent'
import type { PipelineResult, RecognitionRequest } from './types'

/** 端到端：识别 → 智能体。mock 下并行聚合，真实下走 /pipeline 单接口。 */
export async function runPipeline(req?: RecognitionRequest): Promise<PipelineResult> {
  if (USE_MOCK) {
    return mockPipeline()
  }
  const recognition = await recognize(req)
  const agent = await runAgent(recognition)
  return { recognition, agent }
}
