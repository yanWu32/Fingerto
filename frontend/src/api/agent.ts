// 智能体层接口：/api/agent
// 后端契约见 tests/contract.py AGENT_SCHEMA（natural_language / intent / task / clarification / low_confidence_words）。
// task 必为 string（无任务 'none'），clarification 必为 boolean。
import request from './request'
import { USE_MOCK } from '@/utils/env'
import { mockAgent } from './mock'
import type { AgentResult, GlossResult } from './types'

/** 把识别层 gloss 结果交给智能体层，返回对话决策。 */
export async function runAgent(gloss: GlossResult): Promise<AgentResult> {
  if (USE_MOCK) {
    return mockAgent()
  }
  // 真实接口：入参为识别层产出，返回 { code, msg, data: AgentResult }
  return request.post<AgentResult>('/agent', gloss)
}
