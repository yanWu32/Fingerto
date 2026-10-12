// 后端接口类型（事实源：Fingerto/tests/contract.py 的 GLOSS_SCHEMA / AGENT_SCHEMA）
// 前端接口契约与后端层间契约保持一致：gloss JSON（识别层→智能体层）、agent JSON（智能体层→应用层）。
// 统一响应结构见 request.ts：{ code, msg, data }。

/** 识别层输出（识别 → 智能体 契约，对应 contract.GLOSS_SCHEMA） */
export interface GlossResult {
  gloss_sequence: string[]
  confidence: number[]
  timestamps: [number, number][]
}

/** 智能体层输出（智能体 → 应用 契约，对应 contract.AGENT_SCHEMA）
 * 注意：task 必为 string（无任务时为 'none'），clarification 必为 boolean，二者不可为 null。 */
export interface AgentResult {
  natural_language: string
  intent: string
  task: string
  clarification: boolean
  low_confidence_words: string[]
}

/** 端到端 pipeline 输出：识别 + 智能体 串联结果 */
export interface PipelineResult {
  recognition: GlossResult
  agent: AgentResult
}

/** 识别请求入参（前端 → 后端识别层）。
 * 后端真实接口可选上传视频/骨架；此处仅声明契约字段，真实字段以后端 OpenAPI 为准。 */
export interface RecognitionRequest {
  /** 视频/骨架数据标识或 base64；真实字段待后端定稿 */
  source?: string
}
