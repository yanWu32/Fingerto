// 后端接口类型（事实源：Fingerto/tests/contract.py 的 GLOSS_SCHEMA / AGENT_SCHEMA）
// 前端接口契约与后端层间契约保持一致：gloss JSON（识别层→智能体层）、agent JSON（智能体层→应用层）。
// 统一响应结构见 request.ts：{ code, msg, data }。

/** 识别层输出（识别 → 智能体 契约，对应 contract.GLOSS_SCHEMA） */
export interface GlossResult {
  gloss_sequence: string[]
  confidence: number[]
  timestamps: [number, number][]
  /** 可选：每个词的 top1-top2 概率差，判别力指标（后端 margin_top1_top2） */
  margin_top1_top2?: number[]
  /** 可选：margin<0.2 标记为低判别力（后端 low_confidence） */
  low_confidence?: boolean[]
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

/** 请求入参说明：识别 / pipeline 端点均为 multipart 上传视频（见 API接口表.md），
 * 可选参数（model / device / ctx_frames / useAgent）见各 api 模块的 Options 接口；
 * 已被后端契约取代的 source 字段不再声明。 */
