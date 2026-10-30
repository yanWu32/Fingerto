// 后端接口类型（事实源：Fingerto/tests/contract.py 的 GLOSS_SCHEMA / AGENT_SCHEMA）
// 前端接口契约与后端层间契约保持一致：gloss JSON（识别层→智能体层）、agent JSON（智能体层→应用层）。
// 统一响应结构见 request.ts：{ code, msg, data }。
import type { SkeletonFrame } from '@/utils/skeleton'

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

/** 模拟家居环境快照（DialogueState.env 的 JSON；由 application 事件 envBefore/envAfter 下发）。
 * 键为环境对象名（light/window/ac/curtain/pending…），值为状态字符串。
 * 注意：该字段属后端改动请求新增项，后端落地前为 undefined，前端不得编造。 */
export type EnvSnapshot = Record<string, unknown>

/** 信息推送（/push）返回结构（data 部分；外层仍为 {code,msg,data} 信封）。
 * gloss 由后端「信息词→手语词」映射表构造并标注 source=push_lexicon（confidence=1.0），
 * agent 复用真实三步 LLM 管线产出。 */
export interface PushResult {
  /** 识别层结构但来源为推送词表（非视频识别产出） */
  gloss: GlossResult
  agent: AgentResult
  /** 可选：每个 gloss 词对应的骨架片段（供动画，(Ti,27,3)）；某词无素材时 frames 为空数组 */
  skeleton_frames?: { word: string; frames: SkeletonFrame[] }[]
  /** 数据来源标记，预期 "push_lexicon" */
  source?: string
}

/** 审计日志（/audit_log）单条记录（data.records 元素）。
 * 写入时机：/pipeline/stream 的 done 与 application 事件、/push 每次成功。 */
export interface AuditRecord {
  ts: string
  jobId: string
  type: 'recognition' | 'agent' | 'action' | 'clarify'
  gloss_sequence: string[]
  confidence: number[]
  intent: string
  task: string
  clarification: boolean
  low_confidence_words: string[]
  source: 'video' | 'push'
}
