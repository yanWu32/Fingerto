// 信息推送接口：POST /push、GET /push/lexicon。
// 契约见 后端改动请求-信息推送与审计日志与环境快照.md（后端落地前调用会得到 404/503，页面如实透出错误）。
// 红线：推送文本不进识别层；gloss 由后端词表映射构造（source=push_lexicon，confidence=1.0），
// agent 复用真实三步 LLM 管线——前端只展示后端返回内容，不编造。
import http from './request'
import type { PushResult } from './types'

/** POST /push 入参 */
export interface PushOptions {
  /** 推送类型：天气 / 通知 */
  kind: 'weather' | 'notice'
  /** 推送文本（如「今日天气：晴 22℃」） */
  text: string
}

/** 发起一次信息推送，返回 gloss（词表来源）+ agent（真实 LLM）+ 可选骨架片段。 */
export async function pushInfo(opts: PushOptions): Promise<PushResult> {
  return http.post<PushResult>('/push', { kind: opts.kind, text: opts.text })
}

/** GET /push/lexicon 的 data：当前可推送词表（可选接口，后端未实现时页面优雅降级）。 */
export interface PushLexicon {
  kinds: Record<string, string[]>
  /** 词 → 手语词序列 映射示例 */
  mapping?: Record<string, string[]>
}

/** 拉取当前可推送词表（供前端展示「可推送词表」提示；失败返回 null）。 */
export async function fetchPushLexicon(): Promise<PushLexicon | null> {
  try {
    return await http.get<PushLexicon>('/push/lexicon')
  } catch {
    // 词表属可选增强信息；拿不到不阻塞推送主流程
    return null
  }
}
