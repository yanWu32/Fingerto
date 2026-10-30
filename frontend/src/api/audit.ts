// 审计日志接口：GET /audit_log?limit=&filter=。
// 契约见 后端改动请求-信息推送与审计日志与环境快照.md（后端落地前调用会得到 404，页面如实透出）。
// 后端把 /pipeline/stream 的 done/application 事件与 /push 成功记录追加写入 data/audit_log.json。
import http from './request'
import type { AuditRecord } from './types'

/** 日志筛选：全部 / 含澄清 / 家居操作（对应后端 filter 参数） */
export type AuditFilter = 'all' | 'clarify' | 'action'

/** 拉取审计日志（手动刷新，不做轮询）。 */
export async function fetchAuditLog(
  limit = 50,
  filter: AuditFilter = 'all'
): Promise<AuditRecord[]> {
  const data = await http.get<{ records: AuditRecord[] }>('/audit_log', {
    params: { limit, filter },
  } as never)
  return data?.records ?? []
}
