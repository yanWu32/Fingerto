<script setup lang="ts">
// 审计日志页（/log）：查看 /recognize 与 /push 产生的识别/澄清/家居操作记录。
// 数据全部来自后端 GET /audit_log（后端落地前如实透出 404/503 错误，不 mock 降级）。
// 手动刷新，不做轮询 / WebSocket（需求边界 5）。
import { computed, onMounted, ref } from 'vue'
import { ensureBackendReady, BackendUnavailableError } from '@/api/backend'
import { fetchAuditLog, type AuditFilter } from '@/api/audit'
import type { AuditRecord } from '@/api/types'

// ---- 后端就绪 ----
const backendErr = ref('')
const backendOk = ref(false)

onMounted(async () => {
  try {
    await ensureBackendReady()
    backendOk.value = true
    await refresh()
  } catch (e) {
    backendOk.value = false
    backendErr.value =
      e instanceof BackendUnavailableError
        ? `${e.message}（请确认已启动：python serve.py）`
        : (e as Error).message
  }
})

// ---- 筛选与数据 ----
const TABS: { key: AuditFilter; label: string }[] = [
  { key: 'all', label: '全部' },
  { key: 'clarify', label: '含澄清' },
  { key: 'action', label: '家居操作' },
]
const filter = ref<AuditFilter>('all')
const loading = ref(false)
const loadErr = ref('')
const records = ref<AuditRecord[]>([])
const limit = 50

async function refresh() {
  if (loading.value) return
  loading.value = true
  loadErr.value = ''
  try {
    records.value = await fetchAuditLog(limit, filter.value)
  } catch (e) {
    loadErr.value = (e as Error).message || '日志加载失败'
  } finally {
    loading.value = false
  }
}

// ---- 展示辅助 ----
/** gloss 序列折叠：超 4 词收起，点击展开 */
const expanded = ref<Set<number>>(new Set())
const FOLD_AT = 4

function isFolded(rec: AuditRecord, i: number): boolean {
  return rec.gloss_sequence.length > FOLD_AT && !expanded.value.has(i)
}

function toggleFold(i: number) {
  const next = new Set(expanded.value)
  if (next.has(i)) next.delete(i)
  else next.add(i)
  expanded.value = next
}

function shownGloss(rec: AuditRecord, i: number): string[] {
  if (!isFolded(rec, i)) return rec.gloss_sequence
  return rec.gloss_sequence.slice(0, FOLD_AT)
}

/** 置信度最低值（记录内），表格中标红加粗 */
function minConf(rec: AuditRecord): number | null {
  if (!rec.confidence?.length) return null
  return Math.min(...rec.confidence)
}

function fmtTs(ts: string): string {
  // 后端 ISO 时间（含时区）；本地时区短格式展示
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return ts
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

const emptyHint = computed(
  () =>
    '暂无日志记录，请先在 /recognize 或 /push 触发识别' +
    (filter.value === 'all' ? '' : `（当前筛选：${TABS.find((t) => t.key === filter.value)?.label}）`)
)
</script>

<template>
  <div class="log">
    <header class="log__head">
      <h1 class="log__title">审计日志</h1>
      <p class="log__sub">
        记录每次识别 / 推送的 gloss、置信度、意图、任务与澄清情况；数据由后端写入
        <code>data/audit_log.json</code>（后端改动请求新增接口）。
      </p>
      <p v-if="!backendOk && backendErr" role="alert" class="log__err">{{ backendErr }}</p>
    </header>

    <template v-if="backendOk">
      <!-- 筛选栏 -->
      <section class="panel">
        <div class="log__bar">
          <div class="log__tabs" role="tablist" aria-label="日志筛选">
            <button
              v-for="t in TABS"
              :key="t.key"
              type="button"
              class="log__tab"
              :class="{ 'log__tab--on': filter === t.key }"
              role="tab"
              :aria-selected="filter === t.key"
              @click="filter = t.key"
            >
              {{ t.label }}
            </button>
          </div>
          <button type="button" class="btn" :disabled="loading" @click="refresh">
            {{ loading ? '刷新中…' : '↻ 刷新' }}
          </button>
        </div>

        <p v-if="loadErr" role="alert" class="log__err">{{ loadErr }}</p>

        <!-- 日志表格 -->
        <div v-if="records.length" class="log__scroll">
          <table class="log__table">
            <thead>
              <tr>
                <th>时间</th>
                <th>来源</th>
                <th>gloss 序列</th>
                <th>置信度(最低)</th>
                <th>意图</th>
                <th>任务</th>
                <th>澄清</th>
                <th>低置信词</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(r, i) in records" :key="`${r.ts}-${r.jobId}-${i}`">
                <td class="log__mono">{{ fmtTs(r.ts) }}</td>
                <td>
                  <span class="pill" :class="r.source === 'push' ? 'pill--info' : 'pill--ok'">
                    {{ r.source === 'push' ? '推送' : '视频' }}
                  </span>
                </td>
                <td>
                  <button
                    v-if="r.gloss_sequence.length > FOLD_AT"
                    type="button"
                    class="log__gloss-toggle"
                    @click="toggleFold(i)"
                  >
                    {{ shownGloss(r, i).join(' ') }}{{ isFolded(r, i) ? ` …(共${r.gloss_sequence.length}词，展开)` : ' （收起）' }}
                  </button>
                  <span v-else class="log__gloss">{{ r.gloss_sequence.join(' ') }}</span>
                </td>
                <td class="log__mono">
                  <strong v-if="minConf(r) !== null" class="log__minconf">{{ minConf(r)!.toFixed(2) }}</strong>
                  <span v-else>—</span>
                </td>
                <td class="log__mono">{{ r.intent || '—' }}</td>
                <td class="log__mono">{{ r.task || '—' }}</td>
                <td>
                  <span class="pill" :class="r.clarification ? 'pill--warn' : 'pill--ok'">
                    {{ r.clarification ? '是' : '否' }}
                  </span>
                </td>
                <td>
                  <template v-if="r.low_confidence_words?.length">
                    <code v-for="w in r.low_confidence_words" :key="w" class="tag">{{ w }}</code>
                  </template>
                  <span v-else>—</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p v-else-if="!loading && !loadErr" class="log__empty">{{ emptyHint }}</p>
      </section>
    </template>
  </div>
</template>

<style scoped>
.log {
  display: flex;
  flex-direction: column;
  gap: 1.1rem;
}

.log__title {
  margin: 0 0 0.3rem;
  font-size: 1.6rem;
}

.log__sub {
  margin: 0;
  color: var(--color-muted-text, #6b7280);
}

.log__sub code {
  background: var(--color-muted, #f3f4f6);
  padding: 0.1rem 0.3rem;
  border-radius: 0.3rem;
  font-size: 0.8rem;
}

.log__bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.8rem;
  flex-wrap: wrap;
  margin-bottom: 0.8rem;
}

.log__tabs {
  display: flex;
  gap: 0.4rem;
}

.log__tab {
  padding: 0.4rem 0.9rem;
  border-radius: 999px;
  border: 1px solid var(--color-border, #e5e7eb);
  background: var(--color-surface, #fff);
  cursor: pointer;
  font-size: 0.88rem;
}

.log__tab--on {
  background: var(--color-primary, #2563eb);
  border-color: var(--color-primary, #2563eb);
  color: #fff;
}

.log__scroll {
  overflow-x: auto;
}

.log__table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.log__table th,
.log__table td {
  border: 1px solid var(--color-border, #e5e7eb);
  padding: 0.4rem 0.55rem;
  text-align: left;
  vertical-align: top;
  white-space: nowrap;
}

.log__table th {
  background: var(--color-muted, #f9fafb);
  font-weight: 600;
}

.log__mono {
  font-variant-numeric: tabular-nums;
}

.log__gloss {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: inline-block;
  max-width: 100%;
}

.log__gloss-toggle {
  border: none;
  background: none;
  color: var(--color-primary, #2563eb);
  cursor: pointer;
  padding: 0;
  font-size: 0.85rem;
  text-align: left;
}

.log__minconf {
  color: #dc2626;
}

.log__empty {
  color: var(--color-muted-text, #6b7280);
  margin: 0.4rem 0 0;
}

.log__err {
  color: #dc2626;
  margin: 0.4rem 0 0;
  font-weight: 600;
}

.pill {
  display: inline-block;
  padding: 0.1rem 0.55rem;
  border-radius: 999px;
  font-size: 0.75rem;
}

.pill--ok {
  background: #dcfce7;
  color: #166534;
}

.pill--warn {
  background: #fef3c7;
  color: #92400e;
}

.pill--info {
  background: #dbeafe;
  color: #1d4ed8;
}

.tag {
  display: inline-block;
  margin: 0 0.2rem 0.2rem 0;
  padding: 0.1rem 0.4rem;
  border-radius: 0.35rem;
  background: #fef2f2;
  color: #b91c1c;
  font-size: 0.78rem;
}

.panel {
  background: var(--color-surface, #fff);
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: var(--radius, 0.75rem);
  padding: 1.1rem;
  box-shadow: var(--shadow-sm);
}

.btn {
  padding: 0.45rem 1rem;
  border-radius: 0.55rem;
  border: 1px solid var(--color-border, #e5e7eb);
  background: var(--color-surface, #fff);
  cursor: pointer;
  font-size: 0.9rem;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
