<script setup lang="ts">
// 模拟家居环境 JSON 对照面板（/recognize）。
// 展示后端 application 事件下发的 envBefore / envAfter 两份 DialogueState.env 快照，
// 高亮发生变化的键；task=none 或后端未下发快照时给出诚实提示，不编造数据。
import { computed } from 'vue'

const props = defineProps<{
  /** 执行前环境快照（后端 envBefore）；未下发时为 null */
  before: Record<string, unknown> | null
  /** 执行后环境快照（后端 envAfter）；未下发时为 null */
  after: Record<string, unknown> | null
  /** 本轮任务标识（后端 agent.task / application.task） */
  task: string
  /** 控制器状态说明（后端 application.controllerStatus） */
  status?: string
}>()

/** 是否为家居任务轮（none = 无任务） */
const hasTask = computed(() => props.task && props.task !== 'none')

/** 后端是否已下发两份快照（envBefore/envAfter 为本次后端改动请求的新增字段） */
const hasSnapshots = computed(() => !!props.before && !!props.after)

/** 所有出现过的键（并集），保持稳定顺序：after 的键序优先，再补 before 独有键 */
const keys = computed<string[]>(() => {
  if (!props.before || !props.after) return []
  const set = new Set<string>([
    ...Object.keys(props.after),
    ...Object.keys(props.before),
  ])
  return [...set]
})

/** 发生变化的键列表：值不同（含一边缺失） */
function isChanged(k: string): boolean {
  if (!props.before || !props.after) return false
  return JSON.stringify(props.before[k]) !== JSON.stringify(props.after[k])
}

const changedKeys = computed<string[]>(() => keys.value.filter(isChanged))

function fmt(v: unknown): string {
  return typeof v === 'string' ? v : JSON.stringify(v)
}

/** JSON 美化输出（纯插值渲染，无 v-html，天然免注入） */
function pretty(o: Record<string, unknown> | null): string {
  return o ? JSON.stringify(o, null, 2) : ''
}
</script>

<template>
  <section class="panel" aria-labelledby="env-title">
    <h2 id="env-title" class="panel__title">模拟家居环境（JSON）</h2>

    <!-- 无任务轮 -->
    <p v-if="!hasTask" class="env__none">本轮无家居任务，环境不变。</p>

    <template v-else>
      <!-- 有任务但后端未下发快照：如实说明，等待后端改动请求落地 -->
      <p v-if="!hasSnapshots" class="env__pending">
        本轮任务 <code>{{ task }}</code> 已执行，但后端暂未下发环境快照
        （<code>envBefore</code>/<code>envAfter</code> 字段，见后端改动请求）。
        控制器状态：<code>{{ status || '—' }}</code>（模拟环境，未连接真实设备）。
      </p>

      <template v-else>
        <div class="env__grid">
          <figure class="env__pane">
            <figcaption class="env__label">执行前 envBefore</figcaption>
            <pre class="env__json"><code>{{ pretty(before) }}</code></pre>
          </figure>
          <figure class="env__pane">
            <figcaption class="env__label">执行后 envAfter</figcaption>
            <pre class="env__json"><code>{{ pretty(after) }}</code></pre>
          </figure>
        </div>

        <p v-if="changedKeys.length" class="env__changed">
          变更键：
          <code v-for="k in changedKeys" :key="k" class="env__diff">
            {{ k }}: {{ before && fmt(before[k]) }} → {{ after && fmt(after[k]) }}
          </code>
        </p>
        <p v-else class="env__changed env__changed--same">环境键无变化。</p>

        <p class="env__status">
          task=<code>{{ task }}</code> · 控制器状态 <code>{{ status || '—' }}</code>
          （模拟环境，未连接真实设备）
        </p>
      </template>
    </template>
  </section>
</template>

<style scoped>
.env__none,
.env__pending {
  margin: 0;
  font-size: 0.9rem;
  color: var(--color-muted-text, #6b7280);
}

.env__pending code,
.env__status code {
  background: var(--color-muted, #f3f4f6);
  padding: 0.1rem 0.3rem;
  border-radius: 0.3rem;
  font-size: 0.8rem;
}

.env__grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.env__pane {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  min-width: 0;
}

.env__label {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--color-muted-text, #6b7280);
}

.env__json {
  margin: 0;
  padding: 0.7rem 0.9rem;
  background: #0b1020;
  color: #d1e3ff;
  border-radius: 0.55rem;
  font-size: 0.8rem;
  line-height: 1.5;
  overflow-x: auto;
  font-family:
    ui-monospace,
    SFMono-Regular,
    Menlo,
    Consolas,
    monospace;
}

.env__changed {
  margin: 0.7rem 0 0;
  font-size: 0.85rem;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem 0.6rem;
}

.env__changed--same {
  color: var(--color-muted-text, #6b7280);
}

.env__diff {
  background: #fef3c7;
  color: #92400e;
  padding: 0.12rem 0.45rem;
  border-radius: 0.35rem;
  font-size: 0.78rem;
  font-variant-numeric: tabular-nums;
}

.env__status {
  margin: 0.6rem 0 0;
  font-size: 0.82rem;
  color: var(--color-muted-text, #6b7280);
}

@media (max-width: 760px) {
  .env__grid {
    grid-template-columns: 1fr;
  }
}
</style>
