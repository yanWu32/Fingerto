<script setup lang="ts">
// 识别结果面板（演示页 / 识别页共用）：gloss 序列 + 置信度分档着色 + 点击定位骨架。
// 展示「逐窗口原始产出」表（可选，真实流式模式下由后端逐窗口事件喂入）。
import { formatConfidence } from '@/utils/format'
import type { GlossResult } from '@/api/types'

interface WindowRow {
  startTime: number
  endTime: number
  gloss: string
  confidence: number
}

const props = defineProps<{
  gloss: GlossResult | null
  /** 高亮词（点击/当前播放位置） */
  activeIndex?: number
  /** 逐窗口原始产出（可选，真实流式模式下提供） */
  windows?: WindowRow[]
  /** 低置信阈值（分档用）：high>=0.75, mid>=clarify(0.4), 其余 low */
  highThreshold?: number
  clarifyThreshold?: number
}>()

const emit = defineEmits<{ (e: 'seek', index: number): void }>()

function confLevel(c: number): 'high' | 'mid' | 'low' {
  const hi = props.highThreshold ?? 0.75
  const cl = props.clarifyThreshold ?? 0.4
  if (c >= hi) return 'high'
  if (c >= cl) return 'mid'
  return 'low'
}
</script>

<template>
  <section class="panel" aria-labelledby="gloss-title">
    <h2 id="gloss-title" class="panel__title">识别层结果：gloss 序列 + 置信度</h2>

    <ul v-if="gloss" class="glosses">
      <li
        v-for="(w, i) in gloss.gloss_sequence"
        :key="`${w}-${i}`"
        class="gloss"
        :class="[`gloss--${confLevel(gloss.confidence[i])}`, { 'gloss--active': activeIndex === i }]"
      >
        <button type="button" class="gloss__btn" @click="emit('seek', i)">
          <span class="gloss__word">{{ w }}</span>
          <span class="gloss__conf">{{ formatConfidence(gloss.confidence[i]) }}</span>
          <span class="gloss__bar">
            <span class="gloss__fill" :style="{ width: `${gloss.confidence[i] * 100}%` }" />
          </span>
          <span class="gloss__ts">
            {{ gloss.timestamps[i][0].toFixed(2) }}–{{ gloss.timestamps[i][1].toFixed(2) }}s
          </span>
        </button>
      </li>
    </ul>
    <p v-else class="gloss__empty">（暂无识别结果）</p>

    <details v-if="windows && windows.length" class="windows">
      <summary>逐窗口原始产出（{{ windows.length }} 个，点击展开）</summary>
      <table class="wtable">
        <thead>
          <tr>
            <th>窗口时间(s)</th>
            <th>Gloss</th>
            <th>置信度</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(w, i) in windows" :key="i">
            <td>{{ w.startTime.toFixed(2) }} – {{ w.endTime.toFixed(2) }}</td>
            <td><strong>{{ w.gloss }}</strong></td>
            <td :class="`c-${confLevel(w.confidence)}`">{{ w.confidence.toFixed(2) }}</td>
          </tr>
        </tbody>
      </table>
      <p class="windows__note">重叠窗口经 vote 投票与时间合并后，得到上方 gloss 序列。</p>
    </details>
  </section>
</template>

<style scoped>
.panel {
  background: var(--color-surface, #fff);
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: var(--radius, 0.75rem);
  padding: 1.1rem;
  box-shadow: var(--shadow-sm);
}

.panel__title {
  margin: 0 0 0.85rem;
  font-size: 1.05rem;
}

.glosses {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
}

.gloss {
  border-radius: 0.6rem;
  border: 1px solid var(--color-border, #e5e7eb);
  overflow: hidden;
}

.gloss--high {
  border-color: #16a34a;
}

.gloss--mid {
  border-color: #d97706;
}

.gloss--low {
  border-color: #dc2626;
}

.gloss--active {
  box-shadow: 0 0 0 3px rgba(217, 119, 6, 0.3);
}

.gloss__btn {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  padding: 0.5rem 0.7rem;
  border: none;
  background: var(--color-muted, #f9fafb);
  cursor: pointer;
  width: 100%;
  text-align: left;
}

.gloss__word {
  font-size: 1.2rem;
  font-weight: 700;
}

.gloss__conf {
  font-size: 0.8rem;
  color: var(--color-muted-text, #6b7280);
}

.gloss__bar {
  display: block;
  height: 0.3rem;
  background: var(--color-muted, #e5e7eb);
  border-radius: 999px;
  overflow: hidden;
}

.gloss__fill {
  display: block;
  height: 100%;
  background: var(--color-primary, #2563eb);
}

.gloss__ts {
  font-size: 0.72rem;
  color: var(--color-muted-text, #6b7280);
  font-variant-numeric: tabular-nums;
}

.gloss__empty {
  color: var(--color-muted-text, #6b7280);
}

.windows {
  margin-top: 0.8rem;
  font-size: 0.85rem;
}

.wtable {
  width: 100%;
  border-collapse: collapse;
  margin-top: 0.4rem;
}

.wtable th,
.wtable td {
  border: 1px solid var(--color-border, #e5e7eb);
  padding: 0.3rem 0.5rem;
  text-align: left;
}

.windows__note {
  margin: 0.4rem 0 0;
  color: var(--color-muted-text, #6b7280);
}

.c-high {
  color: #16a34a;
}

.c-mid {
  color: #d97706;
}

.c-low {
  color: #dc2626;
}
</style>
