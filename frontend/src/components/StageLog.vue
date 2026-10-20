<script setup lang="ts">
// 阶段执行流水（演示页 / 识别页共用）：把逐阶段快照渲染成时间轴列表。
import type { PipelineStageName, StageSnapshot } from '@/api/demoPipeline'

defineProps<{
  log: StageSnapshot[]
  /** 高亮当前阶段（可选） */
  current?: PipelineStageName
}>()
</script>

<template>
  <section class="panel" aria-labelledby="log-title">
    <h2 id="log-title" class="panel__title">阶段执行流水</h2>
    <ol class="log">
      <li
        v-for="(l, i) in log"
        :key="i"
        class="log__item"
        :class="{ 'log__item--cur': current && l.stage === current }"
      >
        <span class="log__layer">{{ l.layer }}</span>
        <span class="log__label">{{ l.label }}</span>
        <span class="log__detail">{{ l.detail }}</span>
      </li>
    </ol>
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

.log {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  max-height: 15rem;
  overflow-y: auto;
}

.log__item {
  display: flex;
  gap: 0.6rem;
  align-items: baseline;
  font-size: 0.85rem;
  padding: 0.3rem 0.5rem;
  border-left: 3px solid var(--color-primary, #2563eb);
  background: var(--color-muted, #f9fafb);
  border-radius: 0 0.4rem 0.4rem 0;
}

.log__item--cur {
  border-left-color: var(--color-warning, #d97706);
  background: #fffbeb;
}

.log__layer {
  flex: 0 0 auto;
  font-weight: 600;
  color: var(--color-primary, #2563eb);
  min-width: 4.5rem;
}

.log__label {
  font-weight: 600;
}

.log__detail {
  color: var(--color-muted-text, #6b7280);
  margin-left: auto;
  text-align: right;
}
</style>
