<script setup lang="ts">
// 流水线进度时间轴：感知 -> 骨架 -> 切分 -> 识别 -> 智能体三步 -> 应用执行。
import { computed } from 'vue'
import type { PipelineStageName } from '@/api/demoPipeline'

const props = defineProps<{
  current: PipelineStageName
  /** 已经过阶段的明细 key，用于展示耗时说明 */
  elapsedLabel?: string
}>()

interface NodeDef {
  key: PipelineStageName
  layer: string
  label: string
  icon: string
}

const NODES: NodeDef[] = [
  { key: 'sensing', layer: '感知层', label: '读取视频', icon: '🎬' },
  { key: 'skeleton', layer: '感知层', label: '27 点骨架', icon: '🦴' },
  { key: 'segmenting', layer: '切分层', label: '切分/窗口', icon: '✂️' },
  { key: 'recognizing', layer: '识别层', label: 'Gloss 识别', icon: '🔤' },
  { key: 'agentStep1', layer: '智能体', label: 'Step1 消歧', icon: '①' },
  { key: 'agentStep2', layer: '智能体', label: 'Step2 翻译', icon: '②' },
  { key: 'agentStep3', layer: '智能体', label: 'Step3 决策', icon: '③' },
  { key: 'executing', layer: '应用层', label: '任务执行', icon: '🏠' },
]

const ORDER: Record<PipelineStageName, number> = {
  idle: -1,
  sensing: 0,
  skeleton: 1,
  segmenting: 2,
  recognizing: 3,
  agentStep1: 4,
  agentStep2: 5,
  agentStep3: 6,
  executing: 7,
  done: 8,
}

const currentIndex = computed(() => ORDER[props.current] ?? -1)

function stateOf(idx: number): 'done' | 'active' | 'todo' {
  if (props.current === 'done') return 'done'
  if (idx < currentIndex.value) return 'done'
  if (idx === currentIndex.value) return 'active'
  return 'todo'
}

const progress = computed(() => {
  if (props.current === 'done') return 100
  return Math.max(0, (currentIndex.value / (NODES.length - 1)) * 100)
})
</script>

<template>
  <nav class="pflow" aria-label="流水线阶段进度">
    <ol class="pflow__list">
      <li
        v-for="(n, i) in NODES"
        :key="n.key"
        class="pflow__item"
        :class="`pflow__item--${stateOf(i)}`"
        :aria-current="stateOf(i) === 'active' ? 'step' : undefined"
      >
        <div class="pflow__node">
          <span class="pflow__icon" aria-hidden="true">{{ n.icon }}</span>
        </div>
        <span class="pflow__layer">{{ n.layer }}</span>
        <span class="pflow__label">{{ n.label }}</span>
      </li>
    </ol>
    <div
      class="pflow__bar"
      role="progressbar"
      :aria-valuenow="progress"
      aria-valuemin="0"
      aria-valuemax="100"
    >
      <div class="pflow__fill" :style="{ width: `${progress}%` }" />
    </div>
  </nav>
</template>

<style scoped>
.pflow__list {
  list-style: none;
  margin: 0 0 0.75rem;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(8, minmax(0, 1fr));
  gap: 0.35rem;
}

.pflow__item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.15rem;
  text-align: center;
  font-size: 0.72rem;
  color: var(--color-muted-text, #6b7280);
}

.pflow__node {
  width: 2rem;
  height: 2rem;
  border-radius: 999px;
  display: grid;
  place-items: center;
  background: var(--color-muted, #f3f4f6);
  border: 2px solid transparent;
  transition: all 0.2s ease;
}

.pflow__icon {
  font-size: 0.95rem;
  line-height: 1;
}

.pflow__item--done .pflow__node {
  background: #dcfce7;
  border-color: var(--color-success, #16a34a);
}

.pflow__item--active .pflow__node {
  background: #dbeafe;
  border-color: var(--color-primary, #2563eb);
  transform: scale(1.12);
  box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12);
}

.pflow__item--done,
.pflow__item--active {
  color: var(--color-text, #1f2937);
}

.pflow__label {
  font-weight: 600;
  line-height: 1.2;
}

.pflow__layer {
  font-size: 0.65rem;
  opacity: 0.75;
}

.pflow__bar {
  height: 0.35rem;
  border-radius: 999px;
  background: var(--color-muted, #f3f4f6);
  overflow: hidden;
}

.pflow__fill {
  height: 100%;
  background: linear-gradient(90deg, #2563eb, #7c3aed);
  border-radius: 999px;
  transition: width 0.35s ease;
}

@media (max-width: 720px) {
  .pflow__list {
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.6rem 0.35rem;
  }
}
</style>
