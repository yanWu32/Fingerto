<script setup lang="ts">
// 27 点骨架播放器：把 Vite SizedBox 的 (T, 27, 3) 张量渲染成可播放的 SVG 动画。
// 事实源：perception/skeleton.py 的 HAND_EDGES / POSE_EDGES 连接关系。
import { computed, onUnmounted, ref, watch } from 'vue'
import {
  HAND_EDGES,
  POSE_ABS_EDGES,
  framesBounds,
  NUM_HAND_POINTS,
  type SkeletonFrame,
} from '@/utils/skeleton'

const props = defineProps<{
  frames: SkeletonFrame[]
  fps: number
  /** 当前 gloss 高亮区间 [startSec, endSec] */
  highlightRange?: [number, number]
  playing?: boolean
}>()

const emit = defineEmits<{ (e: 'time', sec: number): void }>()

const current = ref(0)
let timer: ReturnType<typeof setInterval> | null = null

const total = computed(() => props.frames.length)
const currentSec = computed(() => (props.fps ? current.value / props.fps : 0))

// 自适应 viewBox：按整段序列包围盒留出 8% 边距，避免硬编码坐标范围
const viewBox = computed(() => {
  const { minX, minY, maxX, maxY } = framesBounds(props.frames)
  const padX = (maxX - minX) * 0.08 || 0.1
  const padY = (maxY - minY) * 0.08 || 0.1
  return `${minX - padX} ${minY - padY} ${maxX - minX + padX * 2} ${maxY - minY + padY * 2}`
})

const frame = computed<SkeletonFrame | null>(() => props.frames[current.value] ?? null)

/** 主手与前臂连接的额外骨（手腕->左腕 Pose26），用于连接手与上身 */
const EXTRA_EDGES: [number, number][] = [[0, 26]]

const edges = computed(() => [...HAND_EDGES, ...POSE_ABS_EDGES, ...EXTRA_EDGES])

function lineCoords(
  edge: [number, number]
): { x1: number; y1: number; x2: number; y2: number } | null {
  const f = frame.value
  if (!f) return null
  const a = f[edge[0]]
  const b = f[edge[1]]
  if (!a || !b) return null
  return { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }
}

const handPoints = computed(() => (frame.value ?? []).slice(0, NUM_HAND_POINTS))
const posePoints = computed(() => (frame.value ?? []).slice(NUM_HAND_POINTS))

const highlighted = computed(() => {
  if (!props.highlightRange) return false
  const [s, e] = props.highlightRange
  return currentSec.value >= s && currentSec.value <= e
})

function play() {
  if (timer) return
  timer = setInterval(
    () => {
      current.value = (current.value + 1) % Math.max(1, total.value)
    },
    1000 / (props.fps || 30)
  )
}

function stop() {
  if (timer) {
    clearInterval(timer)
    timer = null
  }
}

watch(
  () => props.playing,
  (v) => {
    if (v) play()
    else stop()
  },
  { immediate: true }
)

watch([total], () => {
  current.value = 0
})

watch(currentSec, (sec) => emit('time', sec))

onUnmounted(stop)

/** 跳转到指定秒（点击 gloss 时定位到对应骨架区间起点） */
function seekSec(sec: number) {
  const f = Math.round(sec * (props.fps || 30))
  current.value = Math.min(Math.max(0, f), Math.max(0, total.value - 1))
}

defineExpose({ stop, play, seekSec })
</script>

<template>
  <figure class="skel">
    <svg
      class="skel__svg"
      :viewBox="viewBox"
      role="img"
      :aria-label="`骨架序列第 ${current + 1} / ${total} 帧，时刻 ${currentSec.toFixed(2)} 秒`"
    >
      <!-- 骨连接：主手 21 点 + 上半身 6 点 + 手腕与前臂的连接 -->
      <g stroke-width="0.022" stroke-linecap="round" fill="none">
        <line
          v-for="(e, i) in edges"
          :key="`e${i}`"
          :class="[
            'skel__bone',
            { 'skel__bone--hand': i < HAND_EDGES.length, 'skel__bone--hi': highlighted },
          ]"
          v-bind="lineCoords(e) ?? {}"
        />
      </g>

      <!-- 关键点 -->
      <g>
        <circle
          v-for="(p, i) in handPoints"
          :key="`h${i}`"
          :cx="p[0]"
          :cy="p[1]"
          r="0.035"
          class="skel__pt skel__pt--hand"
        />
        <circle
          v-for="(p, i) in posePoints"
          :key="`p${i}`"
          :cx="p[0]"
          :cy="p[1]"
          r="0.045"
          class="skel__pt skel__pt--pose"
        />
      </g>
    </svg>

    <figcaption class="skel__cap">
      <span class="skel__badge" :class="{ 'skel__badge--hi': highlighted }">
        帧 {{ current + 1 }} / {{ total }}
      </span>
      <span class="skel__t">t = {{ currentSec.toFixed(2) }}s</span>
      <span class="skel__hint">输入张量 (T, 27, 3) · 实时渲染</span>
    </figcaption>
  </figure>
</template>

<style scoped>
.skel {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.skel__svg {
  width: 100%;
  height: auto;
  aspect-ratio: 4 / 3;
  background: linear-gradient(180deg, #fbfcfe 0%, #f1f5f9 100%);
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: var(--radius, 0.75rem);
}

.skel__bone {
  stroke: var(--color-primary, #2563eb);
  opacity: 0.75;
}

.skel__bone--hand {
  stroke: #0ea5e9;
}

.skel__bone--hi {
  stroke: var(--color-warning, #d97706);
  opacity: 1;
  filter: drop-shadow(0 0 0.03rem rgba(217, 119, 6, 0.6));
}

.skel__pt--hand {
  fill: #0369a1;
}

.skel__pt--pose {
  fill: var(--color-muted-text, #6b7280);
}

.skel__cap {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
  font-size: 0.82rem;
  color: var(--color-muted-text, #6b7280);
}

.skel__badge {
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  background: var(--color-muted, #f3f4f6);
  font-variant-numeric: tabular-nums;
}

.skel__badge--hi {
  background: #fef3c7;
  color: #92400e;
}

.skel__t {
  font-variant-numeric: tabular-nums;
}

.skel__hint {
  margin-left: auto;
}
</style>
