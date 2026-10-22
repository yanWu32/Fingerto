<script setup lang="ts">
// 真实识别入口（用户要求的「添加视频 → 开始 → 动态演示识别过程」）。
// 真实识别入口（用户要求的「添加视频 → 开始 → 动态演示识别过程」）。
// - 仅允许视频输入（accept=video/*，前端类型/大小提示，后端再次校验）。
// - 点击「开始识别」：先探测 /api/health，ready 才 POST /api/pipeline/stream 消费 SSE。
// - **所有展示数据均为后端真实产出**：不使用 mock，后端不可用/依赖缺失时直接报错说明原因
//   （用户明确要求：结果必须来自后端，禁止前端虚拟数据）。
// - 识别过程中逐事件驱动 PipelineFlow / 骨架 / gloss / 智能体结果。
// - 可「取消」（AbortController 中断 fetch + 通知后端 /pipeline/{jobId}/cancel）。
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import SkeletonCanvas from '@/components/SkeletonCanvas.vue'
import PipelineFlow from '@/components/PipelineFlow.vue'
import StageLog from '@/components/StageLog.vue'
import GlossBoard from '@/components/GlossBoard.vue'
import AgentSteps from '@/components/AgentSteps.vue'
import type { PipelineStageName, StageSnapshot } from '@/api/demoPipeline'
import {
  streamPipeline,
  cancelStream,
  toUiStage,
  type SseEvent,
} from '@/api/stream'
import {
  ensureBackendReady,
  BackendUnavailableError,
  fetchSkeleton,
  backendHealth,
  missingDeps,
} from '@/api/backend'
import type { AgentResult, GlossResult } from '@/api/types'
import type { SkeletonFrame } from '@/utils/skeleton'
import { formatConfidence } from '@/utils/format'

const MAX_BYTES = 50 * 1024 * 1024 // 与后端 MAX_UPLOAD_BYTES 对齐

// ---- 上传 ----
const file = ref<File | null>(null)
const fileName = ref('')
const fileSize = ref(0)
const fileErr = ref('')
const videoPreview = ref('') // object URL，仅用于本地预览（如有视频文件）
/** 拖拽悬停高亮状态（用于入口视觉反馈） */
const dragOver = ref(false)
let dragDepth = 0

function pickFile(e: Event) {
  const input = e.target as HTMLInputElement
  const f = input.files?.[0]
  if (f) validateAndSet(f)
  // 允许再次选择同一文件（清空 input 值）
  input.value = ''
}

// 支持拖拽（带进入/离开计数，避免子元素触发抖动）
function onDragEnter(e: DragEvent) {
  e.preventDefault()
  dragDepth += 1
  if (e.dataTransfer?.types?.includes('Files')) dragOver.value = true
}

function onDragLeave(e: DragEvent) {
  e.preventDefault()
  dragDepth = Math.max(0, dragDepth - 1)
  if (dragDepth === 0) dragOver.value = false
}

function onDragOver(e: DragEvent) {
  e.preventDefault()
}

function onDrop(e: DragEvent) {
  e.preventDefault()
  dragDepth = 0
  dragOver.value = false
  const f = e.dataTransfer?.files?.[0]
  if (f) validateAndSet(f)
}

function validateAndSet(f: File) {
  fileErr.value = ''
  const okType = f.type.startsWith('video/') || /\.(mp4|avi|mov|webm)$/i.test(f.name)
  if (!okType) {
    fileErr.value = '仅支持视频文件（mp4 / avi / mov / webm）。'
    return
  }
  if (f.size > MAX_BYTES) {
    fileErr.value = `文件过大（${mb(f.size)} MB），上限 50 MB。`
    return
  }
  if (videoPreview.value) URL.revokeObjectURL(videoPreview.value)
  file.value = f
  fileSize.value = f.size
  fileName.value = f.name
  videoPreview.value = URL.createObjectURL(f)
}

function clearFile() {
  file.value = null
  fileName.value = ''
  fileSize.value = 0
  fileErr.value = ''
  if (videoPreview.value) URL.revokeObjectURL(videoPreview.value)
  videoPreview.value = ''
}

/** 字节转 MB（保留两位小数） */
function mb(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(2)
}

// ---- 运行状态 ----
const stage = ref<PipelineStageName>('idle')
const stageLog = reactive<StageSnapshot[]>([])
const running = ref(false)
const error = ref('')
const mode = ref<'probe' | 'real'>('probe') // 当前识别所走模式（已移除 mock 降级，见 start()）
const jobId = ref('')

// 骨架 / gloss / 智能体
const frames = ref<SkeletonFrame[]>([])
const fps = ref(30)
const gloss = ref<GlossResult | null>(null)
const agent = ref<AgentResult | null>(null)
const activeGloss = ref(-1)
const playing = ref(true)
const playerComp = ref<InstanceType<typeof SkeletonCanvas> | null>(null)

// 切分 / 逐窗口（真实模式由 SSE 喂入）
const segments = ref<{ startTime: number; endTime: number }[]>([])
/** segmentation_done 携带的识别窗口总数（用于摘要展示） */
const numWindows = ref(0)
const windows = ref<{ startTime: number; endTime: number; gloss: string; confidence: number }[]>([])
const agentSteps = reactive<
  { title: string; promptSystem: string; promptUser: string; output: string }[]
>([])
const highlightTokens = ref<string[]>([])
const activeStep = ref(0)
const durationSec = ref(0)
const skeletonShape = ref('')
const modelName = ref('')

// ---- 后端详情 / 计时 ----
/** /health 探测明细（deps / models_loaded） */
const health = backendHealth()
const missing = computed(() => missingDeps(health.value))
/** 本次识别耗时（后端 CPU 约 100s，需让用户感知进度） */
const elapsed = ref(0)
let elapsedTimer: ReturnType<typeof setInterval> | null = null

function startTimer() {
  stopTimer()
  elapsed.value = 0
  elapsedTimer = setInterval(() => {
    elapsed.value += 0.1
  }, 100)
}

function stopTimer() {
  if (elapsedTimer) {
    clearInterval(elapsedTimer)
    elapsedTimer = null
  }
}

function resetRun() {
  stage.value = 'idle'
  stageLog.length = 0
  error.value = ''
  frames.value = []
  gloss.value = null
  agent.value = null
  segments.value = []
  numWindows.value = 0
  windows.value = []
  agentSteps.length = 0
  highlightTokens.value = []
  activeStep.value = 0
  activeGloss.value = -1
  skeletonShape.value = ''
  modelName.value = ''
  durationSec.value = 0
  resetDemo()
}

function pushStage(s: StageSnapshot) {
  stage.value = s.stage
  stageLog.push(s)
}

const canStart = computed(() => !!file.value && !running.value)

// ---------- 真实后端 SSE ----------
let activeHandle: { promise: Promise<void>; cancel: () => void; jobId: string } | null = null

async function runReal() {
  mode.value = 'real'
  const handle = streamPipeline(
    file.value!,
    onSseEvent,
    (err) => {
      // 后端 msg 直接透出（规范要求：msg 由后端给，前端不自行编造）
      error.value = err.message || '后端识别失败'
      stopTimer()
    }
  )
  jobId.value = handle.jobId
  activeHandle = handle
  try {
    await handle.promise
  } catch {
    /* onError 已处理 */
  } finally {
    running.value = false
    stopTimer()
    if (activeHandle === handle) activeHandle = null
  }
}

function onSseEvent(e: SseEvent) {
  switch (e.type) {
    case 'stage': {
      // 后端下发 snake_case 阶段名（perception/agent_step1…），需映射成
      // PipelineFlow 认识的 camelCase UI 阶段名，否则进度条不推进。
      const ui = toUiStage(e.stage)
      pushStage({
        stage: ui,
        label: e.label,
        layer: layerOf(e.stage),
        detail: '',
      })
      if (ui.startsWith('agentStep')) {
        activeStep.value = Number(ui.replace('agentStep', ''))
      }
      break
    }
    case 'perception_done':
      skeletonShape.value = e.shape
      fps.value = e.fps
      durationSec.value = e.durationSec
      pushStage({
        stage: 'skeleton',
        label: '提取 27 点骨架',
        layer: '感知层',
        detail: `张量 ${e.shape} · ${e.numFrames} 帧 · ${e.fps}fps`,
      })
      // perception_done 只带 skeletonUrl（骨架本体较大），需按 jobId 二次拉取
      fetchSkeletonAsync(jobId.value)
      break
    case 'segmentation_done':
      segments.value = e.segments
      numWindows.value = e.numWindows
      pushStage({
        stage: 'segmenting',
        label: '运动能量切分 + 滑动窗口',
        layer: '切分层',
        detail: `${e.numSegments} 个活跃段 → ${e.numWindows} 个识别窗口`,
      })
      break
    case 'recognition_window':
      windows.value.push(e.window)
      break
    case 'recognition_done':
      gloss.value = e.recognition
      pushStage({
        stage: 'recognizing',
        label: '逐窗口推理并投票聚合',
        layer: '识别层',
        detail: `gloss 序列 ${e.recognition.gloss_sequence.length} 词`,
      })
      break
    case 'agent_done':
      agent.value = e.agent
      // 低置信词作为后续高亮 token（若有）
      highlightTokens.value = e.agent.low_confidence_words
      // 重要：后端 SSE 只下发 agent_done（含最终 AgentSchema），
      // 不下发逐步的 prompt/输出。前端绝不自行编造三步提示词，
      // 因此这里只记录阶段，不填充 agentSteps（卡片保持为空→组件显示「后端未提供分步数据」）。
      activeStep.value = 3
      pushStage({
        stage: 'agentStep3',
        label: '智能体：三步提示管线完成',
        layer: '智能体层',
        detail: `intent=${e.agent.intent} · task=${e.agent.task}`,
      })
      break
    case 'application':
      pushStage({
        stage: 'executing',
        label: e.noop ? '无需执行（无任务）' : e.taskName,
        layer: '应用层',
        detail: e.noop ? 'task 为 none，不触发设备控制' : `${e.apiPath} · ${e.controllerStatus}`,
      })
      break
    case 'done':
      gloss.value = e.recognition
      agent.value = e.agent
      highlightTokens.value = e.agent.low_confidence_words
      pushStage({
        stage: 'done',
        label: '完成：手语视频 → 文本 → 任务',
        layer: '全链路',
        detail: `最终文本「${e.agent.natural_language}」`,
      })
      break
    case 'error':
      error.value = e.msg || '后端处理出错'
      break
  }
}

async function fetchSkeletonAsync(id: string) {
  try {
    const sk = await fetchSkeleton(id)
    if (sk.skeleton && sk.skeleton.length) {
      frames.value = sk.skeleton as SkeletonFrame[]
      fps.value = sk.fps || 30
    }
  } catch {
    /* 骨架为可选项，拉取失败不影响识别结果 */
  }
}


// ---------- 开始 / 取消 ----------
// 重要（用户要求）：页面展示的必须是后端真实输出，不允许前端编造结果。
// 因此这里**不再静默降级到 mock**：后端不可用就直接报错并说明原因，
// 让使用者明确知道「没有真实结果可展示」，而不是看到一段虚拟演示。
async function start() {
  if (!canStart.value) return
  resetRun()
  running.value = true
  error.value = ''
  startTimer()
  try {
    // 先探测后端就绪（结果按 30s 缓存，不阻塞）
    await ensureBackendReady()
    runReal()
  } catch (e) {
    running.value = false
    stopTimer()
    if (e instanceof BackendUnavailableError) {
      // 给出可操作的原因，而不是悄悄用假数据糊过去
      const miss = missing.value.length ? `缺失依赖：${missing.value.join('、')}。` : ''
      error.value =
        e.reason === 'not_ready'
          ? `后端推理依赖未就绪，无法产出真实结果。${miss}请安装依赖后重启后端（python serve.py --backend-fastapi）。`
          : `无法连接后端服务，无法产出真实结果。请确认已启动：python serve.py --backend-fastapi`
    } else {
      error.value = (e as Error).message || '启动失败'
    }
  }
}

function cancel() {
  if (mode.value === 'real' && activeHandle) {
    // 中断前端 fetch（AbortController）+ 通知后端停止（cancelStream 内部已 abort）
    cancelStream(activeHandle)
  }
  // 已无 mock 路径：取消仅作用于真实后端的流式任务
  running.value = false
  stopTimer()
  pushStage({
    stage: stage.value,
    label: '已取消',
    layer: '控制',
    detail: '用户中止识别',
  })
}

/** 阶段归属层：同时接受后端 wire 名（perception/agent_step1…）与 UI 名（sensing/agentStep1…）。 */
function layerOf(s: string): string {
  if (s === 'sensing' || s === 'skeleton' || s === 'perception') return '感知层'
  if (s === 'segmenting' || s === 'segmentation') return '切分层'
  if (s === 'recognizing' || s === 'recognition') return '识别层'
  if (s.startsWith('agentStep') || s.startsWith('agent_step')) return '智能体层'
  if (s === 'executing') return '应用层'
  if (s === 'done') return '全链路'
  return '控制'
}

function playGloss(i: number) {
  activeGloss.value = i
  const ts = gloss.value?.timestamps[i]
  if (ts && playerComp.value) playerComp.value.seekSec(ts[0])
}

// ---------- 切分层时间轴辅助 ----------
/** 活跃段定位（左/宽按占整段时长的百分比） */
function segStyle(s: { startTime: number; endTime: number }) {
  const total = durationSec.value || 1
  return {
    left: `${(s.startTime / total) * 100}%`,
    width: `${Math.max(((s.endTime - s.startTime) / total) * 100, 0.6)}%`,
  }
}

/** 仅当段宽足够（>6% 时长）才在内部显示文字，避免窄段标签互相重叠 */
function segLabelVisible(i: number) {
  const s = segments.value[i]
  if (!s) return false
  const total = durationSec.value || 1
  return (s.endTime - s.startTime) / total > 0.06
}

/** 自适应秒刻度：1s / 2s / 5s 三档，保证刻度数 ≤ 12 */
const tickSecs = computed(() => {
  const dur = durationSec.value
  if (!dur || dur <= 0) return []
  const step = dur <= 12 ? 1 : dur <= 25 ? 2 : 5
  const out: number[] = []
  for (let t = 0; t <= dur; t += step) out.push(Math.round(t))
  if (out[out.length - 1] < dur) out.push(Math.round(dur))
  return out
})

// ---------- 视频 ↔ 骨架演示同步比对 ----------
/** 比对区内的 <video> 元素（与 SkeletonCanvas 同屏，便于逐一对照） */
const videoRef = ref<HTMLVideoElement | null>(null)

/** SkeletonCanvas 逐帧上报时间 → 同步驱动视频（漂移超过 0.35s 才回写，避免高频 seek） */
function onSkeletonTime(sec: number) {
  const v = videoRef.value
  if (!v) return
  const limit = (v.duration || 0) > 0 ? v.duration - 0.05 : sec
  const target = Math.min(sec, limit)
  if (Math.abs(v.currentTime - target) > 0.35) v.currentTime = target
  if (playing.value) {
    if (v.paused) v.play().catch(() => {})
  } else if (!v.paused) {
    v.pause()
  }
}

/** 用户手动拖动视频进度条 → 反向定位骨架动画 */
function onVideoSeeked() {
  const v = videoRef.value
  if (v && playerComp.value) playerComp.value.seekSec(v.currentTime)
}

/** 「中断演示」：暂停骨架动画与视频（结果保留，可再次播放），不取消后端任务 */
function interruptDemo() {
  playing.value = false
  videoRef.value?.pause()
}

/** 新一轮识别开始时重置演示区（回到 0 点、自动播放） */
function resetDemo() {
  playing.value = true
  const v = videoRef.value
  if (v) {
    v.pause()
    v.currentTime = 0
  }
}

onBeforeUnmount(() => {
  if (videoPreview.value) URL.revokeObjectURL(videoPreview.value)
  if (running.value) cancel()
})

// 进入页面即探测后端真实可用性（不阻塞 UI），让上方「当前模式」与依赖明细提前展示，
// 用户上传前就能看到后端是否就绪，避免点了开始才发现没有真实结果。
onMounted(async () => {
  try {
    const ok = await ensureBackendReady()
    if (ok) mode.value = 'real'
  } catch {
    // 不可用：mode 保持 'probe'，UI 提示需启动后端；点击开始时再明确报错。
  }
})
</script>

<template>
  <div class="rc">
    <header class="rc__head">
      <h1 class="rc__title">手语视频识别</h1>
      <p class="rc__sub">
        选择一个手语视频，点击「开始识别」，系统将动态演示
        <strong>感知 → 骨架 → 切分 → 识别 → 智能体 → 应用</strong> 全链路。
      </p>
      <p class="rc__mode">
        当前模式：
        <span class="pill" :class="mode === 'real' ? 'pill--ok' : 'pill--idle'">
          {{ mode === 'real' ? '真实后端（SSE 流式）' : '就绪探测中' }}
        </span>
        <span v-if="running" class="pill pill--idle rc__elapsed">
          {{ elapsed.toFixed(1) }}s
        </span>
      </p>

      <p class="rc__notice">
        本页所有结果均来自后端真实推理（<code>/api/pipeline/stream</code>
        SSE）；后端不可用时直接报错，不使用任何前端虚拟数据。
      </p>

      <!-- 后端就绪明细（/health）：说明走真实还是降级、缺哪个依赖 -->
      <div v-if="health" class="rc__health">
        <span class="rc__health-item">
          依赖：
          <code v-for="(ok, k) in health.deps" :key="k" :class="ok ? 'dep dep--ok' : 'dep dep--bad'">
            {{ k }}={{ ok ? '✓' : '✗' }}
          </code>
        </span>
        <span v-if="health.models_loaded && health.models_loaded.length" class="rc__health-item">
          模型：<code>{{ health.models_loaded.join(', ') }}</code>
        </span>
        <span v-if="missing.length" class="rc__health-item rc__health-warn">
          缺失：{{ missing.join('、') }}（依赖不全，无法产出真实结果）
        </span>
      </div>
    </header>

    <!-- 上传区：拖拽入口 + 已选文件卡片（可更换/移除），与「开始识别」同行 -->
    <section class="upload panel-no">
      <div class="upload__row">
        <label
          class="drop"
          :class="{
            'drop--err': fileErr,
            'drop--filled': file,
            'drop--dragover': dragOver,
          }"
          @dragenter="onDragEnter"
          @dragleave="onDragLeave"
          @dragover="onDragOver"
          @drop="onDrop"
        >
          <input
            type="file"
            accept="video/*,.mp4,.avi,.mov,.webm"
            class="drop__input"
            @change="pickFile"
          >
          <div v-if="!file" class="drop__hint">
            <span class="drop__icon" aria-hidden="true">🎞️</span>
            <p><strong>点击或拖拽手语视频到此处</strong></p>
            <p class="drop__meta">仅限视频（mp4 / avi / mov / webm），≤ 50 MB</p>
          </div>
          <div v-else class="drop__file">
            <span class="drop__file-icon" aria-hidden="true">📹</span>
            <div class="drop__file-info">
              <p class="drop__file-name">{{ fileName }}</p>
              <p class="drop__file-meta">{{ mb(fileSize) }} MB · 点击可更换视频</p>
            </div>
            <button
              type="button"
              class="link"
              @click.prevent.stop="clearFile"
            >
              移除
            </button>
          </div>
        </label>

        <div class="rc__actions">
          <button type="button" class="btn btn--primary" :disabled="!canStart || running" @click="start">
            {{ running ? '识别中…' : '▶ 开始识别' }}
          </button>
          <button type="button" class="btn" :disabled="!running" @click="cancel">
            ■ 中断识别
          </button>
        </div>
      </div>
      <p v-if="fileErr" role="alert" class="rc__err">{{ fileErr }}</p>
    </section>

    <p v-if="error" role="alert" class="rc__err rc__err--run">{{ error }}</p>

    <PipelineFlow :current="stage" />

    <StageLog :log="stageLog" :current="stage" />

    <div class="rc__grid">
      <!-- 左：感知比对区（原始视频 + 骨架演示同屏，时间轴同步） -->
      <section class="panel" aria-labelledby="vis-title">
        <h2 id="vis-title" class="panel__title">① 感知层：视频 → (T, 27, 3) 骨架张量</h2>
        <div class="vis__meta">
          <p><strong>视频源</strong>：{{ file ? fileName : '—' }}</p>
          <p v-if="skeletonShape">
            <strong>骨架张量</strong>：<code>{{ skeletonShape }}</code> ·
            {{ fps }}fps · {{ durationSec.toFixed(1) }}s
            <template v-if="modelName"> · 模型 {{ modelName }}</template>
          </p>
          <p v-else class="vis__hint">（上传并开始后展示实时提取的骨架）</p>
        </div>

        <!-- 演示控制：播放/暂停与「中断演示」（不打断后端任务，仅停画面） -->
        <div class="vis__ctrl">
          <button type="button" class="btn" :disabled="!frames.length" @click="playing = !playing">
            {{ playing ? '⏸ 暂停演示' : '▶ 播放演示' }}
          </button>
          <button type="button" class="btn" :disabled="!frames.length" @click="interruptDemo">
            ■ 中断演示
          </button>
          <span class="vis__ctrl-hint">视频与骨架同屏、时间轴同步；点 gloss 可定位比对</span>
        </div>

        <div class="vis__compare">
          <figure class="vis__pane">
            <figcaption class="vis__pane-label">原始视频</figcaption>
            <video
              v-if="videoPreview"
              ref="videoRef"
              :src="videoPreview"
              class="vis__video"
              controls
              muted
              playsinline
              @seeked="onVideoSeeked"
            />
            <p v-else class="vis__hint">（选择视频后在此播放）</p>
          </figure>

          <figure class="vis__pane">
            <figcaption class="vis__pane-label">骨架演示（27 点实时渲染）</figcaption>
            <SkeletonCanvas
              v-if="frames.length"
              ref="playerComp"
              :frames="frames"
              :fps="fps"
              :highlight-range="
                activeGloss >= 0 && gloss?.timestamps[activeGloss]
                  ? gloss.timestamps[activeGloss]
                  : undefined
              "
              :playing="playing"
              @time="onSkeletonTime"
            />
            <p v-else class="vis__hint">（识别开始后展示实时提取的骨架）</p>
          </figure>
        </div>
      </section>

      <!-- 右：切分层（时间轴 + 秒刻度，窄段自动隐藏文字、悬浮提示） -->
      <section class="panel" aria-labelledby="seg-title">
        <h2 id="seg-title" class="panel__title">② 切分层：运动能量切分</h2>
        <div v-if="segments.length" class="seg">
          <div class="seg__summary">
            <span>共 <strong>{{ segments.length }}</strong> 个活跃段</span>
            <span v-if="numWindows">→ <strong>{{ numWindows }}</strong> 个识别窗口</span>
            <span class="seg__summary-dur">时间轴 0–{{ durationSec.toFixed(1) }}s</span>
          </div>
          <div class="timeline" role="img" :aria-label="`切分时间轴：${segments.length} 个活跃段`">
            <!-- 秒刻度（自适应密度：每 1s / 2s / 5s） -->
            <div
              v-for="t in tickSecs"
              :key="`tick${t}`"
              class="timeline__tick"
              :style="{ left: `${(t / (durationSec || 1)) * 100}%` }"
            >
              <span class="timeline__tick-mark" />
              <span class="timeline__tick-label">{{ t }}s</span>
            </div>
            <!-- 活跃段 -->
            <div
              v-for="(s, i) in segments"
              :key="`sg${i}`"
              class="timeline__seg"
              :class="{ 'timeline__seg--label': segLabelVisible(i) }"
              :title="`段${i + 1}：${s.startTime.toFixed(1)}s – ${s.endTime.toFixed(1)}s（${(s.endTime - s.startTime).toFixed(1)}s）`"
              :style="segStyle(s)"
            >
              <template v-if="segLabelVisible(i)">
                <span class="timeline__seg-no">段{{ i + 1 }}</span>
              </template>
            </div>
          </div>
        </div>
        <p v-else class="vis__hint">（识别开始后展示运动能量切分结果；无活跃段则为空）</p>
      </section>
    </div>

    <GlossBoard
      :gloss="gloss"
      :windows="windows"
      :active-index="activeGloss"
      @seek="playGloss"
    />

    <AgentSteps
      :steps="agentSteps"
      :highlight-tokens="highlightTokens"
      :active-step="activeStep"
      :agent="agent"
    />

    <section v-if="gloss && agent" class="panel panel--final" aria-labelledby="final-title">
      <h2 id="final-title" class="panel__title">最终输出</h2>
      <p class="final__text">「{{ agent.natural_language }}」</p>
      <dl class="kv kv--inline">
        <div v-if="gloss.margin_top1_top2?.length">
          <dt>margin_top1_top2</dt>
          <dd><code>{{ gloss.margin_top1_top2.map((m) => m.toFixed(3)).join(', ') }}</code></dd>
        </div>
        <div v-if="gloss.low_confidence?.some(Boolean)">
          <dt>low_confidence</dt>
          <dd><span class="pill pill--warn">存在低置信词</span></dd>
        </div>
        <div v-if="agent.task && agent.task !== 'none'">
          <dt>执行任务</dt>
          <dd><code>{{ agent.task }}</code></dd>
        </div>
      </dl>
      <p v-if="gloss" class="final__raw">
        原始 gloss：{{ gloss.gloss_sequence.join(' ') }} ·
        置信度 {{ gloss.confidence.map(formatConfidence).join(' / ') }}
      </p>
    </section>
  </div>
</template>

<style scoped>
.rc {
  display: flex;
  flex-direction: column;
  gap: 1.1rem;
}

.rc__title {
  margin: 0 0 0.3rem;
  font-size: 1.6rem;
}

.rc__sub {
  margin: 0;
  color: var(--color-muted-text, #6b7280);
}

.rc__mode {
  margin: 0.4rem 0 0;
  font-size: 0.9rem;
}

.pill {
  padding: 0.12rem 0.6rem;
  border-radius: 999px;
  font-size: 0.78rem;
}

.pill--ok {
  background: #dcfce7;
  color: #166534;
}

.pill--warn {
  background: #fef3c7;
  color: #92400e;
}

.pill--idle {
  background: var(--color-muted, #f3f4f6);
  color: var(--color-muted-text, #6b7280);
}

.rc__elapsed {
  font-variant-numeric: tabular-nums;
  margin-left: 0.35rem;
}

/* 真实性声明：本页只用后端真实输出 */
.rc__notice {
  margin: 0.5rem 0 0;
  font-size: 0.82rem;
  color: var(--color-muted-text, #6b7280);
}

.rc__notice code {
  background: var(--color-muted, #f3f4f6);
  padding: 0.1rem 0.3rem;
  border-radius: 0.3rem;
  font-size: 0.78rem;
}

/* /health 依赖明细 */
.rc__health {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1rem;
  margin-top: 0.5rem;
  font-size: 0.82rem;
  color: var(--color-muted-text, #6b7280);
}

.rc__health-item {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
}

.rc__health-warn {
  color: #92400e;
}

.dep {
  padding: 0.05rem 0.3rem;
  border-radius: 0.3rem;
  font-size: 0.75rem;
}

.dep--ok {
  background: #dcfce7;
  color: #166534;
}

.dep--bad {
  background: #fee2e2;
  color: #991b1b;
}

.upload {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

/* 上传入口与操作按钮同行：左侧拖拽区自适应，右侧按钮组右对齐 */
.upload__row {
  display: flex;
  gap: 0.8rem;
  align-items: stretch;
  flex-wrap: wrap;
}

.upload__row .drop {
  flex: 1 1 260px;
}

.rc__actions {
  display: flex;
  gap: 0.6rem;
  align-items: center;
}

.drop {
  display: block;
  border: 2px dashed var(--color-border, #cbd5e1);
  border-radius: var(--radius, 0.75rem);
  padding: 1rem 1.2rem;
  text-align: center;
  cursor: pointer;
  background: var(--color-surface, #fff);
  transition:
    border-color 0.15s ease,
    background 0.15s ease,
    box-shadow 0.15s ease;
}

/* 拖拽悬停：高亮提示「可以放」 */
.drop--dragover {
  border-color: var(--color-primary, #2563eb);
  background: #eff6ff;
  box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12);
}

/* 已选文件：实线边框 + 横排卡片（图标 / 名称+大小 / 移除） */
.drop--filled {
  border-style: solid;
  border-color: var(--color-border, #e5e7eb);
  background: #fafafa;
}

.drop--err {
  border-color: #dc2626;
}

.drop__input {
  display: none;
}

.drop__hint p {
  margin: 0.2rem 0;
}

.drop__icon {
  font-size: 1.7rem;
}

.drop__meta {
  color: var(--color-muted-text, #6b7280);
  font-size: 0.82rem;
}

/* 已选文件卡片：横排布局，名称与大小分离，长文件名自动省略 */
.drop__file {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  text-align: left;
}

.drop__file-icon {
  font-size: 1.3rem;
  flex: none;
}

.drop__file-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
}

.drop__file-name {
  margin: 0;
  font-weight: 600;
  font-size: 0.92rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.drop__file-meta {
  margin: 0;
  color: var(--color-muted-text, #6b7280);
  font-size: 0.78rem;
}

.link {
  background: none;
  border: none;
  color: var(--color-primary, #2563eb);
  cursor: pointer;
  text-decoration: underline;
  flex: none;
  font-size: 0.85rem;
}

.btn {
  padding: 0.55rem 1.1rem;
  border-radius: 0.55rem;
  border: 1px solid var(--color-border, #e5e7eb);
  background: var(--color-surface, #fff);
  cursor: pointer;
  font-size: 0.95rem;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn--primary {
  background: var(--color-primary, #2563eb);
  color: #fff;
  border-color: var(--color-primary, #2563eb);
}

.rc__err {
  color: #dc2626;
  margin: 0;
}

.rc__err--run {
  font-weight: 600;
}

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

.rc__grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.1rem;
}

@media (max-width: 760px) {
  .rc__grid {
    grid-template-columns: 1fr;
  }
}

.vis__meta p {
  margin: 0.2rem 0;
  font-size: 0.9rem;
}

.vis__hint {
  color: var(--color-muted-text, #6b7280);
  font-size: 0.9rem;
}

/* 视频 ↔ 骨架 同屏比对 */
.vis__ctrl {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 0.6rem 0;
}

.vis__ctrl-hint {
  font-size: 0.78rem;
  color: var(--color-muted-text, #6b7280);
}

.vis__compare {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.vis__pane {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.vis__pane-label {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--color-muted-text, #6b7280);
}

.vis__video {
  width: 100%;
  max-height: 300px;
  object-fit: contain;
  background: #0b1020;
  border-radius: var(--radius, 0.75rem);
}

@media (max-width: 760px) {
  .vis__compare {
    grid-template-columns: 1fr;
  }
}

/* 切分层：摘要行 + 带秒刻度的时间轴 */
.seg {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.seg__summary {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1rem;
  font-size: 0.85rem;
  color: var(--color-muted-text, #6b7280);
}

.seg__summary strong {
  color: var(--color-text, #111827);
  font-variant-numeric: tabular-nums;
}

.seg__summary-dur {
  margin-left: auto;
  font-size: 0.78rem;
  font-variant-numeric: tabular-nums;
}

.timeline {
  position: relative;
  height: 3rem;
  background: var(--color-muted, #f3f4f6);
  border-radius: 0.4rem;
  overflow: hidden;
}

/* 秒刻度：竖线 + 下方小字（位于活跃段下层，不互相遮挡） */
.timeline__tick {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 0;
}

.timeline__tick-mark {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  width: 1px;
  background: rgba(148, 163, 184, 0.5);
}

.timeline__tick-label {
  position: absolute;
  bottom: 1px;
  left: 3px;
  font-size: 0.62rem;
  color: #64748b;
  font-variant-numeric: tabular-nums;
}

.timeline__seg {
  position: absolute;
  top: 0.3rem;
  bottom: 0.9rem; /* 留出底部 0.6rem 给秒刻度标签 */
  display: flex;
  align-items: center;
  overflow: hidden;
  background: rgba(37, 99, 235, 0.25);
  border-left: 2px solid var(--color-primary, #2563eb);
  border-right: 2px solid var(--color-primary, #2563eb);
  border-radius: 0.2rem;
  transition: background 0.15s ease;
}

.timeline__seg:hover {
  background: rgba(37, 99, 235, 0.45);
  cursor: help;
}

/* 段内编号（仅宽段显示；窄段靠悬浮 title 提示） */
.timeline__seg-no {
  font-size: 0.7rem;
  font-weight: 600;
  color: #1d4ed8;
  padding-left: 0.3rem;
  white-space: nowrap;
}

.timeline__seg--label .timeline__seg-no {
  display: inline;
}

.timeline__seg:not(.timeline__seg--label) .timeline__seg-no {
  display: none;
}

.kv {
  margin: 0.5rem 0 0;
  display: grid;
  gap: 0.3rem;
  font-size: 0.85rem;
}

.kv--inline > div {
  display: flex;
  gap: 0.4rem;
  align-items: center;
}

.kv dt {
  color: var(--color-muted-text, #6b7280);
}

.final__text {
  margin: 0 0 0.6rem;
  font-size: 1.1rem;
}

.final__raw {
  margin: 0.5rem 0 0;
  color: var(--color-muted-text, #6b7280);
  font-size: 0.85rem;
}
</style>
