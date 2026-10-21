<script setup lang="ts">
// 真实识别入口（用户要求的「添加视频 → 开始 → 动态演示识别过程」）。
// - 仅允许视频输入（accept=video/*，前端类型/大小提示，后端再次校验）。
// - 点击「开始识别」：优先走真实后端（先探测 /api/health，ready 则 POST /api/pipeline/stream 消费 SSE）；
//   探测失败 / ready=false 自动降级到 mock 演示（规范 1.3 数据无关先行）。
// - 识别过程中逐事件驱动 PipelineFlow / 骨架 / gloss / 智能体三步，动态演示全链路。
// - 可「取消」（AbortController 中断 fetch + 通知后端 /pipeline/{jobId}/cancel；mock 模式则中止演示）。
import { computed, onBeforeUnmount, reactive, ref } from 'vue'
import SkeletonCanvas from '@/components/SkeletonCanvas.vue'
import PipelineFlow from '@/components/PipelineFlow.vue'
import StageLog from '@/components/StageLog.vue'
import GlossBoard from '@/components/GlossBoard.vue'
import AgentSteps from '@/components/AgentSteps.vue'
import { buildScenario, runFullDemo, type DemoCanceller, DemoCancelledError } from '@/api/demoPipeline'
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
const fileErr = ref('')
const videoPreview = ref('') // object URL，仅用于本地预览（如有视频文件）

function pickFile(e: Event) {
  const input = e.target as HTMLInputElement
  const f = input.files?.[0]
  if (!f) return
  validateAndSet(f)
}

// 支持拖拽
function onDrop(e: DragEvent) {
  e.preventDefault()
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
  fileName.value = `${f.name}（${mb(f.size)} MB）`
  videoPreview.value = URL.createObjectURL(f)
}

function clearFile() {
  file.value = null
  fileName.value = ''
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
const mode = ref<'probe' | 'real' | 'mock'>('probe') // 当前识别所走模式
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
  windows.value = []
  agentSteps.length = 0
  highlightTokens.value = []
  activeStep.value = 0
  activeGloss.value = -1
  skeletonShape.value = ''
  modelName.value = ''
  durationSec.value = 0
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
      // 后端不逐条下发 agent_step 事件（只在内部跑完三步后给 agent_done），
      // 因此这里一次性补全三步卡片，保证 AgentSteps 不空。
      if (!agentSteps.length) {
        agentSteps.push(
          {
            title: 'Step 1 · 低置信消歧确认',
            promptSystem: '你是手语识别结果的校对助手，需确认低置信词。',
            promptUser: `gloss: ${e.agent.low_confidence_words.join(' ') || '（无低置信词）'}`,
            output: e.agent.low_confidence_words.length
              ? `标记低置信词：${e.agent.low_confidence_words.join('、')}`
              : '无需消歧',
          },
          {
            title: 'Step 2 · gloss → 自然中文',
            promptSystem: '把手语 gloss 序列翻译成通顺的中文句子。',
            promptUser: gloss.value?.gloss_sequence.join(' ') ?? '',
            output: e.agent.natural_language,
          },
          {
            title: 'Step 3 · 意图与任务决策',
            promptSystem: '按 AGENT_SCHEMA 输出 intent / task / clarification 的 JSON。',
            promptUser: e.agent.natural_language,
            output: JSON.stringify(
              {
                intent: e.agent.intent,
                task: e.agent.task,
                clarification: e.agent.clarification,
              },
              null,
              2
            ),
          }
        )
      }
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

// ---------- mock 回退 ----------
async function runMock() {
  mode.value = 'mock'
  const scenario = buildScenario(0)
  const canceller: DemoCanceller = { cancelled: false }
  mockCanceller = canceller
  durationSec.value = scenario.sense.durationSec
  skeletonShape.value = scenario.skeleton.shape
  modelName.value = scenario.recognition.model
  try {
    await runFullDemo(
      scenario,
      (s) => {
        pushStage(s)
        // 把 mock 场景产物映射到 UI
        if (s.stage === 'skeleton') {
          frames.value = scenario.frames
        }
        if (s.stage === 'segmenting') {
          segments.value = scenario.segmentation.segments
        }
        if (s.stage === 'recognizing') {
          const p = s.payload as { glossResult?: GlossResult }
          if (p.glossResult) gloss.value = p.glossResult
          if (scenario.recognition.windows) {
            windows.value = scenario.recognition.windows.map((w) => ({
              startTime: w.startTime,
              endTime: w.endTime,
              gloss: w.gloss,
              confidence: w.confidence,
            }))
          }
        }
        if (s.stage.startsWith('agentStep')) {
          const p = s.payload as { step?: { title: string; promptSystem: string; promptUser: string; output: string } }
          if (p.step) {
            activeStep.value = Number(s.stage.replace('agentStep', ''))
            agentSteps.push(p.step)
          }
        }
        if (s.stage === 'executing') {
          // 执行态细节已在 s.detail
        }
        if (s.stage === 'done') {
          const p = s.payload as { gloss?: GlossResult; agent?: AgentResult }
          if (p.gloss) gloss.value = p.gloss
          if (p.agent) {
            agent.value = p.agent
            highlightTokens.value = p.agent.low_confidence_words
          }
        }
      },
      520,
      canceller
    )
  } catch (err) {
    if (err instanceof DemoCancelledError) {
      error.value = ''
    } else {
      error.value = '演示生成失败'
    }
  } finally {
    running.value = false
    mockCanceller = null
  }
}

let mockCanceller: DemoCanceller | null = null

// ---------- 开始 / 取消 ----------
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
    if (e instanceof BackendUnavailableError) {
      // 自动降级 mock（数据无关先行）
      runMock()
    } else {
      error.value = (e as Error).message || '启动失败'
      running.value = false
      stopTimer()
    }
  }
}

function cancel() {
  if (mode.value === 'real' && activeHandle) {
    // 中断前端 fetch（AbortController）+ 通知后端停止（cancelStream 内部已 abort）
    cancelStream(activeHandle)
  }
  if (mode.value === 'mock' && mockCanceller) {
    mockCanceller.cancelled = true
  }
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

onBeforeUnmount(() => {
  if (videoPreview.value) URL.revokeObjectURL(videoPreview.value)
  if (running.value) cancel()
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
        <span class="pill" :class="mode === 'real' ? 'pill--ok' : mode === 'mock' ? 'pill--warn' : 'pill--idle'">
          {{ mode === 'real' ? '真实后端（SSE 流式）' : mode === 'mock' ? '本地演示（mock 降级）' : '就绪探测中' }}
        </span>
        <span v-if="running" class="pill pill--idle rc__elapsed">
          {{ elapsed.toFixed(1) }}s
        </span>
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
          缺失：{{ missing.join('、') }}（已自动降级 mock）
        </span>
      </div>
    </header>

    <!-- 上传区 -->
    <section class="upload panel-no">
      <label
        class="drop"
        :class="{ 'drop--err': fileErr, 'drop--filled': file }"
        @dragover.prevent
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
          <p><strong>点击或拖拽视频到此处</strong></p>
          <p class="drop__meta">仅限视频（mp4 / avi / mov / webm），≤ 50MB</p>
        </div>
        <div v-else class="drop__file">
          <p><strong>{{ fileName }}</strong></p>
          <video v-if="videoPreview" :src="videoPreview" class="drop__video" controls muted playsinline />
          <button type="button" class="link" @click.prevent="clearFile">移除</button>
        </div>
      </label>
      <p v-if="fileErr" role="alert" class="rc__err">{{ fileErr }}</p>

      <div class="rc__actions">
        <button type="button" class="btn btn--primary" :disabled="!canStart" @click="start">
          ▶ 开始识别
        </button>
        <button type="button" class="btn" :disabled="!running" @click="cancel">■ 取消</button>
      </div>
    </section>

    <p v-if="error" role="alert" class="rc__err rc__err--run">{{ error }}</p>

    <PipelineFlow :current="stage" />

    <StageLog :log="stageLog" :current="stage" />

    <div class="rc__grid">
      <!-- 左：骨架 -->
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
        />
      </section>

      <!-- 右：切分 -->
      <section class="panel" aria-labelledby="seg-title">
        <h2 id="seg-title" class="panel__title">② 切分层</h2>
        <div v-if="segments.length" class="timeline" aria-hidden="true">
          <div class="timeline__axis" />
          <div
            v-for="(s, i) in segments"
            :key="`sg${i}`"
            class="timeline__seg"
            :style="{
              left: `${(s.startTime / (durationSec || 1)) * 100}%`,
              width: `${((s.endTime - s.startTime) / (durationSec || 1)) * 100}%`,
            }"
          >
            段{{ i + 1 }}
          </div>
        </div>
        <p v-else class="vis__hint">（识别开始后展示运动能量切分结果）</p>
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
  gap: 0.8rem;
}

.drop {
  display: block;
  border: 2px dashed var(--color-border, #cbd5e1);
  border-radius: var(--radius, 0.75rem);
  padding: 1.4rem;
  text-align: center;
  cursor: pointer;
  background: var(--color-surface, #fff);
  transition: border-color 0.15s ease, background 0.15s ease;
}

.drop--filled {
  border-style: solid;
  cursor: default;
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
  font-size: 2rem;
}

.drop__meta {
  color: var(--color-muted-text, #6b7280);
  font-size: 0.85rem;
}

.drop__file {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.drop__video {
  max-width: 100%;
  max-height: 220px;
  border-radius: 0.5rem;
}

.link {
  background: none;
  border: none;
  color: var(--color-primary, #2563eb);
  cursor: pointer;
  text-decoration: underline;
}

.rc__actions {
  display: flex;
  gap: 0.6rem;
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

.timeline {
  position: relative;
  height: 2.2rem;
  margin: 0.5rem 0;
  background: var(--color-muted, #f3f4f6);
  border-radius: 0.4rem;
  overflow: hidden;
}

.timeline__axis {
  position: absolute;
  inset: 0;
}

.timeline__seg {
  position: absolute;
  top: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.72rem;
  background: rgba(37, 99, 235, 0.25);
  border-left: 1px solid var(--color-primary, #2563eb);
  border-right: 1px solid var(--color-primary, #2563eb);
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
