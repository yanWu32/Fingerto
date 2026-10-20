<script setup lang="ts">
// 端到端演示页：手语视频 -> 骨架 -> 切分 -> 识别 -> 智能体三步 -> 文本 + 任务执行。
// 契约事实源 tests/contract.py；业务事实源 agent/pipeline.py + configs/agent.yaml。
import { computed, onMounted, onUnmounted, ref, shallowRef } from 'vue'
import SkeletonCanvas from '@/components/SkeletonCanvas.vue'
import PipelineFlow from '@/components/PipelineFlow.vue'
import {
  buildScenario,
  runFullDemo,
  type DemoOutput,
  type PipelineStageName,
  type StageSnapshot,
} from '@/api/demoPipeline'
import { USE_MOCK } from '@/utils/env'
import { formatConfidence } from '@/utils/format'
// 高亮渲染统一走 utils/sanitize.ts：先转义再包 <mark>，有 XSS 单元测试保障
import { highlightTokens } from '@/utils/sanitize'

const scenario = shallowRef(buildScenario(0))
const stage = ref<PipelineStageName>('idle')
const stageLog = ref<StageSnapshot[]>([])
const output = ref<DemoOutput | null>(null)
const running = ref(false)
const error = ref('')
const playing = ref(false)
const activeGloss = ref(-1)
const activeStep = ref(0)

const playerComp = ref<InstanceType<typeof SkeletonCanvas> | null>(null)

const gloss = computed(() => output.value?.gloss ?? null)
const agent = computed(() => output.value?.agent ?? null)
const sc = computed(() => scenario.value)

/** 低置信阈值（configs/agent.yaml confidence.clarify_threshold） */
const CLARIFY_THRESHOLD = 0.4
const HIGH_THRESHOLD = 0.75

function confLevel(c: number): 'high' | 'mid' | 'low' {
  if (c >= HIGH_THRESHOLD) return 'high'
  if (c >= CLARIFY_THRESHOLD) return 'mid'
  return 'low'
}

/** 从 Step3 输出 JSON 中抽取 task */
const decidedTask = computed(() => {
  const raw = sc.value.steps.steps[2]?.output ?? ''
  const m = /"task"\s*:\s*"([^"]+)"/.exec(raw)
  return m?.[1] ?? sc.value.steps.agentResult?.task ?? 'none'
})

const isNoop = computed(() => decidedTask.value === 'none')

/** 点击某个 gloss：跳转骨架到该词起点并高亮其时间区间 */
function playGloss(i: number) {
  const ts = gloss.value?.timestamps[i]
  if (!ts) return
  activeGloss.value = i
  playing.value = false
  playerComp.value?.seekSec(ts[0])
}

async function startDemo() {
  running.value = true
  playing.value = true
  error.value = ''
  stageLog.value = []
  output.value = null
  activeGloss.value = -1
  try {
    // 所有异步流程必须 try/catch（规范 4.4）
    output.value = await runFullDemo(sc.value, (s) => {
      stage.value = s.stage
      stageLog.value = [...stageLog.value, s]
      if (s.stage === 'agentStep1') activeStep.value = 1
      if (s.stage === 'agentStep2') activeStep.value = 2
      if (s.stage === 'agentStep3') activeStep.value = 3
    })
    // 演示结束后自动带高亮循环播放
    playing.value = true
  } catch (e) {
    error.value = e instanceof Error ? e.message : '演示执行失败'
  } finally {
    running.value = false
  }
}

onMounted(startDemo)
onUnmounted(() => playerComp.value?.stop())
</script>

<template>
  <section class="demo">
    <header class="demo__hero">
      <h1 class="demo__title">手语视频 → 文本 · 全链路演示</h1>
      <p class="demo__sub">
        感知 → 切分 → 识别 → 智能体（三步提示） → 应用执行 · 契约由
        <code>tests/contract.py</code> 统一把关
      </p>
      <div class="demo__ctrl">
        <button type="button" class="btn btn--primary" :disabled="running" @click="startDemo">
          {{ running ? '演示进行中…' : '重新演示' }}
        </button>
        <button type="button" class="btn" :aria-pressed="playing" @click="playing = !playing">
          {{ playing ? '⏸ 暂停骨架' : '▶ 播放骨架' }}
        </button>
        <span class="demo__mode" :class="{ 'demo__mode--real': !USE_MOCK }">
          {{ USE_MOCK ? 'mock 数据' : '真实后端' }}
        </span>
      </div>
      <p v-if="error" role="alert" class="demo__error">{{ error }}</p>
    </header>

    <PipelineFlow :current="stage" />

    <!-- 阶段执行流水 -->
    <section class="panel" aria-labelledby="log-title">
      <h2 id="log-title" class="panel__title">阶段执行流水</h2>
      <ol class="log">
        <li v-for="(l, i) in stageLog" :key="i" class="log__item">
          <span class="log__layer">{{ l.layer }}</span>
          <span class="log__label">{{ l.label }}</span>
          <span class="log__detail">{{ l.detail }}</span>
        </li>
      </ol>
    </section>

    <div class="demo__grid">
      <!-- 左：视频 / 骨架 -->
      <section class="panel" aria-labelledby="vis-title">
        <h2 id="vis-title" class="panel__title">① 感知层：视频 → (T, 27, 3) 骨架张量</h2>
        <div class="vis">
          <div class="vis__meta">
            <p><strong>视频源</strong>：{{ sc.sense.source }}</p>
            <p>
              <strong>规格</strong>：{{ sc.sense.width }}×{{ sc.sense.height }} ·
              {{ sc.sense.fps }}fps · {{ sc.sense.durationSec }}s · {{ sc.sense.numFrames }} 帧
            </p>
            <p><strong>提取器</strong>：{{ sc.sense.extractor }}</p>
            <p class="vis__shape">
              骨架张量 <code>{{ sc.skeleton.shape }}</code> · {{ sc.skeleton.layout }}（手 21 + 上身
              6）
            </p>
          </div>
          <SkeletonCanvas
            v-if="sc.frames.length"
            ref="playerComp"
            :frames="sc.frames"
            :fps="sc.sense.fps"
            :highlight-range="
              activeGloss >= 0 && gloss?.timestamps[activeGloss]
                ? gloss.timestamps[activeGloss]
                : undefined
            "
            :playing="playing"
          />
        </div>
      </section>

      <!-- 右：切分 + 识别 -->
      <section class="panel" aria-labelledby="rec-title">
        <h2 id="rec-title" class="panel__title">②③ 切分层 / 识别层</h2>

        <h3 class="sub">切分：{{ sc.segmentation.method }}</h3>
        <dl class="kv">
          <div>
            <dt>能量参考</dt>
            <dd>{{ sc.segmentation.energyReference }}</dd>
          </div>
          <div>
            <dt>速度阈值</dt>
            <dd>{{ sc.segmentation.velocityThreshold }}</dd>
          </div>
          <div>
            <dt>窗口 / 步长</dt>
            <dd>{{ sc.segmentation.windowSize }} / {{ sc.segmentation.stride }}</dd>
          </div>
          <div>
            <dt>平滑</dt>
            <dd>{{ sc.segmentation.smoothing }}</dd>
          </div>
          <div>
            <dt>结果</dt>
            <dd>{{ sc.segmentation.numSegments }} 段 → {{ sc.segmentation.numWindows }} 窗口</dd>
          </div>
        </dl>

        <!-- 时间轴上的段与窗口 -->
        <div class="timeline" aria-hidden="true">
          <div class="timeline__axis" />
          <div
            v-for="(s, i) in sc.segmentation.segments"
            :key="`sg${i}`"
            class="timeline__seg"
            :style="{
              left: `${(s.startTime / sc.sense.durationSec) * 100}%`,
              width: `${((s.endTime - s.startTime) / sc.sense.durationSec) * 100}%`,
            }"
          >
            段{{ i + 1 }}
          </div>
        </div>

        <h3 class="sub">
          识别：{{ sc.recognition.model }}（输入 {{ sc.recognition.inputShape }}）
        </h3>

        <!-- gloss 序列：点击高亮对应骨架区间 -->
        <ul v-if="gloss" class="glosses">
          <li
            v-for="(w, i) in gloss.gloss_sequence"
            :key="`${w}-${i}`"
            class="gloss"
            :class="[
              `gloss--${confLevel(gloss.confidence[i])}`,
              { 'gloss--active': activeGloss === i },
            ]"
          >
            <button type="button" class="gloss__btn" @click="playGloss(i)">
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

        <details class="windows">
          <summary>逐窗口推理原始产出（{{ sc.recognition.numWindows }} 个，点击展开）</summary>
          <table class="wtable">
            <thead>
              <tr>
                <th>窗口时间(s)</th>
                <th>Gloss</th>
                <th>置信度</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(w, i) in sc.recognition.windows" :key="i">
                <td>{{ w.startTime.toFixed(2) }} – {{ w.endTime.toFixed(2) }}</td>
                <td>
                  <strong>{{ w.gloss }}</strong>
                </td>
                <td :class="`c-${confLevel(w.confidence)}`">{{ w.confidence.toFixed(2) }}</td>
              </tr>
            </tbody>
          </table>
          <p class="windows__note">
            重叠窗口经 vote 投票（阈值 0.5）与时间合并后，得到上方 gloss 序列。
          </p>
        </details>
      </section>
    </div>

    <!-- 智能体三步 -->
    <section class="panel" aria-labelledby="agent-title">
      <h2 id="agent-title" class="panel__title">④ 智能体层：LLM 三步提示管线</h2>
      <div class="steps">
        <article
          v-for="(s, i) in sc.steps.steps"
          :key="s.name"
          class="stepcard"
          :class="{
            'stepcard--active': activeStep === i + 1,
            'stepcard--done': activeStep > i + 1,
          }"
        >
          <h3 class="stepcard__title">{{ s.title }}</h3>
          <dl class="stepcard__io">
            <dt>System Prompt</dt>
            <dd class="mono" v-html="highlightTokens(s.promptSystem, sc.steps.highlightTokens)" />
            <dt>User Prompt</dt>
            <dd class="mono" v-html="highlightTokens(s.promptUser, sc.steps.highlightTokens)" />
            <dt>输出</dt>
            <dd
              class="mono stepcard__out"
              v-html="highlightTokens(s.output, sc.steps.highlightTokens)"
            />
          </dl>
        </article>
      </div>

      <div v-if="agent" class="agentout">
        <p class="agentout__badge">智能体层输出（符合 AGENT_SCHEMA）</p>
        <p class="agentout__text">「{{ agent.natural_language }}」</p>
        <dl class="kv kv--inline">
          <div>
            <dt>intent</dt>
            <dd>
              <code>{{ agent.intent }}</code>
            </dd>
          </div>
          <div>
            <dt>task</dt>
            <dd>
              <code>{{ agent.task }}</code>
            </dd>
          </div>
          <div>
            <dt>clarification</dt>
            <dd>
              <span class="pill" :class="agent.clarification ? 'pill--warn' : 'pill--ok'">
                {{ agent.clarification ? '需要澄清' : '否' }}
              </span>
            </dd>
          </div>
          <div v-if="agent.low_confidence_words.length">
            <dt>low_confidence_words</dt>
            <dd>
              <code v-for="w in agent.low_confidence_words" :key="w" class="tag">{{ w }}</code>
            </dd>
          </div>
        </dl>
      </div>
    </section>

    <!-- 应用执行 -->
    <section class="panel" aria-labelledby="exec-title">
      <h2 id="exec-title" class="panel__title">⑤ 应用层：任务执行与环境变化</h2>
      <div class="exec">
        <div class="exec__code">
          <p class="exec__api mono">{{ sc.execution.apiPath }}</p>
          <dl class="kv">
            <div>
              <dt>任务名</dt>
              <dd>{{ isNoop ? '（无可执行任务）' : sc.execution.taskName }}</dd>
            </div>
            <div>
              <dt>task id</dt>
              <dd>
                <code>{{ decidedTask }}</code>
              </dd>
            </div>
            <div>
              <dt>控制器</dt>
              <dd>{{ sc.execution.controllerStatus }}</dd>
            </div>
          </dl>
        </div>
        <table class="envtable">
          <thead>
            <tr>
              <th>环境项</th>
              <th>执行前</th>
              <th>执行后</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="c in sc.execution.changes" :key="c.label">
              <td>{{ c.label }}</td>
              <td>{{ c.from }}</td>
              <td :class="{ 'c-high': c.from !== c.to }">{{ c.to }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-if="isNoop" class="exec__noop">
        task 为 <code>none</code>，不触发任何设备控制（红灯验证：只有 command 意图才执行）。
      </p>
    </section>

    <!-- 最终文本 -->
    <section class="final" aria-live="polite">
      <p class="final__label">最终手语转写文本</p>
      <p class="final__text">{{ agent?.natural_language ?? '（等待演示完成）' }}</p>
    </section>
  </section>
</template>

<style scoped>
.demo {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.demo__hero {
  text-align: center;
  padding: 1.25rem 0.5rem 0.5rem;
}

.demo__title {
  font-size: clamp(1.4rem, 3.5vw, 2rem);
  margin: 0 0 0.4rem;
}

.demo__sub {
  margin: 0 0 1rem;
  color: var(--color-muted-text, #6b7280);
}

.demo__ctrl {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.demo__mode {
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  background: var(--color-muted, #f3f4f6);
  font-size: 0.8rem;
}

.demo__mode--real {
  background: #dcfce7;
  color: #166534;
}

.demo__error {
  color: var(--color-danger, #dc2626);
}

.btn {
  padding: 0.5rem 1.1rem;
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: var(--radius, 0.75rem);
  background: var(--color-surface, #fff);
  cursor: pointer;
  font-size: 0.95rem;
  transition: all 0.15s ease;
}

.btn:hover:not(:disabled) {
  border-color: var(--color-primary, #2563eb);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn--primary {
  background: var(--color-primary, #2563eb);
  border-color: transparent;
  color: #fff;
}

.btn--primary:hover:not(:disabled) {
  background: var(--color-primary-hover, #1d4ed8);
}

.demo__grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 1.25rem;
}

@media (max-width: 900px) {
  .demo__grid {
    grid-template-columns: minmax(0, 1fr);
  }
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

.sub {
  margin: 1rem 0 0.5rem;
  font-size: 0.95rem;
}

/* 阶段流水 */
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

/* 键值表 */
.kv {
  margin: 0.5rem 0 0;
  display: grid;
  gap: 0.3rem;
  font-size: 0.85rem;
}

.kv > div {
  display: flex;
  gap: 0.5rem;
}

.kv dt {
  color: var(--color-muted-text, #6b7280);
  min-width: 6.5rem;
  flex: 0 0 auto;
}

.kv dd {
  margin: 0;
}

.kv--inline {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1.25rem;
}

.kv--inline > div {
  display: flex;
  gap: 0.4rem;
  align-items: center;
}

.vis {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.vis__meta {
  font-size: 0.85rem;
  color: var(--color-muted-text, #6b7280);
}

.vis__meta p {
  margin: 0.15rem 0;
}

.vis__shape {
  margin-top: 0.4rem;
}

/* 时间轴 */
.timeline {
  position: relative;
  height: 2rem;
  margin: 0.6rem 0 0.4rem;
  background: var(--color-muted, #f3f4f6);
  border-radius: 0.4rem;
  overflow: hidden;
}

.timeline__axis {
  position: absolute;
  left: 0;
  right: 0;
  top: 50%;
  height: 2px;
  background: var(--color-border, #e5e7eb);
}

.timeline__seg {
  position: absolute;
  top: 0.3rem;
  bottom: 0.3rem;
  border-radius: 0.3rem;
  background: #dbeafe;
  border: 1px solid var(--color-primary, #2563eb);
  font-size: 0.7rem;
  display: grid;
  place-items: center;
  color: #1e40af;
  overflow: hidden;
  white-space: nowrap;
}

/* gloss 序列 */
.glosses {
  list-style: none;
  margin: 0.5rem 0 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(6.5rem, 1fr));
  gap: 0.5rem;
}

.gloss__btn {
  width: 100%;
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: 0.5rem;
  background: var(--color-surface, #fff);
  padding: 0.5rem;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  align-items: flex-start;
  text-align: left;
  transition: all 0.15s ease;
}

.gloss__btn:hover {
  border-color: var(--color-primary, #2563eb);
}

.gloss--active .gloss__btn {
  border-color: var(--color-primary, #2563eb);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
}

.gloss__word {
  font-size: 1.25rem;
  font-weight: 700;
}

.gloss__conf {
  font-size: 0.75rem;
  font-variant-numeric: tabular-nums;
}

.gloss__bar {
  display: block;
  width: 100%;
  height: 0.3rem;
  background: var(--color-muted, #f3f4f6);
  border-radius: 999px;
  overflow: hidden;
}

.gloss__fill {
  display: block;
  height: 100%;
  border-radius: 999px;
}

.gloss--high .gloss__fill {
  background: var(--color-success, #16a34a);
}
.gloss--mid .gloss__fill {
  background: var(--color-warning, #d97706);
}
.gloss--low .gloss__fill {
  background: var(--color-danger, #dc2626);
}

.gloss--low .gloss__word {
  color: var(--color-danger, #dc2626);
}

.gloss__ts {
  font-size: 0.7rem;
  color: var(--color-muted-text, #6b7280);
  font-variant-numeric: tabular-nums;
}

/* 窗口表 */
.windows {
  margin-top: 0.85rem;
  font-size: 0.85rem;
}

.windows summary {
  cursor: pointer;
  color: var(--color-primary, #2563eb);
}

.wtable {
  width: 100%;
  border-collapse: collapse;
  margin-top: 0.5rem;
  font-size: 0.82rem;
}

.wtable th,
.wtable td {
  border: 1px solid var(--color-border, #e5e7eb);
  padding: 0.35rem 0.5rem;
  text-align: left;
}

.wtable th {
  background: var(--color-muted, #f9fafb);
}

.windows__note {
  color: var(--color-muted-text, #6b7280);
  font-size: 0.78rem;
  margin: 0.4rem 0 0;
}

.c-high {
  color: var(--color-success, #16a34a);
}
.c-mid {
  color: var(--color-warning, #d97706);
}
.c-low {
  color: var(--color-danger, #dc2626);
}

/* 智能体三步 */
.steps {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
  gap: 0.85rem;
}

.stepcard {
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: 0.6rem;
  padding: 0.75rem;
  background: var(--color-surface, #fff);
  transition: all 0.2s ease;
}

.stepcard--active {
  border-color: var(--color-primary, #2563eb);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
}

.stepcard--done {
  border-color: var(--color-success, #16a34a);
}

.stepcard__title {
  margin: 0 0 0.5rem;
  font-size: 0.95rem;
}

.stepcard__io {
  margin: 0;
  font-size: 0.8rem;
}

.stepcard__io dt {
  color: var(--color-muted-text, #6b7280);
  margin-top: 0.4rem;
}

.stepcard__io dd {
  margin: 0.15rem 0 0;
}

.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.75rem;
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--color-muted, #f9fafb);
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: 0.4rem;
  padding: 0.4rem;
  max-height: 9rem;
  overflow-y: auto;
}

.stepcard__out {
  background: #f0fdf4;
  border-color: #bbf7d0;
}

.agentout {
  margin-top: 1rem;
  padding: 0.85rem;
  border-radius: 0.6rem;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
}

.agentout__badge {
  margin: 0;
  font-size: 0.75rem;
  color: #1e40af;
}

.agentout__text {
  margin: 0.3rem 0 0.6rem;
  font-size: 1.15rem;
  font-weight: 600;
}

.pill {
  padding: 0.1rem 0.5rem;
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

.tag {
  display: inline-block;
  margin-right: 0.3rem;
  padding: 0.1rem 0.4rem;
  border-radius: 0.3rem;
  background: var(--color-muted, #f3f4f6);
}

/* 执行 */
.exec {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr);
  gap: 1rem;
}

@media (max-width: 760px) {
  .exec {
    grid-template-columns: minmax(0, 1fr);
  }
}

.exec__api {
  margin: 0 0 0.5rem;
  background: #111827;
  color: #d1fae7;
  padding: 0.5rem;
  border-radius: 0.4rem;
}

.envtable {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.envtable th,
.envtable td {
  border: 1px solid var(--color-border, #e5e7eb);
  padding: 0.4rem 0.55rem;
  text-align: left;
}

.envtable th {
  background: var(--color-muted, #f9fafb);
}

.exec__noop {
  margin: 0.6rem 0 0;
  font-size: 0.85rem;
  color: var(--color-muted-text, #6b7280);
}

/* 最终文本 */
.final {
  margin-top: 0.5rem;
  padding: 1.25rem;
  border-radius: var(--radius, 0.75rem);
  background: linear-gradient(135deg, #1e3a8a, #2563eb);
  color: #fff;
  text-align: center;
}

.final__label {
  margin: 0 0 0.35rem;
  font-size: 0.85rem;
  opacity: 0.85;
}

.final__text {
  margin: 0;
  font-size: clamp(1.15rem, 3vw, 1.6rem);
  font-weight: 700;
}

/* v-html 高亮 token */
.tk {
  background: #fde68a;
  color: #92400e;
  padding: 0 0.15rem;
  border-radius: 0.2rem;
  font-weight: 600;
}
</style>
