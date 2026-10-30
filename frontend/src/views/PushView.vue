<script setup lang="ts">
// 信息推送页（/push）：把一条推送文本（天气/通知）转换为「手语词序列 + 自然语言 + 骨架动画」。
// 数据链路（红线：不得编造）：
// - 推送文本不进识别层；gloss 序列由后端「信息词→手语词」映射表构造（confidence=1.0，source=push_lexicon）；
// - natural_language / intent / task 由后端真实三步 LLM 管线产出；
// - skeleton_frames 为后端返回的逐词骨架片段，缺素材的词跳过（不造帧）。
// 后端 /push 落地前本页调用将得到 404/503，错误原样透出（不 mock 降级）。
import { computed, onMounted, reactive, ref } from 'vue'
import GlossBoard from '@/components/GlossBoard.vue'
import SkeletonCanvas from '@/components/SkeletonCanvas.vue'
import SpeechButton from '@/components/SpeechButton.vue'
import { ensureBackendReady, BackendUnavailableError } from '@/api/backend'
import { fetchPushLexicon, pushInfo, type PushLexicon } from '@/api/push'
import type { PushResult } from '@/api/types'
import type { SkeletonFrame } from '@/utils/skeleton'

// ---- 后端就绪 ----
const backendErr = ref('')
const backendOk = ref(false)

onMounted(async () => {
  try {
    await ensureBackendReady()
    backendOk.value = true
    // 可选：拉取可推送词表（失败不阻塞）
    lexicon.value = await fetchPushLexicon()
  } catch (e) {
    backendOk.value = false
    backendErr.value =
      e instanceof BackendUnavailableError
        ? `${e.message}（请确认已启动：python serve.py）`
        : (e as Error).message
  }
})

// ---- 推送表单 ----
type Kind = 'weather' | 'notice'
const kind = ref<Kind>('weather')
const kindLabel: Record<Kind, string> = { weather: '天气', notice: '通知' }

/** 预置示例文本（前端固定的输入辅助项，不参与任何结果生成） */
const PRESETS: Record<Kind, string[]> = {
  weather: ['今日天气：晴 22℃', '明日天气：小雨 18℃，注意带伞'],
  notice: ['14:00 康复训练通知', '本周五上午手语角活动，欢迎参加'],
}
const text = ref(PRESETS.weather[0])
const pushing = ref(false)
const pushErr = ref('')

function onKindChange() {
  // 切换类型时带上该类型的第一个示例，减少空输入
  text.value = PRESETS[kind.value][0]
}

// 可推送词表（可选接口返回；仅作提示展示）
const lexicon = ref<PushLexicon | null>(null)

// ---- 结果 ----
const result = ref<PushResult | null>(null)

/** 后端返回的逐词骨架片段 → 拼接成可播放序列（缺素材的词跳过，不造帧） */
const skeletonFrames = computed<SkeletonFrame[]>(() => {
  const parts = result.value?.skeleton_frames ?? []
  const out: SkeletonFrame[] = []
  for (const p of parts) {
    if (p?.frames?.length) out.push(...(p.frames as SkeletonFrame[]))
  }
  return out
})

const skeletonFps = 30 // 骨架素材统一按 30fps 播放（后端约定）
const hasSkeleton = computed(() => skeletonFrames.value.length > 0)
/** 有素材的词（用于标注「已含动画」；无素材词不进动画区） */
const skeletonWords = computed(() =>
  (result.value?.skeleton_frames ?? [])
    .filter((p) => p?.frames?.length)
    .map((p) => p.word)
)

// ---- 推送记录（本页会话内，纯前端状态，不持久化） ----
const history = reactive<{ time: string; kind: Kind; text: string }[]>([])

function fmtNow(): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

async function doPush() {
  if (pushing.value) return
  const t = text.value.trim()
  if (!t) {
    pushErr.value = '请输入推送文本。'
    return
  }
  pushing.value = true
  pushErr.value = ''
  result.value = null
  try {
    const res = await pushInfo({ kind: kind.value, text: t })
    result.value = res
    history.unshift({ time: fmtNow(), kind: kind.value, text: t })
  } catch (e) {
    pushErr.value = (e as Error).message || '推送失败'
  } finally {
    pushing.value = false
  }
}
</script>

<template>
  <div class="push">
    <header class="push__head">
      <h1 class="push__title">信息推送</h1>
      <p class="push__sub">
        把天气 / 通知类文本转换为手语词序列与语音播报：词序列来自后端
        <strong>推送词表映射</strong>（非视频识别），自然语言与意图由后端
        <strong>真实 LLM 三步管线</strong>产出。
      </p>
      <p v-if="!backendOk && backendErr" role="alert" class="push__err">{{ backendErr }}</p>
    </header>

    <template v-if="backendOk">
      <!-- 推送输入区 -->
      <section class="panel" aria-labelledby="push-form-title">
        <h2 id="push-form-title" class="panel__title">发起推送</h2>
        <div class="push__form">
          <label class="push__field">
            <span class="push__field-label">类型</span>
            <select v-model="kind" class="push__select" @change="onKindChange">
              <option value="weather">天气</option>
              <option value="notice">通知</option>
            </select>
          </label>

          <label class="push__field">
            <span class="push__field-label">示例文本</span>
            <select v-model="text" class="push__select">
              <option v-for="p in PRESETS[kind]" :key="p" :value="p">{{ p }}</option>
            </select>
          </label>

          <label class="push__field push__field--grow">
            <span class="push__field-label">推送文本</span>
            <input
              v-model="text"
              type="text"
              class="push__input"
              :placeholder="`输入要推送的${kindLabel[kind]}文本`"
              @keyup.enter="doPush"
            >
          </label>

          <button type="button" class="btn btn--primary" :disabled="pushing" @click="doPush">
            {{ pushing ? '推送中…' : '📣 推送' }}
          </button>
        </div>
        <p v-if="lexicon" class="push__lexicon">
          当前可推送词表：{{ Object.keys(lexicon.kinds || {}).join(' / ') || '—' }}
        </p>
        <p v-if="pushErr" role="alert" class="push__err">{{ pushErr }}</p>
      </section>

      <!-- 推送结果区 -->
      <template v-if="result">
        <GlossBoard :gloss="result.gloss" source-label="来源：推送词表（非识别产出）" />

        <section class="panel" aria-labelledby="push-nl-title">
          <h2 id="push-nl-title" class="panel__title">智能体层输出（真实 LLM 管线）</h2>
          <div class="push__nl">
            <p class="push__nl-text">「{{ result.agent.natural_language }}」</p>
            <SpeechButton :text="result.agent.natural_language" />
          </div>
          <dl class="kv kv--inline">
            <div>
              <dt>intent</dt>
              <dd><code>{{ result.agent.intent }}</code></dd>
            </div>
            <div>
              <dt>task</dt>
              <dd><code>{{ result.agent.task }}</code></dd>
            </div>
            <div v-if="result.source">
              <dt>source</dt>
              <dd><code>{{ result.source }}</code></dd>
            </div>
          </dl>
        </section>

        <!-- 骨架动画区（可选素材，缺失时优雅降级为文字提示，不造帧） -->
        <section class="panel" aria-labelledby="push-skel-title">
          <h2 id="push-skel-title" class="panel__title">手语词骨架动画</h2>
          <SkeletonCanvas
            v-if="hasSkeleton"
            :frames="skeletonFrames"
            :fps="skeletonFps"
            :playing="true"
          />
          <p v-else class="push__hint">该词暂无骨架动画素材。</p>
          <p v-if="hasSkeleton" class="push__hint">
            含动画素材的词：{{ skeletonWords.join('、') }}
          </p>
        </section>
      </template>

      <!-- 推送记录条（会话内） -->
      <section v-if="history.length" class="panel">
        <h2 class="panel__title">本次会话推送记录</h2>
        <ol class="push__history">
          <li v-for="(h, i) in history" :key="`${h.time}-${i}`" class="push__history-item">
            <span class="push__history-time">{{ h.time }}</span>
            <span class="push__history-kind">{{ kindLabel[h.kind] }}</span>
            <span class="push__history-text">{{ h.text }}</span>
          </li>
        </ol>
      </section>
    </template>
  </div>
</template>

<style scoped>
.push {
  display: flex;
  flex-direction: column;
  gap: 1.1rem;
}

.push__title {
  margin: 0 0 0.3rem;
  font-size: 1.6rem;
}

.push__sub {
  margin: 0;
  color: var(--color-muted-text, #6b7280);
}

.push__form {
  display: flex;
  flex-wrap: wrap;
  gap: 0.7rem;
  align-items: flex-end;
}

.push__field {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.push__field--grow {
  flex: 1 1 260px;
}

.push__field-label {
  font-size: 0.82rem;
  color: var(--color-muted-text, #6b7280);
}

.push__select,
.push__input {
  padding: 0.5rem 0.7rem;
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: 0.5rem;
  background: var(--color-surface, #fff);
  font-size: 0.92rem;
  min-width: 0;
}

.push__input {
  width: 100%;
}

.push__lexicon {
  margin: 0.6rem 0 0;
  font-size: 0.82rem;
  color: var(--color-muted-text, #6b7280);
}

.push__err {
  color: #dc2626;
  margin: 0.4rem 0 0;
  font-weight: 600;
}

.push__nl {
  display: flex;
  align-items: center;
  gap: 0.9rem;
  flex-wrap: wrap;
}

.push__nl-text {
  margin: 0;
  font-size: 1.15rem;
  font-weight: 600;
}

.push__hint {
  margin: 0.4rem 0 0;
  color: var(--color-muted-text, #6b7280);
  font-size: 0.85rem;
}

.push__history {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.push__history-item {
  display: flex;
  gap: 0.6rem;
  align-items: baseline;
  font-size: 0.88rem;
}

.push__history-time {
  font-variant-numeric: tabular-nums;
  color: var(--color-muted-text, #6b7280);
  flex: none;
}

.push__history-kind {
  flex: none;
  padding: 0.05rem 0.45rem;
  border-radius: 999px;
  background: var(--color-muted, #f3f4f6);
  font-size: 0.75rem;
}

.push__history-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.kv {
  margin: 0.5rem 0 0;
  display: grid;
  gap: 0.3rem;
  font-size: 0.85rem;
}

.kv--inline {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1.2rem;
}

.kv--inline > div {
  display: flex;
  gap: 0.4rem;
  align-items: center;
}

.kv dt {
  color: var(--color-muted-text, #6b7280);
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
</style>
