<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { runPipeline } from '@/api/pipeline'
import type { AgentResult, GlossResult } from '@/api/types'
import { USE_MOCK } from '@/utils/env'
import { formatConfidence, formatTimeRange } from '@/utils/format'

// 首页：通过统一 API 层跑通「识别 → 智能体」链路。
// VITE_USE_MOCK=true 时走内置 mock；置 false 即无缝切换为真实后端 /api/pipeline。
const recognition = ref<GlossResult | null>(null)
const agent = ref<AgentResult | null>(null)

const loading = ref(false)
const error = ref('')

async function runDemo() {
  loading.value = true
  error.value = ''
  try {
    // 所有接口请求必须 try/catch（规范 4.4）
    const res = await runPipeline()
    recognition.value = res.recognition
    agent.value = res.agent
  } catch (e) {
    error.value = e instanceof Error ? e.message : '加载失败'
  } finally {
    loading.value = false
  }
}

onMounted(runDemo)
</script>

<template>
  <section class="home">
    <section class="home__hero">
      <h1 class="home__title">基于 LLM 智能体的手语对话系统</h1>
      <p class="home__subtitle">词级 / 短语级手语对话演示 · 感知 → 识别 → 智能体 → 应用 四层解耦</p>
    </section>

    <section class="home__demo" aria-labelledby="demo-title">
      <div class="home__demo-head">
        <h2 id="demo-title">识别链路演示（{{ USE_MOCK ? 'mock 数据' : '真实后端' }}）</h2>
        <button type="button" class="home__btn" :disabled="loading" @click="runDemo">
          {{ loading ? '加载中…' : '重新演示' }}
        </button>
      </div>

      <p v-if="error" role="alert" class="home__error">
        {{ error }}
      </p>

      <div v-if="recognition" class="home__card">
        <h3 class="home__card-title">识别层输出（gloss 序列 + 置信度）</h3>
        <ul class="gloss-list">
          <li v-for="(word, i) in recognition.gloss_sequence" :key="i" class="gloss-item">
            <span class="gloss-item__word">{{ word }}</span>
            <span class="gloss-item__conf">
              置信度 {{ formatConfidence(recognition.confidence[i]) }}
            </span>
            <span class="gloss-item__time">
              {{ formatTimeRange(recognition.timestamps[i]) }}
            </span>
          </li>
        </ul>
      </div>

      <div v-if="agent" class="home__card">
        <h3 class="home__card-title">智能体层输出</h3>
        <p class="agent-text">
          {{ agent.natural_language }}
        </p>
        <p class="agent-meta">
          意图：<code>{{ agent.intent }}</code> · 任务：<code>{{ agent.task }}</code> ·
          需澄清：<code>{{ agent.clarification ? '是' : '否' }}</code>
          <template v-if="agent.low_confidence_words.length">
            · 低置信词：<code>{{ agent.low_confidence_words.join('、') }}</code>
          </template>
        </p>
      </div>
    </section>
  </section>
</template>

<style scoped>
.home {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.home__hero {
  text-align: center;
  padding: 2rem 1rem;
}

.home__title {
  font-size: clamp(1.5rem, 4vw, 2.25rem);
  margin: 0 0 0.5rem;
}

.home__subtitle {
  color: var(--color-muted-text, #6b7280);
  margin: 0;
}

.home__demo-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1rem;
}

.home__btn {
  padding: 0.5rem 1.1rem;
  border: none;
  border-radius: var(--radius, 0.75rem);
  background: var(--color-primary, #2563eb);
  color: #fff;
  font-size: 0.95rem;
  cursor: pointer;
  transition: background 0.15s ease;
}

.home__btn:hover:not(:disabled) {
  background: var(--color-primary-hover, #1d4ed8);
}

.home__btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.home__error {
  color: var(--color-danger, #dc2626);
}

.home__card {
  background: var(--color-surface, #fff);
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: var(--radius, 0.75rem);
  padding: 1.25rem;
  box-shadow: var(--shadow-sm);
}

.home__card-title {
  margin: 0 0 0.75rem;
  font-size: 1.05rem;
}

.gloss-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.gloss-item {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  padding: 0.6rem 0.9rem;
  border-radius: 0.5rem;
  background: var(--color-muted, #f3f4f6);
}

.gloss-item__word {
  font-size: 1.25rem;
  font-weight: 600;
}

.gloss-item__conf,
.gloss-item__time {
  font-size: 0.8rem;
  color: var(--color-muted-text, #6b7280);
}

.agent-text {
  font-size: 1.1rem;
  margin: 0 0 0.5rem;
}

.agent-meta {
  margin: 0;
  color: var(--color-muted-text, #6b7280);
}

code {
  background: var(--color-muted, #f3f4f6);
  padding: 0.1rem 0.35rem;
  border-radius: 0.3rem;
  font-size: 0.85em;
}
</style>
