<script setup lang="ts">
// 智能体层三步提示管线展示（演示页 / 识别页共用）。
// 安全红线：所有后端文本经 highlightTokens()（先转义再包 <mark>）后通过 v-html 输出，
// 转义逻辑集中在 utils/sanitize.ts，本组件不自行拼接 HTML。
import { highlightTokens as hl } from '@/utils/sanitize'
import type { AgentResult } from '@/api/types'

interface StepCard {
  name?: string
  title: string
  promptSystem: string
  promptUser: string
  output: string
}

const props = defineProps<{
  steps: StepCard[]
  /** 需要高亮的 token（<lowconf:...>、任务标识等） */
  highlightTokens?: string[]
  /** 当前激活的步骤（1-based），用于动画高亮 */
  activeStep?: number
  agent: AgentResult | null
}>()
</script>

<template>
  <section class="panel" aria-labelledby="agent-title">
    <h2 id="agent-title" class="panel__title">智能体层：LLM 三步提示管线</h2>

    <!-- 后端未下发分步 prompt/输出时不编造：给出诚实说明 -->
    <p v-if="!steps.length" class="steps__empty">
      后端未下发智能体分步数据（SSE 仅提供最终
      <code>agent_done</code>
      结果）。此处不展示任何前端编造的提示词；下方为后端真实产出的 AgentSchema。
    </p>

    <div v-else class="steps">
      <article
        v-for="(s, i) in steps"
        :key="s.name ?? i"
        class="stepcard"
        :class="{
          'stepcard--active': activeStep === i + 1,
          'stepcard--done': activeStep !== undefined && activeStep > i + 1,
        }"
      >
        <h3 class="stepcard__title">{{ s.title }}</h3>
        <dl class="stepcard__io">
          <dt>System Prompt</dt>
          <dd class="mono" v-html="hl(s.promptSystem, props.highlightTokens ?? [])" />
          <dt>User Prompt</dt>
          <dd class="mono" v-html="hl(s.promptUser, props.highlightTokens ?? [])" />
          <dt>输出</dt>
          <dd class="mono stepcard__out" v-html="hl(s.output, props.highlightTokens ?? [])" />
        </dl>
      </article>
    </div>

    <div v-if="agent" class="agentout">
      <p class="agentout__badge">智能体层输出（符合 AGENT_SCHEMA）</p>
      <p class="agentout__text">「{{ agent.natural_language }}」</p>
      <dl class="kv kv--inline">
        <div>
          <dt>intent</dt>
          <dd><code>{{ agent.intent }}</code></dd>
        </div>
        <div>
          <dt>task</dt>
          <dd><code>{{ agent.task }}</code></dd>
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

      <!-- 澄清提示行（固定 UI 文案，非 LLM 输出；仅 clarification=true 时渲染） -->
      <p v-if="agent.clarification" class="agentout__clarify" role="status">
        未识别清楚，可以再比一次吗？低置信词：
        <code v-for="w in agent.low_confidence_words" :key="w" class="tag">{{ w }}</code>
      </p>
    </div>
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

.steps {
  display: grid;
  gap: 0.75rem;
}

.steps__empty {
  margin: 0 0 0.75rem;
  font-size: 0.85rem;
  color: var(--color-muted-text, #6b7280);
}

.steps__empty code {
  background: var(--color-muted, #f3f4f6);
  padding: 0.1rem 0.3rem;
  border-radius: 0.3rem;
  font-size: 0.8rem;
}

.stepcard {
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: 0.6rem;
  padding: 0.8rem;
}

.stepcard--active {
  border-color: var(--color-primary, #2563eb);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
}

.stepcard--done {
  opacity: 0.78;
}

.stepcard__title {
  margin: 0 0 0.5rem;
  font-size: 0.95rem;
}

.stepcard__io {
  margin: 0;
  display: grid;
  gap: 0.2rem;
  font-size: 0.82rem;
}

.stepcard__io dt {
  font-weight: 600;
  margin-top: 0.3rem;
}

.mono {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: var(--color-muted, #f9fafb);
  border-radius: 0.4rem;
  padding: 0.5rem;
  border: 1px solid var(--color-border, #e5e7eb);
}

.stepcard__out {
  border-left: 3px solid var(--color-primary, #2563eb);
}

.agentout {
  margin-top: 0.9rem;
}

.agentout__badge {
  margin: 0 0 0.3rem;
  font-size: 0.8rem;
  color: var(--color-muted-text, #6b7280);
}

.agentout__text {
  margin: 0 0 0.6rem;
  font-size: 1.05rem;
}

/* 澄清提示行（固定 UI 文案；低置信词沿用 .tag） */
.agentout__clarify {
  margin: 0.6rem 0 0;
  padding: 0.5rem 0.8rem;
  border: 1px dashed #d97706;
  border-radius: 0.5rem;
  background: #fffbeb;
  color: #92400e;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.3rem;
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

.pill {
  padding: 0.12rem 0.6rem;
  border-radius: 999px;
  font-size: 0.8rem;
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
  margin-right: 0.3rem;
  padding: 0.1rem 0.4rem;
  border-radius: 0.3rem;
  background: var(--color-muted, #f3f4f6);
  font-size: 0.78rem;
}
</style>
