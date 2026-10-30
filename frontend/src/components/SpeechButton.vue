<script setup lang="ts">
// 语音播报按钮（/recognize 最终输出 与 /push 结果区共用）。
// 使用浏览器原生 window.speechSynthesis（zh-CN），不引第三方库、不产生任何网络请求。
// 三态：idle（🔊 播报）→ speaking（⏹ 停止）→ done（✅ 已朗读，可重播）。
// 红线：只朗读调用方传入的 text（调用方保证其为后端真实产出），本组件不生成任何文案。
import { onBeforeUnmount, ref, watch } from 'vue'

const props = defineProps<{
  /** 待朗读文本；为空时按钮禁用 */
  text: string
}>()

type SpeechState = 'idle' | 'speaking' | 'done'
const state = ref<SpeechState>('idle')

/** 浏览器是否支持 speechSynthesis（SSR / 老浏览器兜底） */
const supported =
  typeof window !== 'undefined' && typeof window.speechSynthesis !== 'undefined'

function speak() {
  if (!supported || !props.text) return
  // 先清掉可能残留的队列，避免叠音
  window.speechSynthesis.cancel()
  const utter = new SpeechSynthesisUtterance(props.text)
  utter.lang = 'zh-CN'
  utter.onend = () => {
    // 用户中途 stop() 会先置 done，这里只在自然读完时收尾
    if (state.value === 'speaking') state.value = 'done'
  }
  utter.onerror = () => {
    state.value = 'idle'
  }
  state.value = 'speaking'
  window.speechSynthesis.speak(utter)
}

/** 中途停止（停止后进入 done 态，可再点重播） */
function stop() {
  if (!supported) return
  window.speechSynthesis.cancel()
  state.value = 'done'
}

/** 文本变化（新一轮识别/推送）时回到初始态 */
watch(
  () => props.text,
  () => {
    if (supported) window.speechSynthesis.cancel()
    state.value = 'idle'
  }
)

// 页面卸载时取消朗读，避免切页后声音残留（需求 1.1 明确要求）
onBeforeUnmount(() => {
  if (supported) window.speechSynthesis.cancel()
})

defineExpose({ stop })
</script>

<template>
  <button
    type="button"
    class="speech-btn"
    :class="`speech-btn--${state}`"
    :disabled="!supported || !text"
    :title="
      !supported
        ? '当前浏览器不支持语音合成'
        : !text
          ? '暂无可朗读的后端结果'
          : state === 'speaking'
            ? '点击停止朗读'
            : '朗读后端返回的自然语言'
    "
    @click="state === 'speaking' ? stop() : speak()"
  >
    <template v-if="!supported">🔇 不支持播报</template>
    <template v-else-if="state === 'speaking'">⏹ 停止</template>
    <template v-else-if="state === 'done'">✅ 已朗读（重播）</template>
    <template v-else>🔊 播报</template>
  </button>
</template>

<style scoped>
.speech-btn {
  flex: none;
  padding: 0.4rem 0.9rem;
  border-radius: 0.5rem;
  border: 1px solid var(--color-border, #e5e7eb);
  background: var(--color-surface, #fff);
  cursor: pointer;
  font-size: 0.9rem;
  transition:
    background 0.15s ease,
    border-color 0.15s ease;
}

.speech-btn:hover:enabled {
  background: var(--color-muted, #f3f4f6);
}

.speech-btn--speaking {
  border-color: var(--color-primary, #2563eb);
  color: var(--color-primary, #2563eb);
  background: #eff6ff;
}

.speech-btn--done {
  border-color: #16a34a;
  color: #166534;
  background: #f0fdf4;
}

.speech-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
