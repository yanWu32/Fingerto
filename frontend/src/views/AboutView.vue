<script setup lang="ts">
// 占位「关于」页：说明系统架构与当前前端进度（后续细化）
const layers = [
  { name: '感知层', desc: 'OpenCV 采集 + MediaPipe 27 点骨架提取' },
  { name: '识别层', desc: 'ST-GCN 输出 gloss 序列 + 置信度 + 时间戳' },
  { name: '智能体层', desc: 'LLM 三步管线：确认 → 翻译 → 决策' },
  { name: '应用层', desc: '本前端 SPA：对话演示 + 模拟智能家居场景' },
]
</script>

<template>
  <section class="about">
    <h1 class="about__title">关于本系统</h1>
    <p class="about__lead">
      本项目为「基于 LLM 智能体的手语对话系统」的前端演示界面，采用前后端分离架构， 后端仅提供 HTTP
      JSON 接口，前端负责渲染与交互。
    </p>

    <h2 class="about__subtitle">四层架构</h2>
    <ul class="about__layers">
      <li v-for="layer in layers" :key="layer.name" class="about__layer">
        <strong>{{ layer.name }}</strong>
        <span>{{ layer.desc }}</span>
      </li>
    </ul>

    <p class="about__note">当前为前端首版骨架（占位页），后端接口接入与页面细化将逐步完善。</p>
  </section>
</template>

<style scoped>
.about {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.about__title {
  font-size: 1.75rem;
  margin: 0;
}

.about__lead,
.about__note {
  color: var(--color-muted-text, #6b7280);
}

.about__layers {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.75rem;
}

.about__layer {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  padding: 0.9rem 1.1rem;
  background: var(--color-surface, #fff);
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: var(--radius, 0.75rem);
  box-shadow: var(--shadow-sm);
}

.about__layer strong {
  font-size: 1.05rem;
}
</style>
