<script setup lang="ts">
// 公共复用组件示例：卡片容器（后续其它页面复用）
withDefaults(
  defineProps<{
    title?: string
  }>(),
  { title: '' },
)
</script>

<template>
  <article class="app-card">
    <h3 v-if="title" class="app-card__title">{{ title }}</h3>
    <slot />
  </article>
</template>

<style scoped>
.app-card {
  background: var(--color-surface, #fff);
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: var(--radius, 0.75rem);
  padding: 1.25rem;
  box-shadow: var(--shadow-sm);
}

.app-card__title {
  margin: 0 0 0.75rem;
  font-size: 1.05rem;
}
</style>
