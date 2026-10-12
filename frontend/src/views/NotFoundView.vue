<script setup lang="ts">
// 404 页面：路由 catch-all 命中（规范：页面级组件按路由放置）
</script>

<template>
  <section class="notfound">
    <p class="notfound__code">404</p>
    <h1 class="notfound__title">页面不存在</h1>
    <p class="notfound__desc">你访问的地址没有对应的页面，可能已被移除或输入有误。</p>
    <RouterLink to="/" class="notfound__link"> 返回首页 </RouterLink>
  </section>
</template>

<style scoped>
.notfound {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  min-height: 50vh;
  text-align: center;
}

.notfound__code {
  margin: 0;
  font-size: clamp(3rem, 10vw, 5rem);
  font-weight: 700;
  color: var(--color-primary, #2563eb);
  line-height: 1;
}

.notfound__title {
  margin: 0;
  font-size: 1.5rem;
}

.notfound__desc {
  margin: 0;
  color: var(--color-muted-text, #6b7280);
}

.notfound__link {
  margin-top: 0.5rem;
  padding: 0.5rem 1.1rem;
  border-radius: var(--radius, 0.75rem);
  background: var(--color-primary, #2563eb);
  color: #fff;
  text-decoration: none;
  transition: background 0.15s ease;
}

.notfound__link:hover {
  background: var(--color-primary-hover, #1d4ed8);
}
</style>
