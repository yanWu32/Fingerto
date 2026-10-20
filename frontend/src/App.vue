<script setup lang="ts">
// 根组件：仅承载布局骨架（header / nav / main / footer），具体页面由路由渲染
</script>

<template>
  <div class="app-shell">
    <header class="app-header">
      <div class="app-header__brand">
        <span class="app-header__logo" aria-hidden="true">🤟</span>
        <span class="app-header__title">手语对话系统</span>
      </div>
      <nav class="app-nav" aria-label="主导航">
        <RouterLink to="/" class="app-nav__link"> 首页 </RouterLink>
        <RouterLink to="/about" class="app-nav__link"> 关于 </RouterLink>
      </nav>
    </header>

    <main id="main" class="app-main">
      <RouterView />
    </main>

    <footer class="app-footer">
      <p>基于LLM智能体的手语对话系统 · 前端演示（前后端分离）</p>
    </footer>
  </div>
</template>

<style scoped>
.app-shell {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.75rem 1.5rem;
  background: var(--color-surface, #ffffff);
  border-bottom: 1px solid var(--color-border, #e5e7eb);
}

.app-header__brand {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  font-size: 1.125rem;
}

.app-header__logo {
  font-size: 1.5rem;
}

.app-nav {
  display: flex;
  gap: 0.5rem;
}

.app-nav__link {
  padding: 0.4rem 0.9rem;
  border-radius: 0.5rem;
  color: var(--color-text, #1f2937);
  text-decoration: none;
  transition: background 0.15s ease;
}

.app-nav__link:hover {
  background: var(--color-muted, #f3f4f6);
}

.app-nav__link.router-link-active {
  background: var(--color-primary, #2563eb);
  color: #fff;
}

.app-main {
  flex: 1;
  width: 100%;
  max-width: 1120px;
  margin: 0 auto;
  padding: 1.5rem;
}

.app-footer {
  padding: 1rem 1.5rem;
  text-align: center;
  color: var(--color-muted-text, #6b7280);
  font-size: 0.875rem;
  border-top: 1px solid var(--color-border, #e5e7eb);
}

/* 适配页面放大 200% 不溢出 */
@media (max-width: 640px) {
  .app-header {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
