import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

// 路由懒加载（规范 4.5 / 性能准则）：首屏只加载当前页面代码
const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'home',
    component: () => import('@/views/HomeView.vue'),
    meta: { title: '首页' },
  },
  {
    path: '/demo',
    name: 'demo',
    component: () => import('@/views/DemoView.vue'),
    meta: { title: '全链路演示' },
  },
  {
    path: '/recognize',
    name: 'recognize',
    component: () => import('@/views/RecognizeView.vue'),
    meta: { title: '视频识别' },
  },
  {
    path: '/push',
    name: 'push',
    component: () => import('@/views/PushView.vue'),
    meta: { title: '信息推送' },
  },
  {
    path: '/log',
    name: 'audit-log',
    component: () => import('@/views/AuditLogView.vue'),
    meta: { title: '审计日志' },
  },
  {
    path: '/about',
    name: 'about',
    component: () => import('@/views/AboutView.vue'),
    meta: { title: '关于' },
  },
  {
    // 404：catch-all 必须放在最后，匹配所有未定义路由
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: () => import('@/views/NotFoundView.vue'),
    meta: { title: '页面不存在' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior() {
    return { top: 0 }
  },
})

// 路由守卫：仅做标题设置；真实权限校验由后端接口兜底（规范 4.5）
router.afterEach((to) => {
  document.title = to.meta.title ? `${to.meta.title} · 手语对话系统` : '手语对话系统'
})

export default router
