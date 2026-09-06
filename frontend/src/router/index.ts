import { createRouter, createWebHistory } from 'vue-router'
import { getApiToken } from '../api/client'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      component: () => import('../pages/home/home.vue'),
      meta: { title: '首页工作台', requiresAuth: true },
    },
    {
      path: '/login',
      name: 'login',
      component: () => import('../pages/login/login.vue'),
      meta: { title: '登录', public: true },
    },
    {
      path: '/chat',
      name: 'chat',
      component: () => import('../pages/chat/chat.vue'),
      meta: { title: '问答工作台', requiresAuth: true },
    },
    {
      path: '/library',
      name: 'library',
      component: () => import('../pages/library/library.vue'),
      meta: { title: '知识库书架', requiresAuth: true },
    },
    {
      path: '/memories',
      name: 'memories',
      component: () => import('../pages/memories/memories.vue'),
      meta: { title: '对话记忆', requiresAuth: true },
    },
    {
      path: '/world',
      name: 'world',
      component: () => import('../pages/world/world.vue'),
      meta: { title: '进入小说世界', requiresAuth: true },
    },
    { path: '/:pathMatch(.*)*', redirect: '/' },
  ],
})

router.beforeEach((to) => {
  if (to.meta.requiresAuth && !getApiToken()) {
    return { path: '/login', query: { redirect: to.fullPath } }
  }
  if (to.path === '/login' && getApiToken()) {
    const redirect = typeof to.query.redirect === 'string' ? to.query.redirect : '/'
    return redirect.startsWith('/') ? redirect : '/'
  }
  return true
})

router.afterEach((to) => {
  document.title = to.meta.title
    ? `${to.meta.title} · 小说智读`
    : '小说智读 · 原文可追溯问答'
})

export default router
