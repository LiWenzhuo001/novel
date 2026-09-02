import { createRouter, createWebHistory } from 'vue-router'
import { getApiToken } from '../api/client'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/chat' },
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
      meta: { title: '小说 RAG 问答', requiresAuth: true },
    },
    {
      path: '/memories',
      name: 'memories',
      component: () => import('../pages/memories/memories.vue'),
      meta: { title: '对话记忆', requiresAuth: true },
    },
    { path: '/:pathMatch(.*)*', redirect: '/chat' },
  ],
})

router.beforeEach((to) => {
  if (to.meta.requiresAuth && !getApiToken()) {
    return { path: '/login', query: { redirect: to.fullPath } }
  }
  if (to.path === '/login' && getApiToken()) {
    const redirect = typeof to.query.redirect === 'string' ? to.query.redirect : '/chat'
    return redirect.startsWith('/') ? redirect : '/chat'
  }
  return true
})

router.afterEach((to) => {
  document.title = to.meta.title
    ? `${to.meta.title} · 小说智读`
    : '小说智读 · 原文可追溯问答'
})

export default router
