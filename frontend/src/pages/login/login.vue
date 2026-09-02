<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import Icon from '../../components/Icon.vue'
import { getMe, login, register, setApiToken, clearApiToken, type UserInfo } from '../../api/client'
import { showToast } from '../../utils/toast'

const route = useRoute()
const router = useRouter()

const mode = ref<'login' | 'register'>('login')
const loading = ref(false)
const username = ref('')
const password = ref('')
const email = ref('')
const displayName = ref('')
const message = ref('')
const currentUser = ref<UserInfo | null>(null)

const title = computed(() => (mode.value === 'login' ? '登录小说智读' : '创建个人账号'))
const subtitle = computed(() =>
  mode.value === 'login'
    ? '登录后，聊天历史与知识库会按用户隔离。'
    : '注册会立即创建独立租户空间，并自动登录。',
)

const redirectTo = () => {
  const target = typeof route.query.redirect === 'string' ? route.query.redirect : '/chat'
  router.replace(target.startsWith('/') ? target : '/chat')
}

const loadCurrentUser = async () => {
  try {
    const res = await getMe()
    currentUser.value = res.data
  } catch {
    currentUser.value = null
  }
}

loadCurrentUser()

const submit = async () => {
  const name = username.value.trim()
  if (!name || !password.value) {
    message.value = '请输入用户名和密码'
    return
  }
  loading.value = true
  message.value = ''
  try {
    const res = mode.value === 'login'
      ? await login({ username: name, password: password.value })
      : await register({
          username: name,
          password: password.value,
          email: email.value.trim() || undefined,
          display_name: displayName.value.trim() || undefined,
        })
    setApiToken(res.data.access_token)
    currentUser.value = res.data.user
    showToast(mode.value === 'login' ? '登录成功' : '注册成功')
    redirectTo()
  } catch (e: any) {
    message.value = e?.message || '认证失败，请稍后重试'
  } finally {
    loading.value = false
  }
}

const switchMode = () => {
  mode.value = mode.value === 'login' ? 'register' : 'login'
  message.value = ''
}

// JWT 无状态鉴权：登出即清除本地 token（服务端无 session 需销毁）。
const logout = () => {
  clearApiToken()
  currentUser.value = null
  message.value = ''
  showToast('已退出登录')
}
</script>

<template>
  <div class="min-h-full flex items-center justify-center px-4 py-8 sm:py-10">
    <section class="w-full max-w-5xl grid lg:grid-cols-[1.05fr_0.95fr] gap-5 items-stretch">
      <div class="surface-pop p-8 sm:p-10 flex flex-col justify-between overflow-hidden">
        <div>
          <div class="inline-flex items-center gap-2 rounded-full bg-brand-50 text-brand-700 ring-1 ring-brand-100 px-3 py-1 text-xs font-semibold">
            <Icon name="sparkles" :size="14" /> Novel RAG Workspace
          </div>
          <h1 class="mt-8 font-display text-3xl sm:text-4xl font-bold text-ink">
            上传小说，追问原文
          </h1>
          <p class="mt-4 text-sm sm:text-base leading-7 text-ink-mute max-w-xl">
            围绕人物关系、情节因果、时间线和章节位置深入阅读。回答基于你的小说原文，并附可核对的章节引用。
          </p>
        </div>

        <div class="mt-10 grid sm:grid-cols-3 gap-3">
          <div class="rounded-lg bg-paper/70 ring-1 ring-black/[0.07] p-4">
            <p class="text-sm font-semibold text-ink">章节优先</p>
            <p class="mt-1 text-xs leading-5 text-ink-mute">自动识别章节并保留原文位置</p>
          </div>
          <div class="rounded-lg bg-paper/70 ring-1 ring-black/[0.07] p-4">
            <p class="text-sm font-semibold text-ink">证据可查</p>
            <p class="mt-1 text-xs leading-5 text-ink-mute">关键结论附章节与片段引用</p>
          </div>
          <div class="rounded-lg bg-paper/70 ring-1 ring-black/[0.07] p-4">
            <p class="text-sm font-semibold text-ink">数据隔离</p>
            <p class="mt-1 text-xs leading-5 text-ink-mute">书库和对话按账号独立保存</p>
          </div>
        </div>
      </div>

      <div class="surface-pop p-6 sm:p-8">
        <div class="flex items-start justify-between gap-4">
          <div>
            <h2 class="font-display text-2xl font-bold text-ink">{{ title }}</h2>
            <p class="mt-2 text-sm leading-6 text-ink-mute">{{ subtitle }}</p>
          </div>
          <div class="w-11 h-11 rounded-lg bg-brand-600 text-white flex items-center justify-center shadow-card">
            <Icon name="user" :size="19" />
          </div>
        </div>

        <div v-if="currentUser" class="mt-5 rounded-2xl bg-brand-50/70 ring-1 ring-brand-100 px-4 py-3 text-sm text-brand-700 flex items-center justify-between gap-3">
          <span>当前已登录：{{ currentUser.display_name || currentUser.username }}</span>
          <div class="flex items-center gap-2.5 shrink-0">
            <button class="font-semibold hover:text-brand-900" @click="redirectTo">进入应用</button>
            <button class="rounded-md px-1.5 py-0.5 -mr-1 font-medium text-brand-600 hover:text-brand-900 hover:underline" @click="logout">退出登录</button>
          </div>
        </div>

        <form class="mt-6 space-y-4" @submit.prevent="submit">
          <label class="block">
            <span class="text-xs font-semibold text-ink-mute">用户名</span>
            <input v-model="username" class="field mt-1.5" autocomplete="username" placeholder="例如：liwenzhuo" :aria-invalid="Boolean(message)" :aria-describedby="message ? 'auth-error' : undefined" />
          </label>

          <label class="block">
            <span class="text-xs font-semibold text-ink-mute">密码</span>
            <input v-model="password" class="field mt-1.5" type="password" :autocomplete="mode === 'login' ? 'current-password' : 'new-password'" placeholder="至少 6 位" :aria-invalid="Boolean(message)" :aria-describedby="message ? 'auth-error' : undefined" />
          </label>

          <template v-if="mode === 'register'">
            <label class="block">
              <span class="text-xs font-semibold text-ink-mute">邮箱（可选）</span>
              <input v-model="email" class="field mt-1.5" type="email" autocomplete="email" placeholder="用于后续账号管理" />
            </label>
            <label class="block">
              <span class="text-xs font-semibold text-ink-mute">显示名称（可选）</span>
              <input v-model="displayName" class="field mt-1.5" placeholder="页面展示名称" />
            </label>
          </template>

          <p v-if="message" id="auth-error" class="rounded-lg bg-red-50 text-red-700 ring-1 ring-red-200 px-3 py-2 text-sm leading-5" role="alert">
            {{ message }}
          </p>

          <button type="submit" class="btn-primary w-full py-3 text-sm" :disabled="loading" :aria-busy="loading">
            <Icon :name="loading ? 'loader' : 'check'" :size="16" />
            {{ loading ? '处理中...' : (mode === 'login' ? '登录' : '注册并登录') }}
          </button>
        </form>

        <div class="mt-6 flex items-center justify-between text-sm">
          <span class="text-ink-faint">{{ mode === 'login' ? '还没有账号？' : '已有账号？' }}</span>
          <button class="text-brand-600 font-semibold hover:text-brand-700" @click="switchMode">
            {{ mode === 'login' ? '立即注册' : '返回登录' }}
          </button>
        </div>
      </div>
    </section>
  </div>
</template>
