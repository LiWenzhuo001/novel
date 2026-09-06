<script setup lang="ts">
import { computed, ref, onMounted, onUnmounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import Icon from './Icon.vue'
import { getMe, clearApiToken, type UserInfo } from '../api/client'
import { showToast } from '../utils/toast'

const route = useRoute()
const router = useRouter()

const collapsed = ref(localStorage.getItem('novel_layout_collapsed') === '1')

const NAV_ITEMS = [
  { to: '/chat', label: '问答工作台', icon: 'messages' },
  { to: '/library', label: '知识库书架', icon: 'book' },
  { to: '/world', label: '进入小说世界', icon: 'book-open' },
  { to: '/memories', label: '对话记忆', icon: 'layers' },
]

const pageTitle = computed(() => (route.meta.title as string) || '')
const currentUser = ref<UserInfo | null>(null)
const accountName = computed(() => currentUser.value?.display_name || currentUser.value?.username || '')
const accountInitial = computed(() => (accountName.value || '?').charAt(0).toUpperCase())
const menuOpen = ref(false)
const menuRef = ref<HTMLDivElement>()

const isActive = (to: string) => route.path === to || route.path.startsWith(to + '/')

const toggleCollapsed = () => {
  collapsed.value = !collapsed.value
  localStorage.setItem('novel_layout_collapsed', collapsed.value ? '1' : '0')
}

const logout = () => {
  clearApiToken()
  localStorage.removeItem('novel_selected_file_id')
  currentUser.value = null
  menuOpen.value = false
  showToast('已退出登录')
  router.replace('/login')
}

const goMemories = () => {
  menuOpen.value = false
  router.push('/memories')
}

const loadCurrentUser = () => {
  getMe()
    .then((res) => (currentUser.value = res.data))
    .catch(() => (currentUser.value = null))
}

const onKeydown = (event: KeyboardEvent) => {
  if (event.key === 'Escape' && menuOpen.value) menuOpen.value = false
}

const onDocClick = (event: MouseEvent) => {
  if (menuOpen.value && !menuRef.value?.contains(event.target as Node)) menuOpen.value = false
}

onMounted(() => {
  window.addEventListener('keydown', onKeydown)
  document.addEventListener('click', onDocClick)
  loadCurrentUser()
})
onUnmounted(() => {
  window.removeEventListener('keydown', onKeydown)
  document.removeEventListener('click', onDocClick)
})

watch(() => route.path, () => {
  menuOpen.value = false
})
</script>

<template>
  <div class="h-full flex flex-col">
    <!-- ===== 顶栏 64px ===== -->
    <header class="fixed top-0 inset-x-0 h-16 z-30 bg-white/95 backdrop-blur border-b border-black/[0.08] flex items-center gap-3 px-4 sm:px-6">
      <div class="flex items-center gap-2.5 min-w-0">
        <div class="shrink-0 w-9 h-9 rounded-lg bg-brand-600 flex items-center justify-center text-white shadow-card">
          <Icon name="book-open" :size="17" />
        </div>
        <p class="font-display text-[15px] font-bold tracking-wide text-ink leading-tight truncate whitespace-nowrap">小说智读</p>
      </div>

      <!-- 全局搜索（占位） -->
      <div class="hidden md:flex flex-1 justify-center min-w-0">
        <div class="w-full max-w-md flex items-center gap-2 rounded-lg border border-black/[0.08] bg-paper px-3 py-2 text-xs text-ink-faint">
          <Icon name="search" :size="13" />
          搜索书籍 · 章节（即将上线）
        </div>
      </div>

      <div class="flex items-center gap-2 shrink-0">
        <p class="hidden sm:block text-xs text-ink-mute truncate">{{ pageTitle }}</p>
        <div ref="menuRef" class="relative">
          <button
            class="w-9 h-9 rounded-full bg-brand-600 text-white flex items-center justify-center text-[13px] font-bold shadow-card"
            aria-label="账号菜单"
            @click="menuOpen = !menuOpen"
          >
            {{ accountInitial }}
          </button>
          <div
            v-if="menuOpen"
            class="absolute right-0 top-full mt-2 w-52 rounded-xl bg-white ring-1 ring-black/[0.1] shadow-xl z-50 py-1.5"
          >
            <p class="px-3 py-1.5 text-[11px] text-ink-faint truncate">{{ accountName || '未登录' }}</p>
            <button
              class="w-full text-left px-3 py-2 text-[13px] text-ink hover:bg-paper flex items-center gap-2"
              @click="goMemories"
            >
              <Icon name="layers" :size="14" class="text-ink-faint" /> 对话记忆
            </button>
            <button
              class="w-full text-left px-3 py-2 text-[13px] text-rose-600 hover:bg-rose-50 flex items-center gap-2"
              @click="logout"
            >
              <Icon name="logout" :size="14" /> 退出登录
            </button>
          </div>
        </div>
      </div>
    </header>

    <div class="pt-16 flex-1 flex min-h-0">
      <!-- ===== 侧栏 220px（可收起 64px） ===== -->
      <aside
        class="hidden lg:flex shrink-0 flex-col border-r border-black/[0.06] bg-white transition-all duration-200"
        :class="collapsed ? 'w-16' : 'w-[220px]'"
      >
        <nav class="flex-1 py-3 space-y-0.5 px-2">
          <RouterLink
            v-for="item in NAV_ITEMS"
            :key="item.to"
            :to="item.to"
            class="flex items-center gap-2.5 rounded-lg px-2.5 py-2.5 text-[13px] transition-colors"
            :class="isActive(item.to)
              ? 'bg-brand-50 text-brand-700 font-semibold'
              : 'text-ink-mute hover:bg-paper hover:text-ink'"
            :title="item.label"
          >
            <Icon :name="item.icon" :size="16" class="shrink-0" />
            <span v-if="!collapsed" class="truncate">{{ item.label }}</span>
          </RouterLink>
        </nav>

        <!-- 底部用户卡 -->
        <div class="border-t border-black/[0.06] p-2">
          <div ref="menuRef" class="relative">
            <button
              class="w-full flex items-center gap-2.5 rounded-lg px-2 py-2 hover:bg-paper transition-colors"
              :class="collapsed && 'justify-center'"
              aria-label="账号"
              @click="menuOpen = !menuOpen"
            >
              <span class="shrink-0 w-8 h-8 rounded-full bg-brand-600 text-white flex items-center justify-center text-[12px] font-bold">
                {{ accountInitial }}
              </span>
              <span v-if="!collapsed" class="min-w-0 flex-1 text-left">
                <span class="block truncate text-[12px] font-medium text-ink">{{ accountName || '未登录' }}</span>
              </span>
            </button>
            <div
              v-if="menuOpen"
              class="absolute left-0 bottom-full mb-2 w-48 rounded-xl bg-white ring-1 ring-black/[0.1] shadow-xl z-50 py-1.5"
            >
              <p class="px-3 py-1.5 text-[11px] text-ink-faint truncate">{{ accountName || '未登录' }}</p>
              <button
                class="w-full text-left px-3 py-2 text-[13px] text-ink hover:bg-paper flex items-center gap-2"
                @click="goMemories"
              >
                <Icon name="layers" :size="14" class="text-ink-faint" /> 对话记忆
              </button>
              <button
                class="w-full text-left px-3 py-2 text-[13px] text-rose-600 hover:bg-rose-50 flex items-center gap-2"
                @click="logout"
              >
                <Icon name="logout" :size="14" /> 退出登录
              </button>
            </div>
          </div>
        </div>

        <button
          class="pb-3 pt-1 flex justify-center text-ink-faint hover:text-ink transition-colors"
          :aria-label="collapsed ? '展开导航' : '收起导航'"
          @click="toggleCollapsed"
        >
          <Icon :name="collapsed ? 'chevron-right' : 'arrow-left'" :size="14" :class="!collapsed && 'rotate-180'" />
        </button>
      </aside>

      <!-- ===== 内容区 ===== -->
      <main class="flex-1 min-w-0 min-h-0 overflow-hidden">
        <router-view />
      </main>
    </div>
  </div>
</template>
