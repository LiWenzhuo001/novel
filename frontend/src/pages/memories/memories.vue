<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import Icon from '../../components/Icon.vue'
import MemoryPanel from '../../components/MemoryPanel.vue'
import { getMe, type UserInfo } from '../../api/client'

const router = useRouter()
const user = ref<UserInfo | null>(null)
const panelRef = ref<InstanceType<typeof MemoryPanel> | null>(null)

const accountName = computed(() => user.value?.display_name || user.value?.username || '')

const back = () => {
  if (window.history.length > 1) router.back()
  else router.push('/chat')
}

const refresh = () => panelRef.value?.refresh()

onMounted(async () => {
  try {
    user.value = (await getMe()).data
  } catch {
    // 401 由 client 统一跳登录；其余场景顶栏不展示用户名即可。
  }
})
</script>

<template>
  <div class="min-h-screen bg-paper flex flex-col">
    <!-- 轻量页头：返回 + 标题 + 当前用户 -->
    <header class="sticky top-0 z-40 h-14 flex items-center gap-3 px-5 border-b border-black/[0.06] bg-white/85 backdrop-blur">
      <button
        class="btn-ghost h-9 px-3 !rounded-lg flex items-center gap-1.5 text-xs text-ink-mute"
        aria-label="返回聊天"
        @click="back"
      >
        <Icon name="arrow-left" :size="14" /> 返回聊天
      </button>
      <div class="flex items-center gap-2">
        <Icon name="sparkles" :size="15" class="text-brand-600" />
        <h1 class="text-sm font-semibold text-ink">对话记忆</h1>
      </div>
      <div class="ml-auto flex items-center gap-2.5">
        <button class="btn-ghost h-9 w-9 !rounded-lg !p-0" aria-label="刷新记忆" @click="refresh">
          <Icon name="refresh" :size="15" />
        </button>
        <span
          v-if="accountName"
          class="w-7 h-7 rounded-full bg-brand-600 text-white flex items-center justify-center text-[11px] font-semibold shadow-sm"
          :title="accountName"
        >{{ accountName.slice(0, 1).toUpperCase() }}</span>
      </div>
    </header>

    <!-- 记忆列表主体 -->
    <main class="flex-1 w-full max-w-3xl mx-auto px-5 py-6">
      <MemoryPanel ref="panelRef" class="!rounded-xl" />
    </main>
  </div>
</template>
