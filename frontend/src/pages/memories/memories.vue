<script setup lang="ts">
import { onMounted, ref } from 'vue'
import Icon from '../../components/Icon.vue'
import MemoryPanel from '../../components/MemoryPanel.vue'
import { getMe, type UserInfo } from '../../api/client'

const user = ref<UserInfo | null>(null)
const panelRef = ref<InstanceType<typeof MemoryPanel> | null>(null)

const refresh = () => panelRef.value?.refresh()

onMounted(async () => {
  try {
    user.value = (await getMe()).data
  } catch {
    // 401 由 client 统一跳登录。
  }
})
</script>

<template>
  <div class="h-full overflow-y-auto scroll-thin">
    <div class="mx-auto w-full max-w-4xl px-5 py-6 space-y-4">
      <!-- 三层记忆说明横幅 -->
      <section class="surface p-4">
        <div class="flex items-center justify-between gap-3">
          <p class="text-xs font-semibold text-ink flex items-center gap-1.5">
            <Icon name="layers" :size="14" class="text-brand-500" /> 三层记忆 · 随对话自动沉淀
          </p>
          <button class="btn-ghost h-8 px-2.5 !rounded-lg flex items-center gap-1 text-xs text-ink-mute" aria-label="刷新记忆" @click="refresh">
            <Icon name="refresh" :size="13" /> 刷新
          </button>
        </div>
        <div class="mt-2.5 grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px] leading-4">
          <div class="rounded-lg bg-brand-50/70 px-3 py-2 ring-1 ring-brand-100">
            <p class="font-semibold text-brand-700">知识记忆</p>
            <p class="mt-0.5 text-ink-mute">角色关系、情节设定等长期事实，跨会话复用</p>
          </div>
          <div class="rounded-lg bg-emerald-50/70 px-3 py-2 ring-1 ring-emerald-100">
            <p class="font-semibold text-emerald-700">对话记忆</p>
            <p class="mt-0.5 text-ink-mute">问答沉淀的要点与你的偏好、修正记录</p>
          </div>
          <div class="rounded-lg bg-paper px-3 py-2 ring-1 ring-black/[0.06]">
            <p class="font-semibold text-ink-soft">会话记忆</p>
            <p class="mt-0.5 text-ink-mute">当前会话使用中的短期上下文</p>
          </div>
        </div>
        <p class="mt-2 text-[10px] text-ink-faint">
          {{ user ? `当前用户：${user.display_name || user.username} · ` : '' }}每类独立清理 · 不影响检索与问答
        </p>
      </section>

      <!-- 记忆面板（三类分组 + 删除） -->
      <MemoryPanel ref="panelRef" class="!rounded-xl" />
    </div>
  </div>
</template>
