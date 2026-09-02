<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import Icon from '../../components/Icon.vue'
import KnowledgeManager from '../../components/KnowledgeManager.vue'
import ChatPanel from '../../components/ChatPanel.vue'
import { getMe, clearApiToken, type UserInfo, type KBFileInfo } from '../../api/client'
import { showToast } from '../../utils/toast'

const router = useRouter()
const role = 'student'
const agentStrategy = ref<'auto' | 'direct' | 'multi_expert'>('auto')
const showKB = ref(false)
const chatRef = ref<InstanceType<typeof ChatPanel>>()
const kbToggleRef = ref<HTMLButtonElement>()
const kbCloseRef = ref<HTMLButtonElement>()
const libraryFiles = ref<KBFileInfo[]>([])
const selectedFileId = ref<string | null>(localStorage.getItem('novel_selected_file_id'))

// 账号展示 + 退出菜单
const menuRef = ref<HTMLDivElement>()
const menuOpen = ref(false)
const currentUser = ref<UserInfo | null>(null)
const accountName = computed(() => currentUser.value?.display_name || currentUser.value?.username || '')
const accountInitial = computed(() => (accountName.value || '?').charAt(0).toUpperCase())

// 通用提问：不绑定具体书籍，四向覆盖人物关系、情节因果、时间线、章节定位
const suggestions = [
  '梳理主要人物之间的关系及其变化',
  '这段情节的直接原因和后续影响是什么？',
  '按时间顺序整理这一事件的发展',
  '这个关键转折最早出现在哪一章？',
]
const indexedFiles = computed(() => libraryFiles.value.filter((file) => file.status === 'indexed'))
const selectedFile = computed(() => libraryFiles.value.find((file) => file.id === selectedFileId.value) || null)
const hasMultipleNovels = computed(() => libraryFiles.value.length > 1)
const canChat = computed(() => selectedFile.value?.status === 'indexed')
const selectionRequired = computed(() => hasMultipleNovels.value && !selectedFileId.value)
const sessionKey = computed(() => `novel_rag_session_id_${selectedFileId.value || 'unselected'}`)

const ask = (q: string) => {
  if (!canChat.value) {
    showToast(hasMultipleNovels.value ? '请先选择要咨询的小说' : '请先上传并等待小说索引完成')
    return
  }
  chatRef.value?.send(q)
}

const onFilesChanged = (files: KBFileInfo[]) => {
  libraryFiles.value = files
  const current = selectedFileId.value
  const currentFile = files.find((file) => file.id === current)

  if (currentFile && currentFile.status === 'indexed') return

  // 默认选中系统内置小说（红楼梦）：用户没有有效选择时始终兜底到系统书，
  // 即使用户后来上传了自己的书，首次进入也无需手动选择。
  const systemBook = files.find((file) => file.is_system && file.status === 'indexed')
  if (!current || !currentFile) {
    if (systemBook) {
      selectedFileId.value = systemBook.id
      localStorage.setItem('novel_selected_file_id', systemBook.id)
      return
    }
  }

  if (files.length <= 1 && indexedFiles.value.length === 1) {
    selectedFileId.value = indexedFiles.value[0].id
    localStorage.setItem('novel_selected_file_id', selectedFileId.value)
    return
  }

  if (current && !currentFile) {
    selectedFileId.value = null
    localStorage.removeItem('novel_selected_file_id')
    // 残留选择被清理后同样兜底到系统书，避免落入"多本书必须手选"的阻断。
    if (systemBook) {
      selectedFileId.value = systemBook.id
      localStorage.setItem('novel_selected_file_id', systemBook.id)
    }
  }
}


const onNovelSelectChange = (event: Event) => {
  selectNovel((event.target as HTMLSelectElement).value)
}

const selectNovel = (fileId: string) => {
  const file = libraryFiles.value.find((item) => item.id === fileId)
  if (!file || file.status !== 'indexed') {
    showToast('只能选择已完成索引的小说')
    return
  }
  if (selectedFileId.value === fileId) return
  selectedFileId.value = fileId
  localStorage.setItem('novel_selected_file_id', fileId)
  showToast(`当前咨询对象已切换为《${file.filename}》`)
}
const closeKB = () => (showKB.value = false)
const onKeydown = (event: KeyboardEvent) => {
  if (event.key === 'Escape' && showKB.value) closeKB()
  if (event.key === 'Escape' && menuOpen.value) menuOpen.value = false
}

// 点击菜单外部时关闭
const onDocClick = (event: MouseEvent) => {
  if (menuOpen.value && !menuRef.value?.contains(event.target as Node)) menuOpen.value = false
}

// 对话记忆中心：跨会话查看/删除用户名下全部记忆
const goMemories = () => {
  menuOpen.value = false
  router.push('/memories')
}

// JWT 无状态鉴权：登出即清除本地 token 并回到登录页
const logout = () => {
  clearApiToken()
  localStorage.removeItem('novel_selected_file_id')
  currentUser.value = null
  menuOpen.value = false
  showToast('已退出登录')
  router.replace('/login')
}

const loadCurrentUser = () => {
  getMe()
    .then((res) => (currentUser.value = res.data))
    .catch(() => (currentUser.value = null))
}

watch(showKB, async (open) => {
  document.body.classList.toggle('drawer-open', open)
  await nextTick()
  if (open) kbCloseRef.value?.focus()
  else kbToggleRef.value?.focus()
})

onMounted(() => {
  window.addEventListener('keydown', onKeydown)
  document.addEventListener('click', onDocClick)
  loadCurrentUser()
})
onUnmounted(() => {
  document.body.classList.remove('drawer-open')
  window.removeEventListener('keydown', onKeydown)
  document.removeEventListener('click', onDocClick)
})
</script>

<template>
  <div class="h-full flex flex-col">
    <!-- ===== 顶栏 ===== -->
    <header class="fixed top-0 inset-x-0 h-16 z-30 bg-white/95 backdrop-blur border-b border-black/[0.08] flex items-center justify-between gap-3 px-4 sm:px-6">
      <div class="flex items-center gap-2.5 min-w-0 sm:gap-3">
        <div class="shrink-0 w-9 h-9 rounded-lg bg-brand-600 flex items-center justify-center text-white shadow-card">
          <Icon name="sparkles" :size="17" />
        </div>
        <div class="min-w-0">
          <p class="font-display text-[15px] font-bold tracking-wide text-ink leading-tight truncate">小说智读</p>
          <p class="text-[10px] uppercase tracking-[0.18em] text-ink-faint leading-tight">Novel RAG Workspace</p>
        </div>
      </div>

      <div class="flex items-center gap-2">
        <div class="flex items-center rounded-lg border border-black/[0.08] bg-paper p-0.5" role="group" aria-label="Agent 策略">
          <button
            v-for="item in ([
              { value: 'direct', short: '直', label: '直接' },
              { value: 'auto', short: '自动', label: '自动' },
              { value: 'multi_expert', short: '专', label: '多专家' },
            ] as const)"
            :key="item.value"
            class="min-w-8 rounded-md px-1.5 py-1.5 text-[11px] transition-colors sm:min-w-0 sm:px-2.5"
            :class="agentStrategy === item.value ? 'bg-ink text-white' : 'text-ink-mute hover:bg-white'"
            :aria-pressed="agentStrategy === item.value"
            @click="agentStrategy = item.value"
          >
            <span class="sm:hidden">{{ item.short }}</span>
            <span class="hidden sm:inline">{{ item.label }}</span>
          </button>
        </div>
        <button
          ref="kbToggleRef"
          @click="showKB = !showKB"
          class="lg:hidden btn-ghost w-9 h-9 !p-0 !rounded-lg"
          aria-label="打开小说书库"
          aria-controls="knowledge-drawer"
          :aria-expanded="showKB"
        >
          <Icon name="file-text" :size="16" />
        </button>

        <!-- 账号菜单 -->
        <div v-if="currentUser" ref="menuRef" class="relative shrink-0">
          <button
            @click="menuOpen = !menuOpen"
            class="flex items-center gap-1.5 rounded-full py-1 pl-1 pr-2 text-sm transition-colors hover:bg-black/[0.04] cursor-pointer"
            aria-haspopup="menu"
            :aria-expanded="menuOpen"
          >
            <span class="w-7 h-7 rounded-full bg-brand-600 text-white flex items-center justify-center text-[11px] font-semibold shadow-sm">
              {{ accountInitial }}
            </span>
            <span class="hidden md:block max-w-24 truncate text-xs text-ink-mute">{{ accountName }}</span>
            <Icon
              name="chevron-down"
              :size="13"
              :class="['text-ink-faint transition-transform duration-200', menuOpen && 'rotate-180']"
            />
          </button>
          <div
            v-if="menuOpen"
            class="absolute right-0 top-full mt-2 w-56 rounded-xl bg-white ring-1 ring-black/[0.1] shadow-xl z-50 overflow-hidden"
            role="menu"
            aria-label="用户菜单"
          >
            <div class="px-3.5 py-2.5 border-b border-black/[0.06]">
              <p class="text-sm font-semibold text-ink leading-tight truncate">{{ accountName }}</p>
              <p class="mt-0.5 text-xs text-ink-faint truncate">{{ currentUser.username }}</p>
            </div>
            <button
              @click="goMemories"
              class="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-left text-sm text-ink-soft transition-colors hover:bg-brand-50 cursor-pointer"
              role="menuitem"
            >
              <span class="shrink-0 w-7 h-7 rounded-lg bg-brand-50 flex items-center justify-center">
                <Icon name="sparkles" :size="15" class="text-brand-600" />
              </span>
              对话记忆
            </button>
            <button
              @click="logout"
              class="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-left text-sm text-rose-600 transition-colors hover:bg-rose-50 cursor-pointer border-t border-black/[0.06]"
              role="menuitem"
            >
              <span class="shrink-0 w-7 h-7 rounded-lg bg-rose-50 flex items-center justify-center">
                <Icon name="logout" :size="15" />
              </span>
              退出登录
            </button>
          </div>
        </div>
      </div>
    </header>

    <!-- ===== 主体三栏 ===== -->
    <div class="pt-16 flex-1 flex min-h-0">
      <!-- 左栏 -->
      <aside class="hidden lg:flex w-60 shrink-0 flex-col gap-4 p-5 border-r border-black/[0.06]">
        <div>
          <p class="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-faint mb-2.5 flex items-center gap-1.5">
            <Icon name="lightbulb" :size="13" class="text-brand-500" /> 使用提示
          </p>
          <div class="surface p-4">
            <p class="text-xs leading-5 text-ink-mute">
              基于已索引小说原文回答人物、情节、时间线和章节定位问题。复杂问题在自动模式下由多个分析智能体共享证据并汇总。
            </p>
          </div>
        </div>

        <div class="flex-1 min-h-0 flex flex-col">
          <p class="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-faint mb-2.5 flex items-center gap-1.5">
            <Icon name="messages" :size="13" class="text-brand-500" /> 快速提问
          </p>
          <div class="space-y-1.5 overflow-y-auto scroll-thin">
            <button
              v-for="(q, i) in suggestions"
              :key="i"
              @click="ask(q)"
              class="group w-full flex items-start gap-2.5 rounded-xl px-3 py-2.5 text-left text-[13px] leading-5 text-ink-mute transition-all duration-200 hover:bg-white hover:text-brand-700 hover:shadow-card hover:ring-1 hover:ring-black/[0.06] cursor-pointer"
            >
              <span class="shrink-0 mt-0.5 font-display text-xs font-bold text-brand-400 group-hover:text-brand-600">{{ String(i + 1).padStart(2, '0') }}</span>
              {{ q }}
            </button>
          </div>
        </div>
      </aside>

      <!-- 中间对话 -->
      <main class="flex-1 min-w-0 flex flex-col">
        <section class="mx-auto w-full max-w-3xl px-5 pt-5 sm:px-6" aria-label="当前咨询小说">
          <div
            v-if="hasMultipleNovels || selectedFile"
            class="rounded-xl border px-4 py-3 shadow-sm"
            :class="selectionRequired || !canChat ? 'border-amber-200 bg-amber-50' : 'border-brand-100 bg-white'"
          >
            <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div class="min-w-0">
                <p class="text-xs font-semibold text-ink">当前咨询小说</p>
                <p v-if="selectionRequired" class="mt-1 text-xs leading-5 text-amber-800">
                  已上传多本小说，请先选择一部作为当前咨询对象，选择前不能开始问答。
                </p>
                <p v-else-if="selectedFile && selectedFile.status !== 'indexed'" class="mt-1 text-xs leading-5 text-amber-800">
                  《{{ selectedFile.filename }}》正在索引或索引失败，请选择其他已完成索引的小说。
                </p>
                <p v-else-if="selectedFile" class="mt-1 truncate text-xs text-ink-mute" :title="selectedFile.filename">
                  后续问答仅基于《{{ selectedFile.filename }}》的原文。
                </p>
              </div>
              <select
                :value="selectedFileId || ''"
                class="min-w-0 rounded-lg border border-black/[0.1] bg-white px-3 py-2 text-xs text-ink outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100 sm:w-64"
                aria-label="选择当前咨询小说"
                @change="onNovelSelectChange"
              >
                <option value="" disabled>请选择已索引小说</option>
                <option v-for="file in indexedFiles" :key="file.id" :value="file.id">
                  {{ file.filename }}
                </option>
              </select>
            </div>
          </div>
        </section>

        <ChatPanel
          v-if="canChat"
          :key="selectedFileId || 'unselected'"
          ref="chatRef"
          :role="role"
          domain="novel"
          :strategy="agentStrategy"
          :file-id="selectedFileId"
          :session-key="sessionKey"
          :suggestions="suggestions"
          title="阅读你的"
          title-suffix="小说"
          subtitle="上传小说文本后，可追问人物关系、情节因果、时间线和章节位置。复杂问题可由人物、情节、时间线和章节定位专家并发分析。"
          notice="严格基于已索引原文回答 · 关键结论附章节引用"
        />
        <div v-else class="flex flex-1 items-center justify-center px-6 py-16 text-center">
          <div class="max-w-md rounded-2xl border border-dashed border-amber-300 bg-amber-50/70 px-6 py-8">
            <Icon name="folder-open" :size="28" class="mx-auto text-amber-600" />
            <p class="mt-3 text-sm font-semibold text-ink">
              {{ hasMultipleNovels ? '请先选择一部小说' : '请先上传并索引小说' }}
            </p>
            <p class="mt-2 text-xs leading-5 text-ink-mute">
              {{ hasMultipleNovels ? '选择目标小说后，问答上下文会严格切换到该作品。' : '小说完成索引后才能开始问答。' }}
            </p>
          </div>
        </div>
      </main>

      <!-- 右侧知识库 -->
      <button
        v-if="showKB"
        class="fixed inset-0 z-40 bg-black/30 lg:hidden"
        aria-label="关闭小说书库"
        @click="closeKB"
      ></button>
      <aside
        id="knowledge-drawer"
        :class="[
          'shrink-0 w-80 max-w-[88vw] p-5 border-l border-black/[0.08] bg-paper flex-col',
          showKB ? 'fixed inset-y-0 right-0 z-50 flex shadow-2xl lg:static lg:z-auto lg:shadow-none' : 'hidden lg:flex',
        ]"
        :aria-hidden="!showKB && undefined"
        aria-label="小说书库"
      >
        <div class="mb-4 flex items-center justify-between lg:hidden">
          <p class="text-sm font-semibold text-ink">小说书库</p>
          <button
            ref="kbCloseRef"
            class="btn-ghost h-9 w-9 !rounded-lg !p-0"
            aria-label="关闭小说书库"
            @click="closeKB"
          >
            <Icon name="x" :size="17" />
          </button>
        </div>
        <KnowledgeManager domain="novel" @files-changed="onFilesChanged" />
      </aside>
    </div>
  </div>
</template>
