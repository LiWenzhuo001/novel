<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import Icon from '../../components/Icon.vue'
import ChatPanel from '../../components/ChatPanel.vue'
import { listKB, listSessions, deleteSession, renameSession, type KBFileInfo, type SessionItem } from '../../api/client'
import { showToast } from '../../utils/toast'

const router = useRouter()
const role = 'student'
const agentStrategy = ref<'auto' | 'direct' | 'multi_expert'>('auto')
const chatRef = ref<InstanceType<typeof ChatPanel>>()
const libraryFiles = ref<KBFileInfo[]>([])
const selectedFileId = ref<string | null>(localStorage.getItem('novel_selected_file_id'))

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
const sessionKey = computed(() => `novel_rag_session_id_${selectedFileId.value || 'unselected'}`)

// ===== 书库数据与选书兜底 =====
const loadNovels = async () => {
  try {
    const { data } = await listKB()
    const files = data || []
    libraryFiles.value = files
    const current = files.find((file) => file.id === selectedFileId.value)
    if (current && current.status === 'indexed') return
    const systemBook = files.find((file) => file.is_system && file.status === 'indexed')
    if (systemBook) {
      selectedFileId.value = systemBook.id
      localStorage.setItem('novel_selected_file_id', systemBook.id)
      return
    }
    if (indexedFiles.value.length === 1) {
      selectedFileId.value = indexedFiles.value[0].id
      localStorage.setItem('novel_selected_file_id', selectedFileId.value)
    }
  } catch {
    libraryFiles.value = []
  }
}

// ===== 同一小说内的多会话管理 =====
const sessions = ref<SessionItem[]>([])
const activeSessionId = ref<string | null>(null)
// 切换/新建/删除会话时自增，驱动 ChatPanel 重挂载；会话在流式中落地时不变，避免打断回答。
const sessionEpoch = ref(0)
const sessionsLoading = ref(false)

const syncActiveFromStorage = () => {
  activeSessionId.value = localStorage.getItem(sessionKey.value)
}

const refreshSessions = async () => {
  if (!selectedFileId.value) {
    sessions.value = []
    return
  }
  sessionsLoading.value = true
  try {
    const { data } = await listSessions(selectedFileId.value)
    // 角色扮演会话只归属「进入小说世界」，不混入主界面的对话历史。
    sessions.value = (data || []).filter((row) => !(row.personas || []).length)
  } catch {
    sessions.value = []
  } finally {
    sessionsLoading.value = false
  }
}

const switchSession = (id: string) => {
  if (id === activeSessionId.value) return
  localStorage.setItem(sessionKey.value, id)
  activeSessionId.value = id
  sessionEpoch.value += 1
}

const newConversation = () => {
  if (!canChat.value) {
    showToast(hasMultipleNovels.value ? '请先选择要咨询的小说' : '请先上传并等待小说索引完成')
    return
  }
  // 懒创建：只清当前指向，首条消息发出时由 ChatPanel 自动建会话，不产生空会话。
  localStorage.removeItem(sessionKey.value)
  activeSessionId.value = null
  sessionEpoch.value += 1
}

const removeSession = async (id: string) => {
  if (!window.confirm('删除后该对话的消息、会话记忆与摘要将一并清除，确定删除？')) return
  try {
    await deleteSession(id)
  } catch {
    showToast('删除失败，请重试')
    return
  }
  showToast('对话已删除')
  if (id === activeSessionId.value) {
    localStorage.removeItem(sessionKey.value)
    activeSessionId.value = null
    sessionEpoch.value += 1
  }
  void refreshSessions()
}

const onSessionCreated = (id: string) => {
  activeSessionId.value = id
  void refreshSessions()
}

// ===== 行内重命名：Enter/失焦提交，Esc 取消，空标题不提交 =====
const renamingId = ref<string | null>(null)
const renamingTitle = ref('')
const renameInputEl = ref<HTMLInputElement | null>(null)

const setRenameInput = (el: any) => {
  if (el) renameInputEl.value = el as HTMLInputElement
}

const startRename = (s: SessionItem) => {
  renamingId.value = s.id
  renamingTitle.value = s.title || ''
}

const cancelRename = () => {
  renamingId.value = null
  renamingTitle.value = ''
}

const commitRename = async () => {
  const id = renamingId.value
  if (!id) return
  const title = renamingTitle.value.trim()
  renamingId.value = null
  renamingTitle.value = ''
  if (!title) return
  const row = sessions.value.find((item) => item.id === id)
  if (!row || row.title === title) return
  try {
    await renameSession(id, title)
    row.title = title
  } catch {
    showToast('重命名失败，请重试')
  }
}

watch(renamingId, async (id) => {
  if (!id) return
  await nextTick()
  renameInputEl.value?.focus()
  renameInputEl.value?.select()
})

watch(selectedFileId, () => {
  syncActiveFromStorage()
  sessionEpoch.value += 1
  void refreshSessions()
}, { immediate: true })

const ask = (q: string) => {
  if (!canChat.value) {
    showToast(hasMultipleNovels.value ? '请先选择要咨询的小说' : '请先上传并等待小说索引完成')
    return
  }
  chatRef.value?.send(q)
}

const goLibrary = () => router.push('/library')

void loadNovels()
</script>

<template>
  <div class="h-full flex min-h-0">
    <!-- ===== 左栏：当前书 + 会话列表 + 快速提问 ===== -->
    <aside class="hidden lg:flex w-64 shrink-0 flex-col gap-4 p-4 border-r border-black/[0.06] min-h-0">
      <!-- 当前咨询书 -->
      <div class="surface p-3.5">
        <p class="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-faint flex items-center gap-1.5">
          <Icon name="book" :size="12" class="text-brand-500" /> 当前咨询书
        </p>
        <p v-if="selectedFile" class="mt-1.5 truncate text-[13px] font-semibold text-ink" :title="selectedFile.filename">
          {{ selectedFile.filename }}
        </p>
        <p v-else class="mt-1.5 text-[12px] text-ink-mute">未选择</p>
        <button
          class="mt-2 flex items-center gap-1 text-[11px] font-medium text-brand-600 hover:text-brand-700"
          @click="goLibrary"
        >
          <Icon name="book" :size="11" /> 去书架换书
        </button>
      </div>

      <!-- 会话历史 -->
      <div class="flex-1 min-h-0 flex flex-col">
        <button
          class="w-full flex items-center justify-center gap-2 rounded-xl border border-brand-200 bg-white px-3 py-2.5 text-[13px] font-medium text-brand-700 transition-all hover:bg-brand-50 hover:shadow-card disabled:cursor-not-allowed disabled:opacity-40"
          :disabled="!canChat"
          aria-label="新建对话"
          @click="newConversation"
        >
          <Icon name="plus" :size="13" /> 新建对话
        </button>
        <p class="mt-3.5 mb-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-faint flex items-center gap-1.5">
          <Icon name="messages" :size="13" class="text-brand-500" /> 对话历史
        </p>
        <div class="flex-1 min-h-0 space-y-1 overflow-y-auto scroll-thin pr-0.5">
          <div
            v-for="s in sessions"
            :key="s.id"
            class="group cursor-pointer rounded-xl px-3 py-2 transition-all duration-200"
            :class="s.id === activeSessionId
              ? 'bg-brand-50 ring-1 ring-brand-200'
              : 'hover:bg-white hover:shadow-card hover:ring-1 hover:ring-black/[0.06]'"
            role="button"
            tabindex="0"
            @click="switchSession(s.id)"
            @keydown.enter="switchSession(s.id)"
          >
            <div class="flex items-start justify-between gap-1">
              <input
                v-if="renamingId === s.id"
                :ref="setRenameInput"
                v-model="renamingTitle"
                class="min-w-0 flex-1 rounded-md border border-brand-300 bg-white px-2 py-1 text-[13px] leading-5 text-ink outline-none focus:ring-2 focus:ring-brand-100"
                aria-label="重命名对话"
                @click.stop
                @keyup.enter="commitRename"
                @keyup.esc="cancelRename"
                @blur="commitRename"
              />
              <p v-else class="min-w-0 flex-1 truncate text-[13px] leading-5" :class="s.id === activeSessionId ? 'font-medium text-brand-800' : 'text-ink'">
                {{ s.title || '新对话' }}
              </p>
              <template v-if="renamingId !== s.id">
                <button
                  class="hidden shrink-0 rounded p-0.5 text-ink-faint transition-colors hover:bg-brand-50 hover:text-brand-600 group-hover:block"
                  title="重命名对话"
                  aria-label="重命名对话"
                  @click.stop="startRename(s)"
                >
                  ✎
                </button>
                <button
                  class="hidden shrink-0 rounded p-0.5 text-ink-faint transition-colors hover:bg-rose-50 hover:text-rose-500 group-hover:block"
                  title="删除对话"
                  aria-label="删除对话"
                  @click.stop="removeSession(s.id)"
                >
                  ×
                </button>
              </template>
            </div>
            <p class="mt-0.5 text-[10px] text-ink-faint">{{ s.updated_at }}</p>
          </div>
          <p v-if="!sessions.length" class="px-3 py-6 text-center text-xs leading-5 text-ink-faint">
            {{ sessionsLoading ? '加载中…' : '还没有对话，发送第一条消息后自动创建' }}
          </p>
        </div>
      </div>

      <!-- 快速提问 -->
      <div class="shrink-0">
        <p class="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-faint mb-2.5 flex items-center gap-1.5">
          <Icon name="lightbulb" :size="13" class="text-brand-500" /> 快速提问
        </p>
        <div class="space-y-1.5 max-h-56 overflow-y-auto scroll-thin">
          <button
            v-for="(q, i) in suggestions"
            :key="i"
            class="group w-full flex items-start gap-2.5 rounded-xl px-3 py-2.5 text-left text-[13px] leading-5 text-ink-mute transition-all duration-200 hover:bg-white hover:text-brand-700 hover:shadow-card hover:ring-1 hover:ring-black/[0.06] cursor-pointer"
            @click="ask(q)"
          >
            <span class="shrink-0 mt-0.5 font-display text-xs font-bold text-brand-400 group-hover:text-brand-600">{{ String(i + 1).padStart(2, '0') }}</span>
            {{ q }}
          </button>
        </div>
      </div>
    </aside>

    <!-- ===== 中栏：策略切换 + 对话 ===== -->
    <main class="flex-1 min-w-0 min-h-0 flex flex-col overflow-hidden">
      <div class="shrink-0 px-5 pt-3 flex items-center justify-between gap-3">
        <div class="flex items-center gap-2 min-w-0">
          <div class="flex items-center rounded-lg border border-black/[0.08] bg-white p-0.5" role="group" aria-label="Agent 策略">
            <button
              v-for="item in ([
                { value: 'direct', label: '单智能体' },
                { value: 'auto', label: '智能路由' },
                { value: 'multi_expert', label: '多专家协作' },
              ] as const)"
              :key="item.value"
              class="rounded-md px-2.5 py-1.5 text-[11px] transition-colors"
              :class="agentStrategy === item.value ? 'bg-ink text-white' : 'text-ink-mute hover:bg-paper'"
              :aria-pressed="agentStrategy === item.value"
              @click="agentStrategy = item.value"
            >
              {{ item.label }}
            </button>
          </div>
        </div>
        <span v-if="canChat" class="hidden sm:inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-medium text-emerald-700 ring-1 ring-emerald-200">
          <Icon name="check" :size="10" /> 已索引 · 章节全文可检索
        </span>
      </div>

      <ChatPanel
        v-if="canChat"
        :key="`${selectedFileId || 'unselected'}_${sessionEpoch}`"
        ref="chatRef"
        :role="role"
        domain="novel"
        :strategy="agentStrategy"
        :file-id="selectedFileId"
        :session-key="sessionKey"
        :suggestions="suggestions"
        citation-panel
        title="阅读你的"
        title-suffix="小说"
        subtitle="上传小说文本后，可追问人物关系、情节因果、时间线和章节位置。复杂问题可由人物、情节、时间线和章节定位专家并发分析。"
        :notice="selectedFile ? `严格基于《${selectedFile.filename}》的原文回答 · 关键结论附章节引用` : '严格基于已索引原文回答 · 关键结论附章节引用'"
        @session-created="onSessionCreated"
      />
      <div v-else class="flex flex-1 items-center justify-center px-6 py-16 text-center">
        <div class="max-w-md rounded-2xl border border-dashed border-amber-300 bg-amber-50/70 px-6 py-8">
          <Icon name="folder-open" :size="28" class="mx-auto text-amber-600" />
          <p class="mt-3 text-sm font-semibold text-ink">
            {{ hasMultipleNovels ? '请在左侧选择一本小说' : '请先上传并索引小说' }}
          </p>
          <p class="mt-2 text-xs leading-5 text-ink-mute">
            {{ hasMultipleNovels ? '点击左侧会话列表上方的书籍信息，或前往书架选择。' : '小说完成索引后才能开始问答。' }}
          </p>
        </div>
      </div>
    </main>
  </div>
</template>
