<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import Icon from '../../components/Icon.vue'
import {
  listKB, listSessions,
  type KBFileInfo, type SessionItem,
} from '../../api/client'
import { showToast } from '../../utils/toast'

const router = useRouter()

const novels = ref<KBFileInfo[]>([])
const sessions = ref<SessionItem[]>([])
const sessionsLoading = ref(false)
const renameTarget = ref<SessionItem | null>(null)
const renameTitle = ref('')
const renameInputEl = ref<HTMLInputElement | null>(null)

const indexedNovels = computed(() => novels.value.filter((n) => n.status === 'indexed'))
const selectedFileId = ref<string | null>(localStorage.getItem('novel_selected_file_id'))
const selectedFile = computed(() => novels.value.find((n) => n.id === selectedFileId.value) || null)
const currentBook = computed(() => {
  if (selectedFile.value && selectedFile.value.status === 'indexed') return selectedFile.value
  return indexedNovels.value[0] || null
})
const hour = new Date().getHours()
const greeting = hour < 6 ? '夜深了' : hour < 12 ? '早上好' : hour < 18 ? '下午好' : '晚上好'
const bookLine = computed(() => {
  if (!currentBook.value) return '书库为空 · 上传小说后即可开始章节级溯源问答'
  const parts = [`《${currentBook.value.filename.replace(/\.[^.]+$/, '')}》`]
  if (currentBook.value.chapter_count) parts.push(`已索引 ${currentBook.value.chapter_count} 章`)
  if (currentBook.value.chunks) parts.push(`${currentBook.value.chunks} 片段`)
  return `${parts.join(' · ')} · 支持章节级原文溯源`
})

const loadNovels = async () => {
  try {
    const { data } = await listKB()
    novels.value = data || []
    if (!currentBook.value && indexedNovels.value.length) {
      const systemBook = novels.value.find((n) => n.is_system && n.status === 'indexed')
      selectedFileId.value = (systemBook || indexedNovels.value[0]).id
    }
  } catch {
    novels.value = []
  }
}

const loadSessions = async () => {
  sessionsLoading.value = true
  try {
    const { data } = await listSessions(currentBook.value?.id)
    // 只展示普通问答会话；角色扮演会话归小说世界。
    sessions.value = (data || []).filter((row) => !(row.personas || []).length).slice(0, 6)
  } catch {
    sessions.value = []
  } finally {
    sessionsLoading.value = false
  }
}

const resumeSession = (row: SessionItem) => {
  if (!row.file_id) {
    showToast('该会话未绑定小说，无法继续')
    return
  }
  localStorage.setItem('novel_selected_file_id', row.file_id)
  localStorage.setItem(`novel_rag_session_id_${row.file_id}`, row.id)
  router.push('/chat')
}

const startRename = (row: SessionItem) => {
  renameTarget.value = row
  renameTitle.value = row.title
  setTimeout(() => renameInputEl.value?.focus(), 50)
}

const commitRename = async () => {
  const row = renameTarget.value
  if (!row) return
  const title = renameTitle.value.trim()
  renameTarget.value = null
  if (!title || title === row.title) return
  try {
    const { renameSession } = await import('../../api/client')
    await renameSession(row.id, title)
    row.title = title
  } catch {
    showToast('重命名失败，请重试')
  }
}

const cancelRename = () => {
  renameTarget.value = null
  renameTitle.value = ''
}

const removeSession = async (row: SessionItem) => {
  if (!window.confirm(`删除对话「${row.title}」？消息与相关记忆将一并清除。`)) return
  try {
    const { deleteSession } = await import('../../api/client')
    await deleteSession(row.id)
    sessions.value = sessions.value.filter((s) => s.id !== row.id)
    showToast('对话已删除')
  } catch {
    showToast('删除失败，请重试')
  }
}

const goChat = () => {
  if (currentBook.value) {
    localStorage.setItem('novel_selected_file_id', currentBook.value.id)
  }
  router.push('/chat')
}

const shortcuts = computed(() => [
  { title: '新建问答', desc: '就当前书籍发起新对话，流式回答', icon: 'messages', to: '/chat' },
  { title: '管理知识库', desc: '上传新书、查看索引进度、删除与重建', icon: 'book', to: '/library' },
  { title: '进入小说世界', desc: '选择角色与章节边界，开启角色扮演', icon: 'book-open', to: '/world' },
  { title: '查看对话记忆', desc: '浏览三层记忆，可按条清理', icon: 'layers', to: '/memories' },
])

void Promise.all([loadNovels()]).then(() => void loadSessions())
</script>

<template>
  <div class="h-full overflow-y-auto scroll-thin">
    <div class="mx-auto w-full max-w-4xl px-5 py-6 sm:px-6 space-y-6">
      <!-- 欢迎横幅 -->
      <section class="rounded-2xl bg-gradient-to-r from-brand-50 to-white border border-brand-100 px-6 py-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div class="min-w-0">
          <p class="font-display text-xl font-bold text-ink">{{ greeting }}，继续你的阅读问答</p>
          <p class="mt-1.5 text-xs leading-5 text-ink-mute truncate">{{ bookLine }}</p>
        </div>
        <button class="btn-primary shrink-0 !rounded-xl px-5 py-2.5 text-sm flex items-center gap-2" @click="goChat">
          <Icon name="messages" :size="15" /> 继续问答
        </button>
      </section>

      <!-- 快捷入口 -->
      <section>
        <div class="flex items-center justify-between mb-2.5">
          <p class="text-xs font-semibold text-ink">快捷入口</p>
          <p class="text-[11px] text-ink-faint">4 项 · 上下文联动的快速任务</p>
        </div>
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <button
            v-for="item in shortcuts"
            :key="item.to"
            class="surface p-4 text-left transition-all duration-200 hover:shadow-pop hover:ring-1 hover:ring-brand-200"
            @click="router.push(item.to)"
          >
            <span class="w-9 h-9 rounded-lg bg-brand-50 text-brand-600 flex items-center justify-center">
              <Icon :name="item.icon" :size="17" />
            </span>
            <p class="mt-2.5 text-[13px] font-semibold text-ink">{{ item.title }}</p>
            <p class="mt-1 text-[11px] leading-4 text-ink-mute">{{ item.desc }}</p>
          </button>
        </div>
      </section>

      <!-- 最近会话 -->
      <section class="surface p-4">
        <div class="flex items-center justify-between mb-2.5">
          <p class="text-xs font-semibold text-ink flex items-center gap-1.5">
            <Icon name="clock" :size="13" class="text-brand-500" /> 最近会话
            <template v-if="currentBook"> · {{ currentBook.filename }}</template>
          </p>
          <p class="text-[11px] text-ink-faint">{{ sessions.length }} 个会话 · 点击继续对话</p>
        </div>

        <div v-if="sessionsLoading" class="py-6 text-center text-xs text-ink-faint" role="status">加载中…</div>
        <p v-else-if="!sessions.length" class="py-6 text-center text-xs leading-5 text-ink-faint">
          还没有普通问答对话。去「问答工作台」发起第一条提问吧。
        </p>
        <div v-else class="space-y-1">
          <div
            v-for="row in sessions"
            :key="row.id"
            class="group cursor-pointer rounded-lg px-3 py-2.5 ring-1 transition-all duration-200"
            :class="row.id === sessions[0]?.id ? 'ring-brand-200 bg-brand-50/50' : 'ring-black/[0.07] bg-white hover:ring-brand-200'"
            role="button"
            tabindex="0"
            @click="resumeSession(row)"
            @keydown.enter="resumeSession(row)"
          >
            <div class="flex items-center justify-between gap-2">
              <template v-if="renameTarget?.id === row.id">
                <input
                  :ref="(el) => { if (el) (renameInputEl as HTMLInputElement | null) = el as HTMLInputElement }"
                  v-model="renameTitle"
                  class="min-w-0 flex-1 rounded-md border border-brand-300 px-2 py-1 text-[13px] outline-none focus:ring-2 focus:ring-brand-100"
                  @click.stop
                  @keyup.enter="commitRename"
                  @keyup.esc="cancelRename"
                  @blur="commitRename"
                />
              </template>
              <p v-else class="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">{{ row.title }}</p>
              <span class="shrink-0 text-[10px] text-ink-faint">{{ row.updated_at }}</span>
            </div>
            <div class="mt-1 flex items-center justify-between gap-2">
              <p class="text-[11px] text-ink-faint">
                {{ (row.personas || []).length ? '角色扮演' : '问答对话' }}
                <template v-if="row.file_id"> · {{ novels.find((n) => n.id === row.file_id)?.filename || '' }}</template>
              </p>
              <span class="hidden group-hover:flex items-center gap-1">
                <button
                  class="rounded p-1 text-ink-faint hover:bg-brand-50 hover:text-brand-600"
                  title="重命名"
                  @click.stop="startRename(row)"
                >
                  <Icon name="wrench" :size="12" />
                </button>
                <button
                  class="rounded p-1 text-ink-faint hover:bg-rose-50 hover:text-rose-600"
                  title="删除"
                  @click.stop="removeSession(row)"
                >
                  <Icon name="trash" :size="12" />
                </button>
              </span>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>
