<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import Icon from '../../components/Icon.vue'
import {
  listKB, uploadKB, deleteKB, reindexKB,
  type KBFileInfo,
} from '../../api/client'
import { showToast } from '../../utils/toast'

type KBFile = KBFileInfo

const files = ref<KBFile[]>([])
const loading = ref(true)
const loadError = ref('')
const uploading = ref(false)
const deletingId = ref('')
const reindexingId = ref('')
const fileInput = ref<HTMLInputElement>()
const knownStatuses = new Map<string, KBFile['status']>()
const searchKeyword = ref('')
const statusFilter = ref<'all' | 'indexing' | 'failed' | 'indexed'>('all')
const selectedFileId = ref<string | null>(localStorage.getItem('novel_selected_file_id'))

const COVER_THEMES = [
  { bg: 'from-amber-200 to-amber-400', ink: 'text-amber-900', deco: 'bg-amber-300/50' },
  { bg: 'from-rose-200 to-rose-400', ink: 'text-rose-900', deco: 'bg-rose-300/50' },
  { bg: 'from-emerald-200 to-emerald-400', ink: 'text-emerald-900', deco: 'bg-emerald-300/50' },
  { bg: 'from-sky-200 to-sky-400', ink: 'text-sky-900', deco: 'bg-sky-300/50' },
]
const coverTheme = (file: KBFile) => {
  let hash = 0
  for (const ch of file.filename) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0
  return COVER_THEMES[hash % COVER_THEMES.length]
}

const isBusy = (file: KBFile) => file.status === 'pending' || file.status === 'indexing'

const statusMeta = (file: KBFile): { text: string; cls: string; icon?: string } => {
  if (file.status === 'pending') return { text: '等待索引', cls: 'text-amber-700 bg-amber-50 ring-amber-200', icon: 'loader' }
  if (file.status === 'indexing') {
    const stage = file.index_message || '正在建立索引'
    return { text: file.index_progress ? `${stage} ${Math.round(Number(file.index_progress))}%` : stage, cls: 'text-brand-700 bg-brand-50 ring-brand-200', icon: 'loader' }
  }
  if (file.status === 'failed') return { text: '索引失败 · 可重试', cls: 'text-rose-700 bg-rose-50 ring-rose-200', icon: 'x' }
  return { text: '已索引 · 可问答', cls: 'text-emerald-700 bg-emerald-50 ring-emerald-200', icon: 'check' }
}

const progressFor = (file: KBFile) => {
  const value = Number(file.index_progress)
  if (!Number.isFinite(value)) return 0
  return Math.min(100, Math.max(0, Math.round(value)))
}
const hasProgress = (file: KBFile) => isBusy(file) && file.status !== 'failed'

const displayName = (filename: string) => filename.replace(/\.[^.]+$/, '')
const extOf = (filename: string) => (filename.split('.').pop() || 'TXT').toUpperCase().slice(0, 4)
const chapterSummary = (file: KBFile) => {
  const parts: string[] = []
  if (file.chapter_count) parts.push(`${file.chapter_count} 章`)
  if (file.chunks) parts.push(`${file.chunks} 片段`)
  return parts.join(' · ')
}

const filteredFiles = computed(() => {
  let list = files.value
  if (searchKeyword.value.trim()) {
    const kw = searchKeyword.value.trim().toLowerCase()
    list = list.filter((f) => f.filename.toLowerCase().includes(kw))
  }
  if (statusFilter.value === 'indexing') list = list.filter((f) => isBusy(f))
  else if (statusFilter.value === 'failed') list = list.filter((f) => f.status === 'failed')
  else if (statusFilter.value === 'indexed') list = list.filter((f) => f.status === 'indexed')
  return list
})

const counts = computed(() => ({
  all: files.value.length,
  indexing: files.value.filter((f) => isBusy(f)).length,
  failed: files.value.filter((f) => f.status === 'failed').length,
  indexed: files.value.filter((f) => f.status === 'indexed').length,
}))

const load = async (quiet = false) => {
  if (!quiet) loading.value = true
  try {
    const { data } = await listKB()
    const nextFiles = data || []
    for (const file of nextFiles) {
      const previous = knownStatuses.get(file.id)
      if ((previous === 'pending' || previous === 'indexing') && file.status === 'indexed') {
        showToast(`《${displayName(file.filename)}》索引完成`)
      }
      knownStatuses.set(file.id, file.status)
    }
    files.value = nextFiles
    loadError.value = ''
  } catch (error: any) {
    loadError.value = error?.message || '书库加载失败'
  } finally {
    if (!quiet) loading.value = false
  }
}

let pollTimerHandle: ReturnType<typeof setTimeout> | null = null
const startPolling = () => {
  const tick = () => {
    if (files.value.some((f) => isBusy(f))) {
      void load(true)
      pollTimerHandle = setTimeout(tick, 1500)
    } else {
      pollTimerHandle = setTimeout(tick, 5000)
    }
  }
  pollTimerHandle = setTimeout(tick, 1500)
}

const onPick = () => fileInput.value?.click()
const onChange = async (event: Event) => {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  uploading.value = true
  try {
    await uploadKB(file)
    showToast('上传成功，已开始索引')
    await load(true)
  } catch (error: any) {
    showToast(error?.message || '上传失败')
  } finally {
    uploading.value = false
    input.value = ''
  }
}

const selectAsCurrent = (file: KBFile) => {
  if (file.status !== 'indexed') {
    showToast(file.status === 'failed' ? '该小说索引失败，无法选择' : '该小说正在索引中，完成后即可选择')
    return
  }
  selectedFileId.value = file.id
  localStorage.setItem('novel_selected_file_id', file.id)
  showToast(`已将《${displayName(file.filename)}》设为当前咨询书籍`)
}

const onReindex = async (file: KBFile) => {
  reindexingId.value = file.id
  try {
    await reindexKB(file.id)
    showToast('已提交重新索引')
    await load(true)
  } catch {
    showToast('重新索引提交失败')
  } finally {
    reindexingId.value = ''
  }
}

const onDelete = async (file: KBFile) => {
  if (!window.confirm(`删除《${displayName(file.filename)}》？其索引与对话记忆将一并清除。`)) return
  deletingId.value = file.id
  try {
    await deleteKB(file.id)
    if (selectedFileId.value === file.id) {
      selectedFileId.value = null
      localStorage.removeItem('novel_selected_file_id')
    }
    showToast('已删除')
    await load(true)
  } catch {
    showToast('删除失败')
  } finally {
    deletingId.value = ''
  }
}

onMounted(() => {
  void load()
  startPolling()
})
onUnmounted(() => {
  if (pollTimerHandle) clearTimeout(pollTimerHandle)
})
</script>

<template>
  <div class="h-full overflow-y-auto scroll-thin">
    <input ref="fileInput" type="file" accept=".pdf,.docx,.txt,.md" class="hidden" @change="onChange" />

    <div class="mx-auto w-full max-w-5xl px-5 py-6 sm:px-6 space-y-5">
      <!-- 工具条 -->
      <div class="flex flex-col gap-3 sm:flex-row sm:items-center">
        <div class="flex items-center gap-2 rounded-lg border border-black/[0.08] bg-white px-3 py-2 min-w-0 sm:w-56">
          <Icon name="search" :size="13" class="text-ink-faint" />
          <input
            v-model="searchKeyword"
            type="text"
            placeholder="搜索书名"
            class="min-w-0 flex-1 bg-transparent text-xs outline-none"
            aria-label="搜索书名"
          />
        </div>
        <div class="flex items-center gap-1.5">
          <button
            v-for="(label, key) in { all: `全部 ${counts.all}`, indexing: `索引中 ${counts.indexing}`, failed: `失败 ${counts.failed}`, indexed: `已索引 ${counts.indexed}` }"
            :key="key"
            class="rounded-full px-3 py-1.5 text-xs transition-colors"
            :class="statusFilter === key ? 'bg-brand-600 text-white' : 'bg-white text-ink-mute ring-1 ring-black/[0.08] hover:text-ink'"
            @click="statusFilter = key as any"
          >
            {{ label }}
          </button>
        </div>
        <button class="btn-primary sm:ml-auto !rounded-lg px-4 py-2 text-xs flex items-center gap-1.5" :disabled="uploading" @click="onPick">
          <Icon :name="uploading ? 'loader' : 'upload-cloud'" :size="14" />
          {{ uploading ? '上传中…' : '上传书籍' }}
        </button>
      </div>

      <!-- 说明行 -->
      <p class="text-[11px] text-ink-faint">
        共 {{ counts.all }} 本 · 已索引 {{ counts.indexed }} · 点击卡片可将其设为当前咨询书籍
      </p>

      <!-- 加载/错误态 -->
      <div v-if="loading" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4" role="status" aria-label="正在加载书库">
        <div v-for="n in 4" :key="n" class="h-48 rounded-xl bg-white/70 ring-1 ring-black/[0.05] animate-pulse"></div>
      </div>
      <div v-else-if="loadError" class="rounded-lg border border-rose-200 bg-rose-50 px-4 py-4" role="alert">
        <p class="text-sm font-semibold text-rose-700">书库加载失败</p>
        <p class="mt-1 text-xs text-rose-600">{{ loadError }}</p>
        <button class="mt-2 text-xs font-semibold text-rose-700 underline underline-offset-2" @click="load()">重新加载</button>
      </div>

      <!-- 书卡网格 -->
      <div v-else class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div
          v-for="file in filteredFiles"
          :key="file.id"
          class="group cursor-pointer overflow-hidden rounded-xl bg-white ring-1 transition-all duration-200 hover:shadow-pop"
          :class="file.id === selectedFileId ? 'ring-2 ring-brand-500' : 'ring-black/[0.07] hover:ring-brand-200'"
          role="button"
          tabindex="0"
          :aria-pressed="file.id === selectedFileId"
          @click="selectAsCurrent(file)"
          @keydown.enter="selectAsCurrent(file)"
        >
          <!-- 彩色封面横幅 -->
          <div class="relative h-24 bg-gradient-to-br" :class="coverTheme(file).bg">
            <span class="absolute inset-x-3 bottom-0 h-6" :class="coverTheme(file).deco" style="clip-path: polygon(0 100%, 12% 40%, 26% 78%, 40% 28%, 55% 70%, 70% 22%, 84% 62%, 100% 35%, 100% 100%, 0 100%)"></span>
            <span
              class="absolute right-2.5 top-2.5 font-display text-lg font-bold tracking-widest opacity-90"
              :class="coverTheme(file).ink"
              style="writing-mode: vertical-rl"
            >{{ displayName(file.filename).slice(0, 6) }}</span>
            <span class="absolute left-2.5 bottom-2 rounded bg-white/80 px-1.5 py-0.5 text-[9px] font-bold text-ink-mute">{{ extOf(file.filename) }}</span>
            <span
              v-if="file.id === selectedFileId"
              class="absolute left-2.5 top-2.5 rounded-full bg-brand-600 px-2 py-0.5 text-[9px] font-semibold text-white"
            >当前咨询</span>
          </div>

          <!-- 信息区 -->
          <div class="px-3.5 py-3">
            <p class="truncate text-[13px] font-semibold text-ink" :title="file.filename">{{ file.filename }}</p>
            <p class="mt-1 text-[11px] text-ink-faint">{{ chapterSummary(file) || '尚未解析' }}</p>
            <div class="mt-2">
              <span :class="['inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] ring-1', statusMeta(file).cls]">
                <Icon :name="statusMeta(file).icon || 'check'" :size="10" />
                {{ statusMeta(file).text }}
              </span>
            </div>
            <div v-if="hasProgress(file)" class="mt-2">
              <div class="h-1.5 w-full overflow-hidden rounded-full bg-brand-100">
                <div class="h-full rounded-full bg-brand-500 transition-[width] duration-500" :style="{ width: `${progressFor(file)}%` }"></div>
              </div>
            </div>
            <p v-if="file.status === 'failed'" class="mt-1.5 line-clamp-2 text-[10px] leading-snug text-rose-600">
              {{ file.error || file.index_message || '索引失败' }}
            </p>
          </div>

          <!-- 操作 -->
          <div class="flex items-center justify-end gap-1 border-t border-black/[0.05] px-2 py-1.5">
            <button
              v-if="!file.is_system"
              class="rounded p-1.5 text-amber-600 hover:bg-amber-50"
              :disabled="isBusy(file) || reindexingId === file.id"
              title="重新索引"
              @click.stop="onReindex(file)"
            >
              <Icon :name="reindexingId === file.id ? 'loader' : 'refresh'" :size="13" />
            </button>
            <button
              v-if="!file.is_system"
              class="rounded p-1.5 text-ink-faint hover:bg-rose-50 hover:text-rose-600"
              :disabled="isBusy(file) || deletingId === file.id"
              title="删除"
              @click.stop="onDelete(file)"
            >
              <Icon :name="deletingId === file.id ? 'loader' : 'trash'" :size="13" />
            </button>
          </div>
        </div>

        <div v-if="!filteredFiles.length" class="col-span-full rounded-lg border border-dashed border-black/[0.1] bg-white/50 py-10 text-center">
          <p class="text-xs text-ink-faint">没有匹配的书籍，试试调整筛选或上传新书</p>
        </div>
      </div>
    </div>
  </div>
</template>
