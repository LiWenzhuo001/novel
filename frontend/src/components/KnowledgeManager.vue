<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import Icon from './Icon.vue'
import { uploadKB, listKB, deleteKB, reindexKB, type KBFileInfo } from '../api/client'
import { showToast } from '../utils/toast'

type KBFile = KBFileInfo

const emit = defineEmits<{
  (event: 'files-changed', files: KBFile[]): void
  (event: 'select', file: KBFile): void
}>()

withDefaults(defineProps<{
  domain?: 'novel'
  selectedFileId?: string | null
}>(), { domain: 'novel', selectedFileId: null })

// 点书即切换当前咨询对象；未就绪的书给出原因提示。
const onCardClick = (file: KBFile) => {
  if (file.status === 'indexed') {
    emit('select', file)
  } else if (file.status === 'failed') {
    showToast('该小说索引失败，无法选择')
  } else {
    showToast('该小说正在索引中，完成后即可选择')
  }
}

const files = ref<KBFile[]>([])
const loading = ref(true)
const loadError = ref('')
const uploading = ref(false)
const deletingId = ref('')
const reindexingId = ref('')
const fileInput = ref<HTMLInputElement>()
const polling = ref(false)
const knownStatuses = new Map<string, KBFile['status']>()
let pollTimer: ReturnType<typeof setTimeout> | null = null

const isBusy = (file: KBFile) => file.status === 'pending' || file.status === 'indexing'

const indexStageText: Record<NonNullable<KBFile['index_stage']>, string> = {
  pending: '等待索引任务启动',
  loading: '正在读取文件',
  parsing: '正在解析文本和章节',
  analyzing_chapters: '正在分析章节格式',
  building_embeddings: '正在建立向量索引',
  switching: '正在切换索引',
  completed: '索引完成',
  failed: '索引失败',
}

const progressFor = (file: KBFile) => {
  const value = Number(file.index_progress)
  if (!Number.isFinite(value)) return 0
  return Math.min(100, Math.max(0, Math.round(value)))
}

const indexProgressText = (file: KBFile) => {
  if (file.index_message) return file.index_message
  if (file.index_stage && indexStageText[file.index_stage]) return indexStageText[file.index_stage]
  return file.status === 'pending' ? '等待索引任务启动' : '正在建立索引'
}

const hasProgress = (file: KBFile) => isBusy(file) && file.status !== 'failed'

// quiet 刷新用于后台轮询，避免索引进度更新时反复触发整块加载骨架屏。
const load = async (quiet = false) => {
  if (!quiet) loading.value = true
  try {
    const { data } = await listKB()
    const nextFiles = data || []
    for (const file of nextFiles) {
      const previous = knownStatuses.get(file.id)
      if ((previous === 'pending' || previous === 'indexing') && file.status === 'indexed') {
        const chapterText = file.chapter_count ? `，识别 ${file.chapter_count} 章` : ''
        showToast(`《${displayName(file.filename)}》索引完成${chapterText}`)
      }
      knownStatuses.set(file.id, file.status)
    }
    files.value = nextFiles
    emit('files-changed', nextFiles)
    loadError.value = ''
  } catch (error: any) {
    loadError.value = error?.message || '知识库加载失败'
  } finally {
    if (!quiet) loading.value = false
  }
}

const statusMeta = (file: KBFile): { text: string; cls: string; icon?: string } => {
  if (file.status === 'pending') return { text: '等待索引', cls: 'text-amber-700 bg-amber-50 ring-amber-200', icon: 'loader' }
  if (file.status === 'indexing') {
    return { text: indexProgressText(file), cls: 'text-brand-700 bg-brand-50 ring-brand-200', icon: 'loader' }
  }
  if (file.status === 'failed') return { text: '索引失败', cls: 'text-rose-700 bg-rose-50 ring-rose-200', icon: 'x' }
  if (file.chapter_index_stale) return { text: '需更新章节索引', cls: 'text-amber-700 bg-amber-50 ring-amber-200', icon: 'refresh' }
  if (file.chapter_parse_status === 'unrecognized') {
    return { text: '可检索 · 未识别章节', cls: 'text-amber-700 bg-amber-50 ring-amber-200', icon: 'check' }
  }
  if (file.chapter_parser_mode === 'llm_assisted') {
    return { text: `可检索 · 模型辅助识别 ${file.chapter_count || 0} 章`, cls: 'text-violet-700 bg-violet-50 ring-violet-200', icon: 'sparkles' }
  }
  return { text: '可检索', cls: 'text-emerald-700 bg-emerald-50 ring-emerald-200', icon: 'check' }
}

const shouldOfferReindex = (file: KBFile) => !isBusy(file) && (
  file.status === 'failed'
  || file.chapter_index_stale
  || file.chapter_parse_status === 'unrecognized'
  || Boolean(file.index_warning)
)

const chapterSummary = (file: KBFile) => file.chapter_count
  ? `${file.chapter_count} 章 · ${file.chunks} 段`
  : `${file.chunks} 段`

const stopPolling = () => {
  if (pollTimer) clearTimeout(pollTimer)
  pollTimer = null
  polling.value = false
}

// 沿用轻量轮询获取索引阶段；所有文件完成或失败后自动停止定时器。
const startPolling = () => {
  if (pollTimer) return
  polling.value = true
  const tick = async () => {
    await load(true)
    if (!files.value.some(isBusy) || loadError.value) {
      stopPolling()
      return
    }
    pollTimer = setTimeout(tick, 1500)
  }
  pollTimer = setTimeout(tick, 1500)
}

const totalChunks = computed(() => files.value.reduce((sum, file) => sum + (file.chunks || 0), 0))
const onPick = () => fileInput.value?.click()

// 上传只负责提交原文，索引进度随后由文件列表轮询展示。
const onChange = async () => {
  const file = fileInput.value?.files?.[0]
  if (!file) return
  if (file.size > 10 * 1024 * 1024) {
    showToast('文件不能超过 10 MB')
    if (fileInput.value) fileInput.value.value = ''
    return
  }
  uploading.value = true
  try {
    await uploadKB(file)
    showToast('上传成功，正在建立索引')
    await load(true)
    startPolling()
  } catch (error: any) {
    showToast(`上传失败：${error?.message || '未知错误'}`)
  } finally {
    uploading.value = false
    if (fileInput.value) fileInput.value.value = ''
  }
}

// 重新索引保留原 file_id 和会话绑定，后端负责原子替换向量。
const onReindex = async (file: KBFile) => {
  if (!window.confirm(`将重新解析《${displayName(file.filename)}》的章节并重建向量索引，是否继续？`)) return
  reindexingId.value = file.id
  try {
    await reindexKB(file.id)
    showToast('重新索引任务已启动')
    await load(true)
    startPolling()
  } catch (error: any) {
    showToast(`重新索引失败：${error?.message || '未知错误'}`)
  } finally {
    reindexingId.value = ''
  }
}

const onDelete = async (file: KBFile) => {
  if (!window.confirm(`确定移除《${displayName(file.filename)}》吗？相关索引也会一并删除。`)) return
  deletingId.value = file.id
  try {
    await deleteKB(file.id)
    knownStatuses.delete(file.id)
    showToast('文档已移除')
    await load(true)
  } catch (error: any) {
    showToast(`删除失败：${error?.message || '未知错误'}`)
  } finally {
    deletingId.value = ''
  }
}

const extOf = (name: string) => (name.split('.').pop() || '').toUpperCase().slice(0, 4)
const displayName = (name: string) => name.replace(/\.(pdf|docx|txt|md)$/i, '')

onMounted(async () => {
  await load()
  if (files.value.some(isBusy)) startPolling()
})

// 离开页面时清理轮询，避免组件已卸载仍请求知识库接口。
onUnmounted(stopPolling)
</script>

<template>
  <div class="flex flex-col h-full">
    <input ref="fileInput" type="file" accept=".pdf,.docx,.txt,.md" class="hidden" @change="onChange" />
    <div class="flex items-center justify-between mb-3">
      <p class="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-faint flex items-center gap-1.5">
        <Icon name="folder-open" :size="13" class="text-brand-500" /> 小说书库
      </p>
      <span v-if="files.length" class="chip !text-[10px]">{{ files.length }} 份 · {{ totalChunks }} 段</span>
    </div>

    <button
      @click="onPick"
      :disabled="uploading"
      class="group relative flex flex-col items-center justify-center gap-2 py-6 rounded-lg border border-dashed border-brand-300 bg-brand-50/50 text-brand-700 transition-colors duration-200 hover:border-brand-500 hover:bg-brand-50 cursor-pointer disabled:cursor-not-allowed disabled:opacity-60"
    >
      <span class="w-10 h-10 rounded-lg bg-white ring-1 ring-brand-100 shadow-card flex items-center justify-center">
        <Icon :name="uploading ? 'loader' : 'upload-cloud'" :size="20" />
      </span>
      <span class="text-[13px] font-semibold">{{ uploading ? '正在上传并解析' : '上传小说文本' }}</span>
      <span class="text-[11px] text-ink-mute">TXT / MD / PDF / DOCX · 最大 10 MB</span>
    </button>

    <div class="mt-3 flex-1 overflow-y-auto scroll-thin space-y-1.5" :aria-busy="loading || polling">
      <div v-if="loading" class="space-y-2" role="status" aria-label="正在加载小说书库">
        <div v-for="n in 3" :key="n" class="h-14 rounded-lg bg-white/70 ring-1 ring-black/[0.05] animate-pulse"></div>
      </div>
      <div v-else-if="loadError" class="rounded-lg border border-rose-200 bg-rose-50 px-4 py-4" role="alert">
        <p class="text-sm font-semibold text-rose-700">书库加载失败</p>
        <p class="mt-1 text-xs leading-5 text-rose-600 break-words">{{ loadError }}</p>
        <button class="mt-2 text-xs font-semibold text-rose-700 underline underline-offset-2" @click="load()">重新加载</button>
      </div>
      <template v-else>
        <div
          v-for="file in files"
          :key="file.id"
          class="group flex items-center gap-2.5 rounded-lg bg-white ring-1 px-3 py-2.5 transition-all duration-200 animate-fadeIn"
          :class="[
            file.id === selectedFileId
              ? 'ring-2 ring-inset ring-brand-400 bg-brand-50/60 cursor-pointer'
              : 'ring-black/[0.07] hover:ring-brand-200',
            file.status === 'indexed' ? 'cursor-pointer' : 'cursor-default',
          ]"
          role="button"
          tabindex="0"
          :aria-pressed="file.id === selectedFileId"
          :aria-label="`选择 ${displayName(file.filename)} 为当前咨询小说`"
          @click="onCardClick(file)"
          @keydown.enter="onCardClick(file)"
        >
          <span class="shrink-0 w-9 h-9 rounded-lg bg-brand-50 ring-1 ring-brand-100/60 flex items-center justify-center text-[9px] font-bold text-brand-700 tracking-wide">
            {{ extOf(file.filename) }}
          </span>
          <div class="flex-1 min-w-0">
            <p class="text-[13px] font-medium text-ink truncate" :title="displayName(file.filename)">
              {{ displayName(file.filename) }}
              <span v-if="file.is_system" class="ml-1 inline-flex items-center rounded-full bg-brand-50 px-1.5 py-0.5 text-[9px] font-semibold text-brand-700 ring-1 ring-brand-100 align-middle">系统</span>
              <span v-if="file.id === selectedFileId" class="ml-1 inline-flex items-center rounded-full bg-emerald-50 px-1.5 py-0.5 text-[9px] font-semibold text-emerald-700 ring-1 ring-emerald-200 align-middle">当前</span>
            </p>
            <div class="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-ink-faint">
              <span :class="['inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 ring-1', statusMeta(file).cls]">
                <Icon :name="statusMeta(file).icon || 'check'" :size="10" />
                {{ statusMeta(file).text }}
              </span>
              <span>{{ chapterSummary(file) }}</span>
              <span v-if="file.chapter_parser_mode === 'inline_fallback'">· 兼容模式识别</span>
              <span v-if="file.chapter_parser_mode === 'llm_assisted' && file.chapter_rule_confidence != null">· 置信度 {{ Math.round(file.chapter_rule_confidence * 100) }}%</span>
              <span v-if="file.detected_encoding && !file.detected_encoding.startsWith('utf-8')">· {{ file.detected_encoding.toUpperCase() }}</span>
              <span aria-hidden="true">·</span>
              <span>{{ file.created_at }}</span>
            </div>
            <!-- 索引任务进行中时，通过现有轮询刷新阶段进度。 -->
            <div v-if="hasProgress(file)" class="mt-2" role="group" :aria-label="`${indexProgressText(file)}，${progressFor(file)}%`">
              <div class="flex items-center justify-between gap-2 text-[10px] text-ink-mute">
                <span class="truncate">{{ indexProgressText(file) }}</span>
                <span class="shrink-0 tabular-nums">{{ progressFor(file) }}%</span>
              </div>
              <div
                class="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-brand-100"
                role="progressbar"
                :aria-label="`${displayName(file.filename)} 索引进度`"
                aria-valuemin="0"
                aria-valuemax="100"
                :aria-valuenow="progressFor(file)"
              >
                <div
                  class="h-full rounded-full bg-brand-500 transition-[width] duration-500 ease-out"
                  :style="{ width: `${progressFor(file)}%` }"
                ></div>
              </div>
            </div>
            <p v-if="file.status === 'failed'" class="text-[10px] text-rose-600 mt-0.5 leading-snug break-words">
              {{ file.error || file.index_message || '索引失败，旧索引仍可使用' }}
            </p>
            <p v-else-if="file.index_warning && !hasProgress(file)" class="text-[10px] text-amber-700 mt-0.5 leading-snug break-words">
              {{ file.index_warning }}
            </p>
          </div>
          <div class="shrink-0 flex items-center gap-0.5">
            <button
              v-if="!file.is_system && shouldOfferReindex(file)"
              @click.stop="onReindex(file)"
              class="w-8 h-8 rounded-lg flex items-center justify-center text-amber-600 transition-colors hover:bg-amber-50 cursor-pointer disabled:cursor-not-allowed disabled:opacity-30"
              :disabled="isBusy(file) || reindexingId === file.id"
              :aria-label="`重新索引 ${displayName(file.filename)}`"
              title="重新解析章节并建立索引"
            >
              <Icon :name="reindexingId === file.id ? 'loader' : 'refresh'" :size="14" />
            </button>
            <button
              v-if="!file.is_system"
              @click.stop="onDelete(file)"
              class="w-8 h-8 rounded-lg flex items-center justify-center text-ink-faint opacity-100 sm:opacity-0 sm:group-hover:opacity-100 sm:focus-visible:opacity-100 transition-colors hover:bg-rose-50 hover:text-rose-600 cursor-pointer disabled:cursor-not-allowed disabled:opacity-30"
              :disabled="isBusy(file) || deletingId === file.id"
              :aria-label="`删除 ${displayName(file.filename)}`"
            >
              <Icon :name="deletingId === file.id ? 'loader' : 'trash'" :size="14" />
            </button>
          </div>
        </div>
      </template>
      <div v-if="!loading && !loadError && !files.length" class="rounded-lg border border-dashed border-black/[0.1] bg-white/50 py-8 text-center">
        <p class="text-xs text-ink-faint leading-5">内置《红楼梦》可直接问答<br />也可上传自己的小说</p>
      </div>
    </div>
  </div>
</template>
