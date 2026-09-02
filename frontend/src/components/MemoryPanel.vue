<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import Icon from './Icon.vue'
import { deleteMemory, getMemoryContext, listMemories, type MemoryContext, type MemoryItem } from '../api/client'
import { showToast } from '../utils/toast'

// 双模式组件：
// - 传入 sessionKey → 会话级（会话摘要 + 该会话记忆），用于聊天侧栏；
// - 不传 sessionKey → 用户级（跨会话全部记忆，不展示会话摘要），用于记忆中心页。
const props = withDefaults(defineProps<{
  sessionKey?: string
  fileId?: string | null
}>(), { sessionKey: '', fileId: null })

const loading = ref(false)
const error = ref('')
const context = ref<MemoryContext>({ summary: '', memories: [] })
const removing = ref('')
const expanded = ref(true)

const sessionScoped = computed(() => Boolean(props.sessionKey))
const sessionId = computed(() => (props.sessionKey ? localStorage.getItem(props.sessionKey) || '' : ''))
const grouped = computed(() => {
  const order = (item: MemoryItem) => item.created_at || ''
  return {
    preference: context.value.memories
      .filter((item) => item.memory_type === 'user_preference')
      .sort((left, right) => order(right).localeCompare(order(left))),
    novel: context.value.memories
      .filter((item) => item.memory_type === 'novel_fact')
      .sort((left, right) => order(right).localeCompare(order(left))),
    session: context.value.memories
      .filter((item) => item.memory_type === 'session_fact')
      .sort((left, right) => order(right).localeCompare(order(left))),
  }
})

const load = async () => {
  loading.value = true
  error.value = ''
  try {
    if (sessionScoped.value && !sessionId.value) {
      context.value = { summary: '', memories: [] }
      return
    }
    if (sessionScoped.value) {
      const [summaryResponse, memoryResponse] = await Promise.all([
        getMemoryContext(sessionId.value, props.fileId),
        listMemories(sessionId.value, props.fileId),
      ])
      context.value = {
        summary: summaryResponse.data?.summary || '',
        summary_id: summaryResponse.data?.summary_id,
        memories: memoryResponse.data || summaryResponse.data?.memories || [],
      }
    } else {
      // 用户级：拉取名下全部记忆（跨会话）；后端 /memories 不带过滤即用户范围。
      const memoryResponse = await listMemories()
      context.value = { summary: '', memories: memoryResponse.data || [] }
    }
  } catch (e: any) {
    error.value = e?.message || '记忆加载失败'
  } finally {
    loading.value = false
  }
}

const remove = async (item: MemoryItem) => {
  if (removing.value) return
  removing.value = item.id
  try {
    await deleteMemory(item.id)
    context.value.memories = context.value.memories.filter((memory) => memory.id !== item.id)
    showToast('已删除这条记忆')
  } catch (e: any) {
    showToast(e?.message || '记忆删除失败')
  } finally {
    removing.value = ''
  }
}

const label = (type: string) => ({
  user_preference: '用户偏好',
  novel_fact: '小说记忆',
  session_fact: '本轮会话',
}[type] || '记忆')

const sections = computed(() => [
  { key: 'preference', title: '用户偏好', items: grouped.value.preference },
  { key: 'novel', title: '小说记忆', items: grouped.value.novel },
  { key: 'session', title: '会话记忆', items: grouped.value.session },
].filter((section) => section.items.length))

watch(() => [props.sessionKey, props.fileId], () => void load())
onMounted(() => void load())

defineExpose({ refresh: load })
</script>

<template>
  <section class="rounded-xl border border-black/[0.08] bg-white/70 shadow-sm overflow-hidden" aria-label="对话记忆">
    <button
      class="w-full flex items-center justify-between gap-3 px-3.5 py-3 text-left hover:bg-brand-50/50 transition-colors"
      @click="expanded = !expanded"
    >
      <span class="flex items-center gap-2 text-xs font-semibold text-ink">
        <Icon name="sparkles" :size="14" class="text-brand-600" /> 对话记忆
        <span v-if="context.memories.length" class="chip !py-0.5 !px-1.5 !text-[10px]">{{ context.memories.length }}</span>
      </span>
      <Icon :name="expanded ? 'chevron-down' : 'chevron-right'" :size="13" class="text-ink-faint" />
    </button>

    <div v-if="expanded" class="border-t border-black/[0.06] px-3.5 py-3">
      <div v-if="loading" class="flex items-center gap-2 text-xs text-ink-faint">
        <Icon name="loader" :size="13" /> 正在加载记忆
      </div>
      <p v-else-if="error" class="text-xs leading-5 text-rose-600">{{ error }}</p>
      <template v-else>
        <div v-if="sessionScoped" class="mb-3">
          <div v-if="context.summary" class="rounded-lg bg-brand-50/60 px-3 py-2.5">
            <p class="mb-1 text-[10px] font-semibold uppercase tracking-wide text-brand-700">会话摘要</p>
            <p class="text-xs leading-5 text-ink-soft">{{ context.summary }}</p>
          </div>
          <p v-else class="text-xs text-ink-faint">暂时还没有自动摘要。</p>
        </div>

        <div v-if="!context.memories.length" class="rounded-lg border border-dashed border-black/[0.1] px-3 py-3 text-xs leading-5 text-ink-faint">
          {{ sessionScoped ? '继续对话后，系统会自动记住稳定的偏好和小说事实。' : '还没有任何记忆。继续对话后，系统会自动记住稳定的偏好和小说事实。' }}
        </div>
        <div v-else class="space-y-3">
          <div v-for="section in sections" :key="section.key">
            <p class="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-ink-faint">{{ section.title }}</p>
            <div class="space-y-2">
              <div v-for="item in section.items" :key="item.id" class="rounded-lg bg-paper px-2.5 py-2 ring-1 ring-black/[0.05]">
                <div class="flex items-center justify-between gap-2">
                  <span class="text-[10px] font-semibold text-brand-700">{{ label(item.memory_type) }}</span>
                  <button
                    class="text-ink-faint hover:text-rose-600 disabled:opacity-40"
                    :disabled="removing === item.id"
                    :aria-label="`删除${label(item.memory_type)}`"
                    @click.stop="remove(item)"
                  >
<Icon name="x" :size="12" />
</button>
                </div>
                <p class="mt-1 text-xs leading-5 text-ink-soft">{{ item.content }}</p>
              </div>
            </div>
          </div>
        </div>
      </template>
    </div>
  </section>
</template>
