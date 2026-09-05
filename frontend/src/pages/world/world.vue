<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import Icon from '../../components/Icon.vue'
import ChatPanel from '../../components/ChatPanel.vue'
import {
  listKB, listSessions, listWorldCharacters, selectWorldCharacters,
  type KBFileInfo, type CharacterCard, type SessionItem,
} from '../../api/client'
import { showToast } from '../../utils/toast'

const router = useRouter()
const MAX_PERSONAS = 3
const CONFIG_KEY = 'novel_world_config'

const novels = ref<KBFileInfo[]>([])
const selectedFileId = ref<string | null>(null)
const characters = ref<string[]>([])
const charsLoading = ref(false)
const charsError = ref('')
const selectedNames = ref<string[]>([])
const customName = ref('')
const chapterUntil = ref<number | null>(null)
const starting = ref(false)
const started = ref(false)
const cards = ref<CharacterCard[]>([])
const scenario = ref('')
const pastSessions = ref<SessionItem[]>([])
const pastError = ref('')

const indexedNovels = computed(() => novels.value.filter((n) => n.status === 'indexed'))
const selectedFile = computed(() => novels.value.find((n) => n.id === selectedFileId.value) || null)
const chapterCount = computed(() => selectedFile.value?.chapter_count || 0)
const chapterOptions = computed(() => Array.from({ length: chapterCount.value }, (_, i) => i + 1))
const canStart = computed(() =>
  Boolean(selectedFileId.value) && selectedNames.value.length > 0 && !starting.value,
)
const sessionKey = computed(() =>
  worldSessionKey(selectedFileId.value, selectedNames.value, chapterUntil.value),
)

const worldSessionKey = (fileId: string | null, names: string[], chapterUntil: number | null) =>
  `novel_world_${fileId || 'none'}_${[...names].sort().join('-')}_${chapterUntil || 'all'}`
const assistantNames = computed(() => selectedNames.value)
const membersLabel = computed(() => selectedNames.value.join(' · '))

// 开场消息：斜体情景旁白 + 各角色粗体名开场白（内容由缓存角色卡确定性生成）。
const openingMessage = computed(() => {
  if (!scenario.value && !cards.value.length) return ''
  const parts: string[] = []
  if (scenario.value) parts.push(`*（旁白）${scenario.value}*`)
  for (const card of cards.value) {
    if (card.greeting) parts.push(`**${card.name}**：${card.greeting}`)
  }
  return parts.join('\n\n')
})

const saveConfig = () => {
  localStorage.setItem(CONFIG_KEY, JSON.stringify({
    fileId: selectedFileId.value,
    names: [...selectedNames.value],
    chapterUntil: chapterUntil.value,
    started: started.value,
  }))
}

const loadConfig = (): { fileId: string | null; names: string[]; chapterUntil: number | null; started: boolean } | null => {
  try {
    const raw = localStorage.getItem(CONFIG_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object') return null
    return parsed
  } catch {
    return null
  }
}

const resetConfig = () => {
  localStorage.removeItem(CONFIG_KEY)
}

const loadNovels = async () => {
  try {
    const { data } = await listKB()
    novels.value = data || []
  } catch (error: any) {
    showToast(error?.message || '书库加载失败')
  }
}

const loadCharacters = async () => {
  if (!selectedFileId.value) {
    characters.value = []
    return
  }
  charsLoading.value = true
  charsError.value = ''
  try {
    const { data } = await listWorldCharacters(selectedFileId.value, chapterUntil.value || undefined)
    characters.value = data?.characters || []
  } catch (error: any) {
    charsError.value = error?.message || '人物提取失败，请重试或直接输入人物名'
    characters.value = []
  } finally {
    charsLoading.value = false
  }
}

const toggleCharacter = (name: string) => {
  const trimmed = name.trim()
  if (!trimmed) return
  const index = selectedNames.value.indexOf(trimmed)
  if (index >= 0) {
    selectedNames.value.splice(index, 1)
    return
  }
  if (selectedNames.value.length >= MAX_PERSONAS) {
    showToast(`最多同时选择 ${MAX_PERSONAS} 个人物`)
    return
  }
  selectedNames.value.push(trimmed)
}

const addCustom = () => {
  const name = customName.value.trim()
  if (!name) return
  if (!characters.value.includes(name)) characters.value.push(name)
  customName.value = ''
  if (!selectedNames.value.includes(name)) toggleCharacter(name)
}

const loadPastSessions = async () => {
  if (!selectedFileId.value) {
    pastSessions.value = []
    return
  }
  try {
    const { data } = await listSessions(selectedFileId.value)
    const rows = data || []
    // 展示该书全部角色扮演会话；人物组合与章节信息在每条上呈现，便于区分不同的历史线。
    pastSessions.value = rows.filter((row) => (row.personas || []).length > 0)
    console.debug(
      '[world] 历史对话加载：总数', rows.length,
      '含人物标记', pastSessions.value.length,
      rows.map((r) => ({ title: r.title, personas: r.personas, chapter_until: r.chapter_until })),
    )
  } catch (error: any) {
    pastSessions.value = []
    pastError.value = error?.message || '历史对话加载失败'
  }
}

// 点击历史对话：还原人物/章节配置，并显式写回该会话 id（防浏览器数据丢失后无法找回）。
const resumeSession = async (row: SessionItem) => {
  const names = row.personas || []
  if (!names.length || !row.file_id) return
  starting.value = true
  try {
    selectedFileId.value = row.file_id
    selectedNames.value = names
    chapterUntil.value = row.chapter_until ?? null
    localStorage.setItem(worldSessionKey(row.file_id, names, row.chapter_until ?? null), row.id)
    const { data } = await selectWorldCharacters(
      row.file_id, [...names], row.chapter_until || undefined,
    )
    cards.value = data?.cards || []
    scenario.value = data?.scenario || ''
    started.value = true
    saveConfig()
  } catch {
    showToast('恢复对话失败，请重试')
  } finally {
    starting.value = false
  }
}

const start = async () => {
  if (!canStart.value) return
  starting.value = true
  try {
    const { data } = await selectWorldCharacters(
      selectedFileId.value!, [...selectedNames.value], chapterUntil.value || undefined,
    )
    cards.value = data?.cards || []
    scenario.value = data?.scenario || ''
    // 每次点击「进入对话」都开启独立会话：清掉旧指向，首条消息发出时懒创建新会话；
    // 旧会话仍保留在下方历史列表中，可随时点回。
    localStorage.removeItem(sessionKey.value)
    started.value = true
    saveConfig()
    void loadPastSessions()
  } catch (error: any) {
    showToast(error?.message || '角色卡生成失败，请重试')
  } finally {
    starting.value = false
  }
}

const backToSetup = () => {
  started.value = false
  resetConfig()
  void loadPastSessions()
}

// 进入/刷新时恢复上次配置：对话态则重新拉角色卡（缓存命中，近乎瞬时）续上会话。
const restore = async () => {
  await loadNovels()
  const config = loadConfig()
  if (!config) return
  selectedFileId.value = config.fileId
  chapterUntil.value = config.chapterUntil ?? null
  if (config.started && config.names?.length && selectedFile.value?.status === 'indexed') {
    selectedNames.value = config.names
    starting.value = true
    try {
      const { data } = await selectWorldCharacters(
        selectedFileId.value!, [...selectedNames.value], chapterUntil.value || undefined,
      )
      cards.value = data?.cards || []
      scenario.value = data?.scenario || ''
      started.value = true
    } catch {
      started.value = false
      showToast('上次的小说世界恢复失败，请重新选择人物')
    } finally {
      starting.value = false
    }
  } else {
    started.value = false
    void loadCharacters()
  }
  // watch 注册在本函数调用之后，初始化时的赋值不会触发它——历史列表必须显式加载。
  void loadPastSessions()
}

// 初始加载书库并自动兜底到系统书 / 唯一一本书。
void restore().then(() => {
  if (!started.value && !selectedFileId.value) {
    const systemBook = novels.value.find((n) => n.is_system && n.status === 'indexed')
    const fallback = systemBook || indexedNovels.value[0]
    if (fallback) selectedFileId.value = fallback.id
  }
})

// 人物勾选变化也刷新历史列表，保证列表内容始终与后端一致。
watch([selectedFileId, selectedNames], () => {
  saveConfig()
  if (selectedFileId.value) {
    void loadCharacters()
    void loadPastSessions()
  }
})

watch(selectedNames, () => saveConfig())
</script>

<template>
  <div class="h-full flex flex-col">
    <!-- ===== 顶栏 ===== -->
    <header class="fixed top-0 inset-x-0 h-16 z-30 bg-white/95 backdrop-blur border-b border-black/[0.08] flex items-center justify-between gap-3 px-4 sm:px-6">
      <div class="flex items-center gap-2.5 min-w-0">
        <button
          class="btn-ghost w-9 h-9 !p-0 !rounded-lg"
          :aria-label="started ? '返回选人界面' : '返回问答'"
          :title="started ? '返回选人界面' : '返回问答'"
          @click="started ? backToSetup() : router.push('/chat')"
        >
          <Icon name="arrow-left" :size="16" />
        </button>
        <div class="shrink-0 w-9 h-9 rounded-lg bg-brand-600 flex items-center justify-center text-white shadow-card">
          <Icon name="book-open" :size="17" />
        </div>
        <div class="min-w-0">
          <p class="font-display text-[15px] font-bold tracking-wide text-ink leading-tight truncate">进入小说世界</p>
          <p class="text-[10px] uppercase tracking-[0.18em] text-ink-faint leading-tight">Step Into The Story</p>
        </div>
      </div>
      <div v-if="started" class="flex items-center gap-2 min-w-0">
        <span class="chip !max-w-[420px] truncate" :title="membersLabel">{{ membersLabel }}<template v-if="chapterUntil"> · 截至{{ chapterUntil }} 章</template></span>
        <button class="btn-ghost !rounded-lg px-3 py-1.5 text-xs shrink-0" @click="backToSetup">换人物</button>
      </div>
    </header>

    <!-- ===== 选人阶段 ===== -->
    <div v-if="!started" class="pt-20 flex-1 min-h-0 overflow-y-auto scroll-thin">
      <div class="mx-auto w-full max-w-2xl px-5 pb-10 space-y-5">
        <!-- 选小说 -->
        <section class="surface p-4">
          <p class="text-xs font-semibold text-ink mb-2.5 flex items-center gap-1.5">
            <Icon name="book-open" :size="13" class="text-brand-500" /> 选择小说
          </p>
          <select
            :value="selectedFileId || ''"
            class="w-full rounded-lg border border-black/[0.1] bg-white px-3 py-2 text-xs text-ink outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
            aria-label="选择小说"
            @change="selectedFileId = ($event.target as HTMLSelectElement).value || null"
          >
            <option value="" disabled>请选择已索引小说</option>
            <option v-for="n in indexedNovels" :key="n.id" :value="n.id">{{ n.filename }}</option>
          </select>
        </section>

        <!-- 剧情时间线 -->
        <section class="surface p-4">
          <p class="text-xs font-semibold text-ink mb-2.5 flex items-center gap-1.5">
            <Icon name="messages" :size="13" class="text-brand-500" /> 剧情进行到
          </p>
          <select
            :value="chapterUntil || ''"
            class="w-full rounded-lg border border-black/[0.1] bg-white px-3 py-2 text-xs text-ink outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
            aria-label="剧情截至章节"
            @change="chapterUntil = ($event.target as HTMLSelectElement).value ? Number(($event.target as HTMLSelectElement).value) : null"
          >
            <option value="">全书（人物知晓全部剧情）</option>
            <option v-for="ch in chapterOptions" :key="ch" :value="ch">第 {{ ch }} 章（人物只知此前剧情）</option>
          </select>
          <p class="mt-2 text-[11px] leading-4 text-ink-faint">
            人物只「记得」所选章节及之前的情节，之后的剧情对其而言尚未发生。
          </p>
        </section>

        <!-- 选人物 -->
        <section class="surface p-4">
          <div class="flex items-center justify-between mb-2.5">
            <p class="text-xs font-semibold text-ink flex items-center gap-1.5">
              <Icon name="user" :size="13" class="text-brand-500" /> 选择人物
              <span class="font-normal text-ink-faint">（可选 {{ selectedNames.length }}/{{ MAX_PERSONAS }}）</span>
            </p>
            <div class="flex items-center gap-1.5">
              <input
                v-model="customName"
                type="text"
                placeholder="输入人物名"
                class="w-32 rounded-lg border border-black/[0.1] bg-white px-2.5 py-1.5 text-xs outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                aria-label="自定义人物名"
                @keyup.enter="addCustom"
              />
              <button class="btn-ghost !rounded-lg px-2.5 py-1.5 text-xs" @click="addCustom">添加</button>
            </div>
          </div>

          <div v-if="charsLoading" class="py-6 text-center text-xs text-ink-faint" role="status">
            <Icon name="loader" :size="14" class="inline" /> 正在从小说中提取主要人物…
          </div>
          <div v-else-if="charsError" class="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2.5">
            <p class="text-xs leading-5 text-amber-800">{{ charsError }}</p>
            <button class="mt-1.5 text-xs font-semibold text-amber-700 underline underline-offset-2" @click="loadCharacters">重新提取</button>
          </div>
          <div v-else class="flex flex-wrap gap-2">
            <button
              v-for="name in characters"
              :key="name"
              class="rounded-full px-3.5 py-1.5 text-[13px] transition-all duration-200"
              :class="selectedNames.includes(name)
                ? 'bg-brand-600 text-white shadow-card'
                : 'bg-white text-ink ring-1 ring-black/[0.1] hover:ring-brand-300 hover:text-brand-700'"
              :aria-pressed="selectedNames.includes(name)"
              @click="toggleCharacter(name)"
            >
              {{ name }}
            </button>
            <span v-if="!characters.length" class="text-xs text-ink-faint px-1 py-1.5">
              暂无推荐人物，直接在上方输入想聊的人物名即可。
            </span>
          </div>
        </section>

        <!-- 历史对话 -->
        <section class="surface p-4">
          <p class="text-xs font-semibold text-ink mb-2.5 flex items-center gap-1.5">
            <Icon name="messages" :size="13" class="text-brand-500" /> 历史对话
          </p>
          <div v-if="pastError" class="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2.5">
            <p class="text-xs leading-5 text-rose-700">{{ pastError }}</p>
          </div>
          <div v-else-if="pastSessions.length" class="space-y-1.5">
            <div
              v-for="row in pastSessions"
              :key="row.id"
              class="group cursor-pointer rounded-lg bg-white ring-1 ring-black/[0.07] px-3 py-2.5 transition-all duration-200 hover:ring-brand-300 hover:shadow-card"
              role="button"
              tabindex="0"
              @click="resumeSession(row)"
              @keydown.enter="resumeSession(row)"
            >
              <div class="flex items-center justify-between gap-2">
                <p class="min-w-0 truncate text-[13px] font-medium text-ink">
                  {{ (row.personas || []).join(' · ') }}
                </p>
                <span class="shrink-0 text-[10px] text-ink-faint">{{ row.updated_at }}</span>
              </div>
              <p class="mt-0.5 truncate text-[11px] text-ink-faint">
                {{ row.chapter_until ? `剧情截至第 ${row.chapter_until} 章` : '全书剧情' }}
                <template v-if="row.title"> · {{ row.title }}</template>
              </p>
            </div>
          </div>
          <p v-else class="px-1 py-4 text-center text-xs leading-5 text-ink-faint">
            还没有角色扮演对话。选择人物并点击「进入对话」后，这里会记录每一次对话。
          </p>
        </section>

        <button
          class="btn-primary w-full !py-3 text-sm"
          :disabled="!canStart"
          @click="start"
        >
          <Icon v-if="starting" name="loader" :size="15" class="animate-spin" />
          {{ starting ? '正在生成角色卡与开场情景…' : '进入对话' }}
        </button>
        <p class="text-center text-[11px] leading-4 text-ink-faint">
          人物以小说设定与你交谈；其「记忆」在生成角色卡时已按所选章节固化，之后的剧情对其而言尚未发生。
        </p>
      </div>
    </div>

    <!-- ===== 对话阶段 ===== -->
    <div v-else class="pt-16 flex-1 flex min-h-0">
      <main class="flex-1 min-w-0 min-h-0 flex flex-col overflow-hidden">
        <ChatPanel
          :key="sessionKey"
          role="student"
          domain="novel"
          strategy="roleplay"
          :file-id="selectedFileId"
          :session-key="sessionKey"
          :assistant-names="assistantNames"
          :placeholder="`与${membersLabel}交谈…`"
          :title="`欢迎来到`"
          title-suffix="小说世界"
          :subtitle="cards.map((c) => `${c.name}：${c.persona}`).join('\n')"
          :notice="chapterUntil ? `剧情进行到第 ${chapterUntil} 章 · 人物只记得此前经历` : '人物知晓全书剧情'"
          :opening-message="openingMessage"
          :extra-payload="{
            personas: [...selectedNames],
            chapter_until: chapterUntil || undefined,
          }"
          :suggestions="cards.map((c) => `向${c.name}打个招呼`)"
        />
      </main>
    </div>
  </div>
</template>
