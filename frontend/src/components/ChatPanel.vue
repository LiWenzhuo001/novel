<script setup lang="ts">
import { computed, ref, nextTick, onMounted, onUnmounted } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import Icon from './Icon.vue'
import { streamChat, createSession, getMessages, type SourceItem, type MemoryContext, type OutputPolicy } from '../api/client'

const props = withDefaults(defineProps<{
  role: string
  suggestions?: string[]
  title?: string
  titleSuffix?: string
  subtitle?: string
  notice?: string
  sessionKey?: string
  domain?: 'novel'
  strategy?: 'auto' | 'direct' | 'multi_expert'
  fileId?: string | null
}>(), {
  suggestions: () => [],
  title: '你好，我是你的',
  titleSuffix: '小说智读助手',
  subtitle: '上传小说文本后，可追问人物关系、情节因果、时间线和章节位置。',
  notice: '严格基于已索引原文回答 · 关键结论附章节引用',
  sessionKey: 'novel_rag_session_id',
  domain: 'novel',
  strategy: 'auto',
  fileId: null,
})
type ExpertTask = { label?: string; task?: string }

const sourceChapterLabel = (source: SourceItem) => {
  if (!source.chapter || source.chapter === '未分章') return '未识别章节'
  return source.chapter
}

const sourceLocationLabel = (source: SourceItem) => {
  if (source.page != null && source.source_type === 'pdf') return `第 ${source.page + 1} 页`
  if (source.source_type !== 'pdf' && source.char_start != null && source.char_end != null) {
    return `字符 ${source.char_start + 1}-${source.char_end}`
  }
  return ''
}
// 来源分数不是概率：邻居片段显示上下文补充，其余类型显示明确的分数语义。
const sourceScoreLabel = (source: SourceItem) => {
  if (source.neighbor || source.score_type === 'neighbor') return '上下文补充'
  if (source.score == null) return ''
  if (!source.score_type) return `${(source.score * 100).toFixed(0)}%`
  const value = source.score.toFixed(2)
  const labels: Record<Exclude<NonNullable<SourceItem['score_type']>, 'neighbor'>, string> = {
    vector: '向量分',
    fts: '词法分',
    hybrid: '混合相关分',
    reranker: '重排分',
  }
  return `${labels[source.score_type]} ${value}`
}
type ExpertTaskMap = Record<string, ExpertTask>

type ToolStep = {
  id?: string
  tool?: string
  agent?: string
  label?: string
  task?: string
  step?: number
  retry?: number
  status?: 'running' | 'ok' | 'corrected' | 'invalid' | 'timeout' | 'fallback' | 'error'
  summary?: string
  text?: string
  latency_ms?: number
  first_token_ms?: number
}

type ChatMessage = {
  role: string
  content: string
  sources?: SourceItem[]
  tools?: ToolStep[]
  expertTasks?: ExpertTaskMap
  dispatchMode?: string
  rendered?: string
  route?: any
  memoryContext?: MemoryContext
  outputPolicy?: OutputPolicy
}

const messages = ref<ChatMessage[]>([])
const inputText = ref('')
const streaming = ref(false)
const memoryMode = ref<'auto' | 'off'>('auto')
const sessionLoading = ref(true)
const sessionError = ref('')
const scroll = ref<HTMLElement>()
const sessionId = ref<string>('')
let controller: AbortController | null = null
let sessionPromise: Promise<void> | null = null
let scrollFrame: number | null = null
let renderTimer: ReturnType<typeof setTimeout> | null = null

const STORAGE_KEY = props.sessionKey
const canSend = computed(() => Boolean(inputText.value.trim()) && !streaming.value && !sessionLoading.value)

// 仅在用户已经接近底部时自动跟随，避免阅读历史消息时被强制滚动。
const scrollToBottom = () => {
  if (scrollFrame !== null) return
  scrollFrame = window.requestAnimationFrame(async () => {
    scrollFrame = null
    await nextTick()
    const el = scroll.value
    if (!el) return
    // 智能跟随：仅当本就贴近底部时才随内容滚到底；用户上翻查看历史时不再强制拉回。
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 32
    if (nearBottom) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  })
}

const scrollAgentOutput = async (id: string) => {
  await nextTick()
  const el = document.querySelector(`[data-agent-output="${CSS.escape(id)}"]`) as HTMLElement | null
  if (!el) return
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 32
  if (nearBottom) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
}

// 对模型 Markdown 做轻量预处理后交给 DOMPurify，兼顾表格显示和 XSS 安全。
const renderMd = (text: string) => {
  // 模型有时会把整段回答包进 ```markdown 代码围栏，导致表格以纯文本显示——整包则拆封
  let body = text.trim()
  const fence = body.match(/^```(?:markdown|md)\s*\n([\s\S]*?)\n?```$/)
  if (fence) body = fence[1]

  // 逐行预处理：表格行与代码块原样放行（行内正则修补会拆断单元格里的 **粗体**、--- 分隔线等），
  // 其余行只做两件安全的事：标题补空格、冒号词加粗。
  const out: string[] = []
  let inCode = false
  for (const line of body.split('\n')) {
    if (/^\s*```/.test(line)) {
      inCode = !inCode
      out.push(line)
      continue
    }
    if (inCode || /^\s*\|/.test(line)) {
      out.push(line)
      continue
    }
    let l = line
    l = l.replace(/(#{1,6})([^\s#])/g, '$1 $2')
    l = l.replace(/([：;；。?!？!])(\s*)([一-龥]{2,10})([：:])/g, '$1\n\n**$3**$4')
    out.push(l)
  }
  const t = out.join('\n').replace(/\n{3,}/g, '\n\n')
  return DOMPurify.sanitize(marked(t) as string, { USE_PROFILES: { html: true } })
}

const updateRendered = (index: number, immediate = false) => {
  const render = () => {
    const message = messages.value[index]
    if (message) message.rendered = renderMd(message.content)
    renderTimer = null
  }
  if (immediate) {
    if (renderTimer) clearTimeout(renderTimer)
    render()
    return
  }
  if (!renderTimer) renderTimer = setTimeout(render, 50)
}

// 会话创建使用 Promise 复用，防止初始化期间重复创建会话。
const ensureSession = () => {
  if (sessionPromise) return sessionPromise
  sessionLoading.value = true
  sessionError.value = ''
  sessionPromise = (async () => {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      sessionId.value = saved
      try {
        const { data } = await getMessages(saved)
        if (data?.length) {
          messages.value = data.map((m: any) => ({
            role: m.role,
            content: m.content,
            sources: m.sources || [],
            rendered: renderMd(m.content),
          }))
          return
        }
      } catch {
        localStorage.removeItem(STORAGE_KEY)
      }
    }
    if (!props.fileId) throw new Error('请先选择要咨询的小说')
    const { data } = await createSession(props.fileId)
    sessionId.value = data.id
    localStorage.setItem(STORAGE_KEY, data.id)
  })()
    .catch((error: any) => {
      sessionError.value = error?.message || '会话初始化失败，请重试'
      throw error
    })
    .finally(() => {
      sessionLoading.value = false
    })
  return sessionPromise
}

const retrySession = () => {
  sessionPromise = null
  void ensureSession().catch(() => undefined)
}

// 发起一次聊天并将 route、专家任务、工具事件、来源和最终 token 写入同一条消息。
const send = async () => {
  const text = inputText.value.trim()
  if (!text || streaming.value) return
  try {
    await ensureSession()
  } catch {
    return
  }
  messages.value.push({ role: 'user', content: text })
  const ai: ChatMessage = { role: 'assistant', content: '', rendered: '', sources: [], tools: [] }
  messages.value.push(ai)
  // 必须用响应式代理（messages.value[idx]）来改内容，直接改原始 ai 不会触发重渲染
  const aiIndex = messages.value.length - 1
  inputText.value = ''
  streaming.value = true
  scrollToBottom()

  const history = messages.value
    .slice(0, -2)
    .map((m) => ({ role: m.role, content: m.content }))
  controller = streamChat(
    {
      message: text,
      role: props.role,
      domain: props.domain,
      strategy: props.strategy,
      memory_mode: memoryMode.value,
      history,
      session_id: sessionId.value,
      file_id: props.fileId || undefined,
    },
    {
      onSession: (id) => {
        sessionId.value = id
        localStorage.setItem(STORAGE_KEY, id)
      },
      onMemoryContext: (payload) => {
        messages.value[aiIndex].memoryContext = {
          summary: payload?.summary || '',
          summary_id: payload?.summary_id,
          memories: payload?.memories || [],
          output_policy: payload?.output_policy || {},
        }
        scrollToBottom()
      },
      onRoute: (payload) => {
        const m = messages.value[aiIndex]
        m.route = payload
        scrollToBottom()
      },
      onMeta: (payload) => {
        const m = messages.value[aiIndex]
        m.outputPolicy = payload?.output_policy || m.memoryContext?.output_policy || {}
        if (payload?.output_policy && m.memoryContext) m.memoryContext.output_policy = payload.output_policy
        scrollToBottom()
      },
      onExpertTasks: (payload) => {
        const m = messages.value[aiIndex]
        m.expertTasks = payload?.tasks || {}
        m.dispatchMode = payload?.mode
        scrollToBottom()
      },
      onValidation: (payload) => {
        const m = messages.value[aiIndex]
        const results = payload?.reports || {}
        for (const [agent, result] of Object.entries(results) as [string, any][]) {
          const step = m.tools?.find((item) => item.agent === agent)
          if (!step || result.contract_ok) continue
          if (payload?.retry) {
            step.status = 'invalid'
            step.summary = '纠偏后仍未通过校验'
          } else {
            step.summary = '报告需要纠偏'
          }
        }
      },
      onSources: (s) => (messages.value[aiIndex].sources = s),
      onToken: (t) => {
        messages.value[aiIndex].content += t
        updateRendered(aiIndex)
        scrollToBottom()
      },
      onTokenReplace: (t) => {
        // 后端输出护栏净化稿（去引用/截断引语）：整体覆盖已流式渲染的内容
        messages.value[aiIndex].content = t
        updateRendered(aiIndex)
        scrollToBottom()
      },
      onToolStart: (t) => {
        const m = messages.value[aiIndex]
        if (!m.tools) m.tools = []
        const existing = m.tools.find((step) => step.id === t.id)
        if (existing) {
          if (t.reset) existing.text = ''
          Object.assign(existing, t, {
            status: 'running',
            summary: t.retry ? '正在纠偏重试' : existing.summary,
          })
        } else {
          m.tools.push({ ...t, label: t.label || t.tool, status: 'running', text: '' })
        }
        scrollToBottom()
      },
      onToolToken: (t) => {
        const m = messages.value[aiIndex]
        const step = m.tools?.find((item) => item.id === t.id)
        if (step) {
          step.text = (step.text || '') + (t.delta || '')
          scrollAgentOutput(t.id)
          scrollToBottom()
        }
      },
      onToolEnd: (t) => {
        const m = messages.value[aiIndex]
        const step = m.tools?.find((s: any) => s.id === t.id)
        if (step) {
          step.status = t.status
          step.summary = t.summary
          step.latency_ms = t.latency_ms
          step.first_token_ms = t.first_token_ms
        }
        scrollAgentOutput(t.id)
        scrollToBottom()
      },
      onDone: () => {
        if (!messages.value[aiIndex].content && !messages.value[aiIndex].tools?.length) {
          messages.value[aiIndex].content = '未生成回答，请重试。'
        }
        updateRendered(aiIndex, true)
        streaming.value = false
        controller = null
      },
      onError: (e: any) => {
        const msg = e?.message || '未知错误'
        messages.value[aiIndex].content += `\n[出错了：${msg}]`
        updateRendered(aiIndex, true)
        streaming.value = false
        controller = null
      },
    },
  )
}

const stop = () => {
  controller?.abort()
  const last = messages.value[messages.value.length - 1]
  if (last?.role === 'assistant' && !last.content && !last.tools?.length) {
    last.content = '已停止生成。'
  }
  if (last?.role === 'assistant') updateRendered(messages.value.length - 1, true)
  streaming.value = false
  controller = null
}

// 供外部（侧栏快捷问题）触发提问
const ask = (q: string) => {
  inputText.value = q
  send()
}

defineExpose({ send: ask })

onMounted(() => void ensureSession().catch(() => undefined))
// 组件销毁时主动取消流式请求，避免后台继续推送已不可见的消息。
onUnmounted(() => {
  controller?.abort()
  if (scrollFrame !== null) window.cancelAnimationFrame(scrollFrame)
  if (renderTimer) clearTimeout(renderTimer)
})
</script>

<template>
  <div class="relative flex flex-col h-full">
    <!-- ===== 消息流 ===== -->
    <div ref="scroll" class="flex-1 overflow-y-auto scroll-thin" :aria-busy="streaming" aria-live="polite">
      <div class="max-w-3xl mx-auto px-5 sm:px-6 py-8">
        <!-- 空状态 Hero -->
        <div v-if="!messages.length" class="min-h-[58vh] flex flex-col items-center justify-center text-center animate-fadeIn">
          <div class="w-14 h-14 rounded-xl bg-brand-600 flex items-center justify-center text-white shadow-card">
            <Icon name="bot" :size="27" />
          </div>
          <h2 class="mt-6 font-display text-2xl sm:text-3xl font-bold text-ink">
            {{ title }}<span class="text-gradient whitespace-nowrap">{{ titleSuffix }}</span>
          </h2>
          <p class="mt-3 text-sm leading-6 text-ink-mute max-w-md whitespace-pre-line">
            {{ subtitle }}
          </p>

          <div v-if="sessionLoading" class="mt-6 inline-flex items-center gap-2 text-sm text-ink-mute" role="status">
            <Icon name="loader" :size="15" /> 正在准备会话
          </div>
          <div v-else-if="sessionError" class="mt-6 max-w-md rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-left" role="alert">
            <p class="text-sm font-medium text-rose-700">无法连接到会话</p>
            <p class="mt-1 text-xs leading-5 text-rose-600">{{ sessionError }}</p>
            <button class="mt-2 text-xs font-semibold text-rose-700 underline underline-offset-2" @click="retrySession">重新连接</button>
          </div>

          <div v-if="suggestions.length && !sessionError" class="mt-7 grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-xl">
            <button
              v-for="(q, i) in suggestions"
              :key="i"
              @click="ask(q)"
              class="group flex items-center gap-3 rounded-lg bg-white ring-1 ring-black/[0.08] px-4 py-3 text-left text-[13px] text-ink-soft transition-colors duration-200 hover:bg-brand-50 hover:ring-brand-300 hover:text-brand-700 cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
              :disabled="sessionLoading"
            >
              <span class="shrink-0 w-6 h-6 rounded-lg bg-brand-50 text-brand-600 flex items-center justify-center transition-colors group-hover:bg-brand-600 group-hover:text-white">
                <Icon name="messages" :size="13" />
              </span>
              <span class="leading-5">{{ q }}</span>
            </button>
          </div>
        </div>

        <!-- 消息列表 -->
        <div class="space-y-6">
          <div
            v-for="(m, i) in messages"
            :key="i"
            class="flex animate-fadeIn"
            :class="m.role === 'user' ? 'justify-end' : 'justify-start'"
          >
            <!-- 用户消息：墨色气泡 -->
            <div
              v-if="m.role === 'user'"
              class="max-w-[88%] px-4 py-2.5 rounded-2xl rounded-br-md bg-ink text-white text-[14px] leading-6 shadow-card sm:max-w-[80%]"
            >
              {{ m.content }}
            </div>

            <!-- AI 消息：编辑式排版 -->
            <div v-else class="max-w-full flex gap-2.5 sm:max-w-[90%] sm:gap-3">
              <div class="shrink-0 mt-0.5 w-7 h-7 rounded-lg bg-gradient-to-br from-brand-600 to-violet-600 flex items-center justify-center text-white shadow-sm">
                <Icon name="bot" :size="15" />
              </div>
              <div class="min-w-0 flex-1">
                <div v-if="m.route?.retrieval_skipped" class="mb-2 flex flex-wrap items-center gap-2 text-[11px] text-ink-faint">
                  <span class="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-1 text-emerald-700 ring-1 ring-emerald-200">
                    <Icon name="check" :size="11" /> 本轮无需检索小说原文
                  </span>
                  <span v-if="m.memoryContext?.memories?.length" class="inline-flex items-center gap-1 rounded-full bg-brand-50 px-2.5 py-1 text-brand-700 ring-1 ring-brand-100">
                    <Icon name="sparkles" :size="11" /> 使用 {{ m.memoryContext.memories.length }} 条记忆
                  </span>
                </div>
                <div v-else-if="m.memoryContext?.memories?.length" class="mb-2 text-[11px] text-brand-600">
                  <Icon name="sparkles" :size="11" class="inline" /> 使用 {{ m.memoryContext.memories.length }} 条记忆辅助回答
                </div>
                <div
                  v-if="m.expertTasks && Object.keys(m.expertTasks).length"
                  class="mb-2.5 rounded-lg bg-brand-50/60 px-3.5 py-3 ring-1 ring-brand-100"
                >
                  <div class="mb-2 flex items-center gap-2 text-[11px] font-semibold tracking-wide text-brand-700">
                    <Icon name="sparkles" :size="12" />
                    本轮专家任务分派
                    <span v-if="m.dispatchMode" class="font-normal text-brand-400">· {{ m.dispatchMode }}</span>
                  </div>
                  <div class="grid gap-1.5 sm:grid-cols-2">
                    <div
                      v-for="(task, key) in m.expertTasks"
                      :key="key"
                      class="rounded-md bg-white/80 px-2.5 py-2 text-[11px] leading-4 text-ink-mute ring-1 ring-black/[0.05]"
                    >
                      <span class="font-semibold text-ink-soft">{{ task.label }}</span>
                      <span class="ml-1">{{ task.task }}</span>
                    </div>
                  </div>
                </div>

                <!-- 工具调用过程 -->
                <div
                  v-if="m.tools?.length"
                  class="mb-2.5 overflow-hidden rounded-lg ring-1 ring-black/[0.08] bg-white"
                >
                  <div class="flex items-center gap-2 border-b border-black/[0.05] px-3.5 py-2 text-[11px] font-semibold tracking-wide text-ink-mute">
                    <Icon name="wrench" :size="12" class="text-brand-500" />
                    调用过程 · {{ m.tools.length }}
                  </div>
                  <ul class="space-y-1.5 px-3.5 py-2.5">
                    <li
                      v-for="(t, ti) in m.tools"
                      :key="ti"
                      class="text-[12.5px] leading-5"
                    >
                      <div class="flex items-center gap-2">
                        <span class="shrink-0 w-4 h-4 flex items-center justify-center">
                          <Icon v-if="t.status === 'running'" name="loader" :size="12" class="text-brand-500" />
                          <Icon v-else-if="t.status === 'ok' || t.status === 'corrected'" name="check" :size="12" class="text-emerald-500" />
                          <Icon v-else-if="t.status === 'fallback'" name="check" :size="12" class="text-amber-500" />
                          <Icon v-else name="x" :size="12" class="text-rose-500" />
                        </span>
                        <span class="text-ink-soft">{{ t.label || t.tool }}</span>
                        <span v-if="t.step" class="text-ink-faint">· 第 {{ t.step }} 步</span>
                        <span v-if="t.summary && t.status !== 'running'" class="truncate text-ink-faint">· {{ t.summary }}</span>
                      </div>
                      <p v-if="t.task" class="ml-6 mt-0.5 text-[11px] leading-4 text-ink-faint">{{ t.task }}</p>
                      <div
                        v-if="t.text && m.outputPolicy?.show_agent_details !== false"
                        :data-agent-output="t.id"
                        class="agent-output-md chat-md mt-1.5 mb-0.5 text-[12.5px] leading-5 text-ink-soft"
                      >
                        <div class="agent-output-block" v-html="renderMd(t.text)"></div>
                        <span v-if="t.status === 'running'" class="stream-caret"></span>
                      </div>
</li>
                  </ul>
                </div>

                <template v-if="m.content">
                  <div class="chat-md text-[14px] text-ink-soft" v-html="m.rendered || renderMd(m.content)"></div>
                  <span v-if="streaming && i === messages.length - 1" class="stream-caret"></span>
                </template>
                <div v-else-if="!m.tools?.length" class="flex items-center gap-1.5 py-1.5">
                  <span class="thinking-dot w-1.5 h-1.5 rounded-full bg-brand-500"></span>
                  <span class="thinking-dot w-1.5 h-1.5 rounded-full bg-brand-500"></span>
                  <span class="thinking-dot w-1.5 h-1.5 rounded-full bg-brand-500"></span>
                  <span class="text-xs text-ink-faint ml-1.5 tracking-wide">正在思考</span>
                </div>

                <details v-if="m.sources?.length && m.outputPolicy?.show_citations !== false" class="mt-2 text-xs text-ink-faint group/src">
                  <summary class="cursor-pointer select-none flex items-center gap-1.5 w-fit rounded-full px-2.5 py-1 ring-1 ring-black/[0.06] bg-white/60 transition-colors hover:text-brand-600 hover:ring-brand-200">
                    <Icon name="chevron-down" :size="12" /> 引用来源 · {{ m.sources.length }}
                  </summary>
                  <div class="mt-2 space-y-1.5">
                    <div
                      v-for="(s, si) in m.sources"
                      :key="si"
                      class="rounded-lg bg-white ring-1 ring-black/[0.08] px-3 py-2"
                    >
                      <div class="flex items-center justify-between gap-2">
                        <span class="font-medium text-ink-mute flex items-center gap-1.5">
                          <Icon name="file-text" :size="12" class="text-brand-500" />
                          <span class="truncate">{{ s.source || '未知来源' }}</span>
                          <span class="shrink-0 text-ink-faint">· {{ sourceChapterLabel(s) }}</span>
                          <span v-if="sourceLocationLabel(s)" class="shrink-0 text-ink-faint">· {{ sourceLocationLabel(s) }}</span>
                          <span v-if="s.chunk_no != null" class="shrink-0 text-ink-faint">· 片段 {{ s.chunk_no }}</span>
                        </span>
                        <span
                          v-if="sourceScoreLabel(s)"
                          class="shrink-0 chip !py-0.5 !px-2 !text-[10px]"
                          :class="s.neighbor || s.score_type === 'neighbor' ? '!text-ink-faint !bg-black/[0.03]' : ''"
                        >{{ sourceScoreLabel(s) }}</span>
                      </div>
                      <p v-if="m.outputPolicy?.show_source_text === true" class="text-ink-faint mt-1 line-clamp-2 leading-5">{{ s.snippet }}</p>
                    </div>
                  </div>
                </details>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== 悬浮输入舱 ===== -->
    <div class="shrink-0 px-3 sm:px-6 pb-3 sm:pb-5 pt-2">
      <div class="max-w-3xl mx-auto">
        <div
          class="flex items-end gap-2 rounded-xl bg-white ring-1 ring-black/[0.09] shadow-card pl-4 pr-2 py-2 transition-all duration-200 focus-within:ring-2 focus-within:ring-brand-500/60"
        >
          <label for="chat-input" class="sr-only">输入小说问题</label>
          <input
            id="chat-input"
            v-model="inputText"
            @keyup.enter="send"
            type="text"
            :placeholder="'询问人物关系、情节、时间线或章节位置…'"
            class="flex-1 bg-transparent outline-none text-sm text-ink placeholder:text-ink-faint py-2"
          />
          <button
            v-if="!streaming"
            @click="send"
            class="btn-primary !rounded-lg w-10 h-10 !p-0 shrink-0"
            :disabled="!canSend"
            aria-label="发送"
          >
            <Icon name="send" :size="16" />
          </button>
          <button
            v-else
            @click="stop"
            class="shrink-0 w-10 h-10 rounded-lg bg-ink text-white flex items-center justify-center shadow-card transition-colors hover:bg-black cursor-pointer"
            aria-label="停止"
          >
            <span class="block w-3 h-3 rounded-[3px] bg-white"></span>
          </button>
        </div>
        <div class="mt-2 flex flex-wrap items-center justify-center gap-2 text-[11px] tracking-wide text-ink-faint">
          <span>{{ notice }}</span>
          <button
            type="button"
            class="rounded-full px-2 py-0.5 ring-1 ring-black/[0.08] transition-colors hover:text-brand-600 hover:ring-brand-200"
            :class="memoryMode === 'off' ? 'bg-black/[0.03]' : 'bg-brand-50/70 text-brand-700 ring-brand-100'"
            @click="memoryMode = memoryMode === 'auto' ? 'off' : 'auto'"
          >
            {{ memoryMode === 'auto' ? '自动记忆已开启' : '自动记忆已关闭' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
