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
  strategy?: 'auto' | 'direct' | 'multi_expert' | 'react' | 'plan_execute' | 'roleplay'
  fileId?: string | null
  assistantNames?: string[]
  placeholder?: string
  extraPayload?: Record<string, any>
  openingMessage?: string
  citationPanel?: boolean
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
  assistantNames: () => [],
  placeholder: '询问人物关系、情节、时间线或章节位置…',
  extraPayload: () => ({}),
  openingMessage: '',
  citationPanel: false,
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
// 来源分数不是概率：邻居片段显示上下文补充；重排分已不对外展示（仅内部评测用），其余类型显示明确的分数语义。
const sourceScoreLabel = (source: SourceItem) => {
  if (source.neighbor || source.score_type === 'neighbor') return '上下文补充'
  if (source.score == null) return ''
  if (!source.score_type) return `${(source.score * 100).toFixed(0)}%`
  if (source.score_type === 'reranker') return ''
const value = source.score.toFixed(2)
  const labels: Record<Exclude<NonNullable<SourceItem['score_type']>, 'neighbor' | 'reranker'>, string> = {
    vector: '向量分',
    fts: '词法分',
    hybrid: '混合相关分',
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
  status?: 'running' | 'ok' | 'corrected' | 'invalid' | 'timeout' | 'fallback' | 'error' | 'fallback_generation' | 'empty_output'
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
  ragCalled?: boolean
  plan?: PlanStep[]
  reflection?: Reflection
  decisions?: AgentDecision[]
  validation?: any
  mainThinking?: ThinkingItem[]
  expertThinking?: Record<string, ThinkingItem[]>
  memoryContext?: MemoryContext
  outputPolicy?: OutputPolicy
}

type PlanStep = { step?: number; action?: string; purpose?: string }
type Reflection = { decision?: string; reason?: string; step?: number }
type AgentDecision = { action?: string; reason?: string; step?: number }
type ThinkingStream = 'main_agent' | 'multi_agent'
type ThinkingPhase = 'decide' | 'plan' | 'reflect' | 'reasoning'

type ThinkingItem = {
  id: string
  stream: ThinkingStream
  agent?: string
  label?: string
  phase: ThinkingPhase
  status: 'running' | 'completed' | 'error' | 'timeout' | 'cancelled' | 'corrected'
  text: string
  step?: number
  retry?: number
  reasoning_chars?: number
  truncated?: boolean
  latency_ms?: number
}

const messages = ref<ChatMessage[]>([])
const inputText = ref('')
const streaming = ref(false)
const memoryMode = ref<'auto' | 'off'>('auto')
// 思考过程展示开关：只控制已收到 reasoning 的渲染；原文是否下发由服务端开关决定。
const showThinking = ref(localStorage.getItem('novel_show_thinking') !== '0')
const toggleThinking = () => {
  showThinking.value = !showThinking.value
  localStorage.setItem('novel_show_thinking', showThinking.value ? '1' : '0')
}
const phaseLabel = (phase: string) =>
  (({ decide: '决策', plan: '计划', reflect: '反思', reasoning: '分析' } as Record<string, string>)[phase] || phase)
const thinkingStatus = (t: ThinkingItem) => {
  if (t.status === 'running') return '推理中…'
  const base = (({
    completed: '完成', corrected: '纠偏后完成', error: '失败', timeout: '超时', cancelled: '已取消',
  } as Record<string, string>)[t.status] || t.status)
  const parts = [base]
  if (t.reasoning_chars) parts.push(`${t.reasoning_chars} 字`)
  if (t.truncated) parts.push('已截断')
  return parts.join(' · ')
}
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

// 引用溯源右栏：最近一条带来源的助手消息。
const latestSources = computed(() => {
  for (let i = messages.value.length - 1; i >= 0; i--) {
    const m = messages.value[i]
    if (m.role === 'assistant' && m.sources?.length) return m.sources
  }
  return []
})

// 开场消息（如角色扮演的情景旁白+人物开场白）：消息区为空时自动呈现。
const maybeInsertOpening = () => {
  if (!messages.value.length && props.openingMessage) {
    messages.value.push({
      role: 'assistant',
      content: props.openingMessage,
      rendered: renderMd(props.openingMessage),
      sources: [],
      tools: [],
    })
  }
}

// 会话创建使用 Promise 复用，防止初始化期间重复创建会话。
// createIfMissing=false（挂载时）只加载已有会话：新建对话采用懒创建，
// 首条消息发出时才真正落库，避免产生空会话孤儿。
const ensureSession = (createIfMissing = false) => {
  if (sessionPromise && (!createIfMissing || sessionId.value)) return sessionPromise
  if (createIfMissing && !sessionId.value) sessionPromise = null
  sessionLoading.value = true
  sessionError.value = ''
  sessionPromise = (async () => {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      sessionId.value = saved
      try {
        const { data } = await getMessages(saved)
        messages.value = (data || []).map((m: any) => ({
          role: m.role,
          content: m.content,
          sources: m.sources || [],
          rendered: renderMd(m.content),
        }))
        // 空会话同样复用：落穿新建会在多会话场景持续制造孤儿。
        maybeInsertOpening()
        return
      } catch {
        localStorage.removeItem(STORAGE_KEY)
        sessionId.value = ''
      }
    }
    // 首次进入（懒创建）：开场消息先呈现，首条用户消息发出时才真正建会话。
    maybeInsertOpening()
    if (!createIfMissing) return
    if (!props.fileId) throw new Error('请先选择要咨询的小说')
    const { data } = await createSession(props.fileId)
    sessionId.value = data.id
    localStorage.setItem(STORAGE_KEY, data.id)
    maybeInsertOpening()
    emit('session-created', data.id)
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

// 新会话落地（自动创建或后端补发）时通知父组件刷新会话列表。
const emit = defineEmits<{
  (e: 'session-created', id: string): void
  (e: 'memory-updated'): void
}>()

// 发起一次聊天并将 route、专家任务、工具事件、来源和最终 token 写入同一条消息。
const send = async () => {
  const text = inputText.value.trim()
  if (!text || streaming.value) return
  try {
    await ensureSession(true)
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
        ...(props.extraPayload || {}),
      },
    {
      onSession: (id) => {
        sessionId.value = id
        localStorage.setItem(STORAGE_KEY, id)
        emit('session-created', id)
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
      onPlan: (payload) => {
        // react/plan_execute 的执行计划（含 plan_execute 中模型产出的计划）
        const steps = payload?.steps || []
        if (steps.length) messages.value[aiIndex].plan = steps
        scrollToBottom()
      },
      onReflection: (payload) => {
        // ReAct 循环评估：保留最新一次"继续执行/进入汇总"决策
        messages.value[aiIndex].reflection = payload
        scrollToBottom()
      },
      onAgentDecision: (payload) => {
        // Agent 决策时间线：直答 / 计划 / 调用工具，每次决策追加一条
        const m = messages.value[aiIndex]
        if (!m.decisions) m.decisions = []
        m.decisions.push(payload)
        scrollToBottom()
      },
      onThinkingStart: (t) => {
        const m = messages.value[aiIndex]
        const item: ThinkingItem = {
          id: t.id, stream: t.stream, agent: t.agent, label: t.label,
          phase: t.phase, status: 'running', text: '', step: t.step, retry: t.retry,
        }
        if (t.stream === 'multi_agent') {
          if (!m.expertThinking) m.expertThinking = {}
          const bucket = m.expertThinking[t.agent || ''] || []
          // 纠偏使用新 id，同 id 重复 start 视为重置
          const existing = bucket.findIndex((x) => x.id === t.id)
          if (existing >= 0) bucket[existing] = item
          else bucket.push(item)
          m.expertThinking[t.agent || ''] = bucket
        } else {
          if (!m.mainThinking) m.mainThinking = []
          const existing = m.mainThinking.findIndex((x) => x.id === t.id)
          if (existing >= 0) m.mainThinking[existing] = item
          else m.mainThinking.push(item)
        }
        scrollAgentOutput(t.id)
        scrollToBottom()
      },
      onThinkingToken: (t) => {
        const m = messages.value[aiIndex]
        const bucket = t.stream === 'multi_agent'
          ? m.expertThinking?.[t.agent || '']
          : m.mainThinking
        const item = bucket?.find((x) => x.id === t.id)
        if (item) {
          item.text += t.delta
          scrollAgentOutput(t.id)
          scrollToBottom()
        }
      },
      onThinkingEnd: (t) => {
        const m = messages.value[aiIndex]
        const bucket = t.stream === 'multi_agent'
          ? m.expertThinking?.[t.agent || '']
          : m.mainThinking
        const item = bucket?.find((x) => x.id === t.id)
        if (item) {
          item.status = t.status
          item.reasoning_chars = t.reasoning_chars
          item.truncated = t.truncated
          item.latency_ms = t.latency_ms
          scrollAgentOutput(t.id)
        }
      },
      onMemoryUpdated: () => emit('memory-updated'),
      onMeta: (payload) => {
        const m = messages.value[aiIndex]
        m.outputPolicy = payload?.output_policy || m.memoryContext?.output_policy || {}
        if (payload?.output_policy && m.memoryContext) m.memoryContext.output_policy = payload.output_policy
        // 检索与否的唯一事实来源：meta.rag_called（实际行为），路由建议不作展示依据
        if (payload && 'rag_called' in payload) m.ragCalled = !!payload.rag_called
        scrollToBottom()
      },
      onExpertTasks: (payload) => {
        const m = messages.value[aiIndex]
        m.expertTasks = payload?.tasks || {}
        m.dispatchMode = payload?.mode
        scrollToBottom()
      },
      onValidation: (payload) => {
        // 保存完整校验负载并在专家步骤上展示具体失败原因（缺失项/相似度），替代统一文案
        const m = messages.value[aiIndex]
        m.validation = payload
        const agentLabel: Record<string, string> = {
          character: '人物关系专家', plot: '情节发展专家',
          timeline: '时间线专家', locator: '章节定位专家',
        }
        const results = payload?.reports || {}
        for (const [agent, result] of Object.entries(results) as [string, any][]) {
          const step = m.tools?.find((item) => item.agent === agent)
          if (!step || result.contract_ok) continue
          const reasons: string[] = []
          if (result.missing_sections?.length) reasons.push(`缺少：${result.missing_sections.join('、')}`)
          for (const flag of result.similarity_flags || []) {
            reasons.push(`与${agentLabel[flag.agent] || flag.agent}重复度过高：${flag.score ?? '?'}`)
          }
          const reasonText = reasons.join('；') || '未通过契约校验'
          if (payload?.retry) {
            step.status = 'invalid'
            step.summary = `纠偏后仍未通过：${reasonText}`
          } else {
            step.summary = `报告需要纠偏：${reasonText}`
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
  <div class="relative flex h-full min-h-0">
    <!-- ===== 对话列 ===== -->
    <div class="flex flex-col flex-1 min-w-0 min-h-0">
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
              <div class="shrink-0 mt-0.5 w-7 h-7 rounded-lg bg-gradient-to-br from-brand-600 to-violet-600 flex items-center justify-center text-white shadow-sm" :title="assistantNames.join('、')">
                <span v-if="assistantNames.length" class="font-display text-[12px] font-bold leading-none">{{ assistantNames[0].slice(0, 1) }}</span>
                <Icon v-else name="bot" :size="15" />
              </div>
              <div class="min-w-0 flex-1">
                <!-- 检索徽标以 meta.rag_called（实际行为）为准：路由时刻无法预知 Agent 是否检索 -->
                <div v-if="m.ragCalled === false" class="mb-2 flex flex-wrap items-center gap-2 text-[11px] text-ink-faint">
                  <span class="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2.5 py-1 text-slate-600 ring-1 ring-slate-200">
                    <Icon name="check" :size="11" /> 本轮未检索原文 · 基于上下文与记忆回答
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

                <!-- 主 Agent 推理（实验能力：展示模型返回的 reasoning，非完整思维链） -->
                <div
                  v-if="showThinking && m.mainThinking?.length"
                  class="mb-2.5 rounded-lg bg-slate-50/80 px-3.5 py-2.5 ring-1 ring-slate-200/70"
                >
                  <div class="mb-1.5 flex items-center gap-2 text-[11px] font-semibold tracking-wide text-slate-600">
                    <Icon name="lightbulb" :size="12" />
                    主 Agent 推理
                    <span class="font-normal text-slate-400">· 实验能力</span>
                  </div>
                  <ul class="space-y-1">
                    <li v-for="t in m.mainThinking" :key="t.id">
                      <details :open="t.status === 'running'">
                        <summary class="cursor-pointer select-none text-[11px] text-slate-500">
                          {{ phaseLabel(t.phase) }}<span v-if="t.step"> · 第 {{ t.step }} 步</span>
                          <span class="ml-1 text-[10px]" :class="t.status === 'running' ? 'text-sky-600' : 'text-ink-faint'">{{ thinkingStatus(t) }}</span>
                        </summary>
                        <!-- 纯文本插值：reasoning 不走 Markdown 渲染器，禁止 v-html -->
                        <pre class="mt-1 max-h-60 overflow-y-auto whitespace-pre-wrap break-words rounded bg-white/80 px-2.5 py-1.5 text-left text-[11px] leading-4 text-slate-600">{{ t.text }}<span v-if="t.status === 'running'" class="stream-caret"></span></pre>
                      </details>
                    </li>
                  </ul>
                </div>
                <!-- 专家分析过程（multi_agent，按专家分组折叠） -->
                <div
                  v-if="showThinking && m.expertThinking && Object.keys(m.expertThinking).length"
                  class="mb-2.5 rounded-lg ring-1 ring-slate-200/70 bg-white/70"
                >
                  <div class="px-3.5 pt-2.5 pb-1 flex items-center gap-2 text-[11px] font-semibold tracking-wide text-slate-600">
                    <Icon name="layers" :size="12" />
                    专家分析过程
                    <span class="font-normal text-slate-400">· 实验能力</span>
                  </div>
                  <div class="space-y-0.5 px-2 pb-2">
                    <details
                      v-for="(items, agent) in m.expertThinking"
                      :key="agent"
                      class="rounded-md"
                      :open="items.some((t) => t.status === 'running')"
                    >
                      <summary class="cursor-pointer select-none px-2 py-1 text-[11px] text-ink-mute">
                        {{ items[0]?.label || agent }}
                        <span class="ml-1 text-[10px]" :class="items.some((t) => t.status === 'running') ? 'text-sky-600' : 'text-ink-faint'">
                          {{ items.some((t) => t.status === 'running') ? '分析中…' : thinkingStatus(items[items.length - 1]!) }}
                        </span>
                      </summary>
                      <div v-for="t in items" :key="t.id" class="px-2 pb-1">
                        <pre
                          v-if="t.text"
                          class="max-h-52 overflow-y-auto whitespace-pre-wrap break-words rounded bg-slate-50 px-2 py-1 text-left text-[11px] leading-4 text-slate-600"
                        >{{ t.text }}<span v-if="t.status === 'running'" class="stream-caret"></span></pre>
                        <p class="mt-0.5 text-[10px] text-ink-faint">{{ thinkingStatus(t) }}</p>
                      </div>
                    </details>
                  </div>
                </div>
                <!-- Agent 决策时间线 -->
                <div
                  v-if="m.decisions?.length"
                  class="mb-2.5 rounded-lg bg-sky-50/60 px-3.5 py-2.5 ring-1 ring-sky-100"
                >
                  <div class="mb-1.5 flex items-center gap-2 text-[11px] font-semibold tracking-wide text-sky-700">
                    <Icon name="lightbulb" :size="12" />
                    Agent 决策 · {{ m.decisions.length }}
                  </div>
                  <ul class="space-y-1">
                    <li
                      v-for="(d, di) in m.decisions"
                      :key="di"
                      class="flex items-start gap-1.5 text-[11px] leading-4 text-ink-mute"
                    >
                      <span
                        class="mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ring-1"
                        :class="d.action === 'tool_call'
                          ? 'bg-white text-sky-700 ring-sky-200'
                          : d.action === 'plan'
                            ? 'bg-violet-50 text-violet-700 ring-violet-200'
                            : 'bg-emerald-50 text-emerald-700 ring-emerald-200'"
                      >
                        {{ d.action === 'tool_call' ? '调用工具' : d.action === 'plan' ? '制定计划' : '直接回答' }}
                      </span>
                      <span class="min-w-0 flex-1 truncate">{{ d.reason }}<span v-if="d.step" class="text-ink-faint"> · 第 {{ d.step }} 步</span></span>
                    </li>
                  </ul>
                </div>
                <!-- 执行计划（仅模型真实制定时展示：静态执行框架不渲染，决策动态见「Agent 决策」） -->
                <div
                  v-if="m.plan?.some((s) => s.action === 'model_step')"
                  class="mb-2.5 rounded-lg bg-violet-50/60 px-3.5 py-3 ring-1 ring-violet-100"
                >
                  <div class="mb-2 flex items-center gap-2 text-[11px] font-semibold tracking-wide text-violet-700">
                    <Icon name="list" :size="12" />
                    执行计划
                  </div>
                  <ol class="space-y-1 text-[11px] leading-4 text-ink-mute">
                    <li
                      v-for="s in m.plan"
                      :key="s.step"
                      class="rounded-md bg-white/80 px-2.5 py-1.5"
                    >
                      <span class="font-semibold text-ink-soft">{{ s.step }}.</span>
                      {{ s.purpose }}
                    </li>
                  </ol>
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
                          <Icon v-if="t.status === 'running' || t.status === 'fallback_generation'" name="loader" :size="12" class="text-brand-500" />
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

                <!-- ReAct 循环评估：继续执行 / 进入汇总 -->
                <div v-if="m.reflection?.decision" class="mb-2 flex items-center gap-2 text-[11px] text-ink-faint">
                  <span
                    class="inline-flex items-center gap-1 rounded-full px-2.5 py-1 ring-1"
                    :class="m.reflection.decision === 'continue'
                      ? 'bg-amber-50 text-amber-700 ring-amber-200'
                      : 'bg-emerald-50 text-emerald-700 ring-emerald-200'"
                  >
                    <Icon :name="m.reflection.decision === 'continue' ? 'loader' : 'check'" :size="11" />
                    {{ m.reflection.decision === 'continue' ? `第 ${m.reflection.step ?? '?'} 步后继续执行` : '证据评估完成' }}
                    <span v-if="m.reflection.reason" class="font-normal">· {{ m.reflection.reason }}</span>
                  </span>
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

                <details v-if="m.sources?.length && m.outputPolicy?.show_citations !== false && !citationPanel" class="mt-2 text-xs text-ink-faint group/src">
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
    <div class="shrink-0 px-3 sm:px-6 pb-4 sm:pb-6 pt-2">
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
            :placeholder="placeholder"
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
          <button
            type="button"
            class="rounded-full px-2 py-0.5 ring-1 ring-black/[0.08] transition-colors hover:text-brand-600 hover:ring-brand-200"
            :class="showThinking ? 'bg-sky-50/70 text-sky-700 ring-sky-100' : 'bg-black/[0.03]'"
            @click="toggleThinking"
          >
            {{ showThinking ? '思考过程已开启' : '思考过程已关闭' }}
          </button>
        </div>
      </div>
    </div>
    </div>
    <!-- ===== 引用溯源右栏（citation-panel 模式） ===== -->
    <aside
      v-if="citationPanel"
      class="hidden xl:flex w-80 shrink-0 flex-col border-l border-black/[0.06] bg-white/70 p-4 overflow-y-auto scroll-thin"
      aria-label="引用溯源"
    >
      <div class="flex items-center justify-between mb-3">
        <p class="text-xs font-semibold text-ink flex items-center gap-1.5">
          <Icon name="book" :size="13" class="text-brand-500" /> 引用溯源
        </p>
        <span v-if="latestSources.length" class="text-[10px] text-ink-faint">{{ latestSources.length }} 条</span>
      </div>
      <div v-if="latestSources.length" class="space-y-2">
        <div
          v-for="(s, si) in latestSources"
          :key="s.id || si"
          class="rounded-lg bg-white px-3 py-2.5 ring-1"
          :class="si === 0 ? 'ring-amber-300' : 'ring-black/[0.07]'"
        >
          <p class="text-[11px] font-semibold text-ink truncate">
            {{ sourceChapterLabel(s) }}<template v-if="s.page != null"> · 第 {{ s.page + 1 }} 页</template>
            <template v-if="s.chunk_no != null"> · 片段 {{ String(s.chunk_no).padStart(4, '0') }}</template>
          </p>
          <p class="mt-1 line-clamp-3 text-[11px] leading-4 text-ink-mute">{{ s.snippet }}</p>
          <p class="mt-1 text-[10px] text-ink-faint">
            相似度 {{ (s.score ?? 0).toFixed(2) }}
            <template v-if="s.neighbor"> · 邻块扩展</template>
          </p>
        </div>
      </div>
      <p v-else class="py-6 text-center text-[11px] leading-5 text-ink-faint">
        最近的回答没有引用来源
      </p>
      <p class="mt-auto pt-3 text-[10px] leading-4 text-ink-faint">全文可溯源 · 交叉验证冲突时以原文为准</p>
    </aside>
  </div>
</template>

