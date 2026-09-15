// 纯 Web API 客户端（fetch 实现）
// - 普通请求 / 文件上传用 fetch + FormData
// - 流式对话用 fetch + ReadableStream 解析 SSE
// 所有请求走相对路径 /api，由 vite dev proxy 或 nginx 转发到后端。

const BASE = '/api'
const TOKEN_KEY = 'novel_rag_api_token'

export const getApiToken = () => localStorage.getItem(TOKEN_KEY) || ''
export const setApiToken = (token: string) => {
  const value = token.trim()
  if (value) localStorage.setItem(TOKEN_KEY, value)
  else localStorage.removeItem(TOKEN_KEY)
}
export const clearApiToken = () => localStorage.removeItem(TOKEN_KEY)

const redirectToLogin = () => {
  clearApiToken()
  const current = window.location.pathname + window.location.search
  if (!window.location.pathname.startsWith('/login')) {
    window.location.href = `/login?redirect=${encodeURIComponent(current)}`
  }
}

const authHeaders = (extra: Record<string, string> = {}) => {
  const token = getApiToken()
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra
}

// 参数校验（422）错误的中文化：FastAPI/pydantic 的 detail 是数组，每条带
// loc（字段路径）/type（错误类型）/ctx（上下文参数）/msg（英文描述）。
// 只在这里做字段名与错误类型的中文映射，不改后端错误契约。
const FIELD_LABELS: Record<string, string> = {
  username: '用户名',
  password: '密码',
  email: '邮箱',
  display_name: '显示名称',
  message: '消息内容',
  file_id: '小说 ID',
  session_id: '会话 ID',
  history: '对话历史',
  personas: '在场人物',
  chapter_until: '剧情截至章节',
  max_steps: '执行步数',
  strategy: '执行策略',
  interaction_mode: '交互模式',
  memory_mode: '记忆开关',
  names: '人物名列表',
  title: '标题',
}

const describeValidationError = (e: any): string => {
  const loc: string[] = Array.isArray(e?.loc) ? e.loc.map((p: unknown) => String(p)) : []
  const fieldKey = loc.find((p: string) => p !== 'body') || ''
  const field = FIELD_LABELS[fieldKey] || fieldKey
  const type = String(e?.type || '')
  const ctx = e?.ctx || {}
  const msg = String(e?.msg || '')
  switch (type) {
    case 'string_too_short':
      return ctx?.min_length != null
        ? `${field}至少 ${ctx.min_length} 个字符`
        : `${field}过短`
    case 'string_too_long':
      return ctx?.max_length != null
        ? `${field}最多 ${ctx.max_length} 个字符`
        : `${field}过长`
    case 'missing':
      return field ? `请填写${field}` : '请填写必填项'
    case 'value_error':
      // 后端自定义校验器的 msg 本身是中文（如"interaction_mode=roleplay 需要 1~3 个 personas"）
      return msg.replace(/^Value error,\s*/, '')
    case 'greater_than_equal':
      return `${field}不能小于 ${ctx?.ge ?? '下限'}`
    case 'less_than_equal':
      return `${field}不能大于 ${ctx?.le ?? '上限'}`
    default:
      return field ? `${field}：${msg || '取值不合法'}` : (msg || '参数不合法')
  }
}

const readErrorDetail = async (res: Response, fallback = `HTTP ${res.status}`) => {
  try {
    const payload = await res.json()
    const detail = payload?.detail
    // FastAPI 422 的 detail 是数组（每个不合法字段一条错误对象），拼成人话再展示，
    // 否则会渲染成 [object Object]。
    if (Array.isArray(detail)) {
      return detail.map(describeValidationError).filter(Boolean).join('；') || fallback
    }
    return detail || fallback
  } catch {
    return fallback
  }
}

const ensureSuccess = async (res: Response, fallback?: string) => {
  if (res.ok) return
  const detail = await readErrorDetail(res, fallback)
  if (res.status === 401) redirectToLogin()
  throw new Error(detail)
}

export interface ApiResponse<T = any> {
  code: number
  data: T
  message?: string
}

export interface UserInfo {
  id: string
  username: string
  email?: string | null
  display_name: string
  is_admin: boolean
}

export interface AuthData {
  access_token: string
  token_type: string
  user: UserInfo
}

// 统一处理 JSON 请求、认证头、HTTP 错误和 401 跳转。
async function request<T = any>(
  url: string,
  method: 'GET' | 'POST' | 'DELETE' | 'PUT' | 'PATCH' = 'GET',
  data?: any,
): Promise<ApiResponse<T>> {
  const res = await fetch(BASE + url, {
    method,
    headers: authHeaders(data !== undefined ? { 'Content-Type': 'application/json' } : {}),
    body: data !== undefined ? JSON.stringify(data) : undefined,
  })
  await ensureSuccess(res)
  return res.json()
}

// ---------- 认证 ----------
export const register = (payload: { username: string; password: string; email?: string; display_name?: string }) =>
  request<AuthData>('/auth/register', 'POST', payload)

export const login = (payload: { username: string; password: string }) =>
  request<AuthData>('/auth/login', 'POST', payload)

export const getMe = () => request<UserInfo>('/auth/me')

// ---------- 知识库 ----------
export type SourceScoreType = 'vector' | 'fts' | 'hybrid' | 'reranker' | 'neighbor'

// RAG 来源的可选定位和分数元数据；旧会话缺少新增字段时仍可正常渲染。
export interface SourceItem {
  id?: string
  source?: string
  source_type?: string
  chapter?: string
  chapter_no?: number | null
  page?: number | null
  chunk_no?: number | null
  char_start?: number | null
  char_end?: number | null
  score?: number | null
  score_type?: SourceScoreType
  neighbor?: boolean
  retrieval_rank?: number | null
  vector_score?: number | null
  fts_score?: number | null
  rrf_score?: number | null
  reranked?: boolean
  snippet?: string
}

// 知识库文件及其索引、章节解析和进度状态。
export interface KBFileInfo {
  id: string
  filename: string
  filetype: string
  size: number
  chunks: number
  domain?: 'novel'
  is_system?: boolean
  status?: 'pending' | 'indexing' | 'indexed' | 'failed'
  index_stage?:
    | 'pending'
    | 'loading'
    | 'parsing'
    | 'analyzing_chapters'
    | 'building_embeddings'
    | 'switching'
    | 'completed'
    | 'failed'
    | null
  index_progress?: number | null
  index_message?: string | null
  error?: string | null
  index_version?: string | null
  embedding_model?: string | null
  embed_dim?: number | null
  chunk_size?: number | null
  chunk_overlap?: number | null
  indexed_at?: string | null
  chapter_count?: number | null
  unassigned_chunk_count?: number | null
  chapter_parse_status?: 'ok' | 'unrecognized' | null
  chapter_parser_mode?: 'strict' | 'inline_fallback' | 'llm_assisted' | 'none' | null
  chapter_parser_version?: string | null
  chapter_index_stale?: boolean
  detected_encoding?: string | null
  index_warning?: string | null
  chapter_rule_confidence?: number | null
  chapter_rule_validated?: boolean | null
  chapter_detection_model?: string | null
  chapter_detection_error?: string | null
  created_at: string
}

// 文件上传使用 FormData，索引本身由后端后台任务异步完成。
export const uploadKB = async (file: File): Promise<ApiResponse<KBFileInfo>> => {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch(`${BASE}/kb/upload`, { method: 'POST', headers: authHeaders(), body: fd })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      detail = (await res.json())?.detail || detail
    } catch {
      // 非 JSON 错误体：保留 HTTP 状态码兜底
    }
    if (res.status === 401) redirectToLogin()
    throw new Error(detail)
  }
  return res.json()
}

export const listKB = () => request<KBFileInfo[]>('/kb/files')
export const deleteKB = (id: string) => request(`/kb/files/${id}`, 'DELETE')
export const reindexKB = (id: string) =>
  request<{ id: string; status: 'indexing'; message: string }>(`/kb/files/${id}/reindex`, 'POST')

// ---------- 会话 ----------
export const createSession = (fileId: string) => request(`/chat/sessions?domain=novel&file_id=${encodeURIComponent(fileId)}`, 'POST')
export interface SessionItem {
  id: string
  title: string
  role: string
  domain?: string
  file_id?: string | null
  personas?: string[]
  chapter_until?: number | null
  updated_at: string
}
export const listSessions = (fileId?: string) => {
  const query = fileId ? `?file_id=${encodeURIComponent(fileId)}` : ''
  return request<SessionItem[]>(`/chat/sessions${query}`)
}
export const deleteSession = (sessionId: string) => request(`/chat/sessions/${encodeURIComponent(sessionId)}`, 'DELETE')
export const renameSession = (sessionId: string, title: string) =>
  request<{ id: string; title: string }>(`/chat/sessions/${encodeURIComponent(sessionId)}`, 'PATCH', { title })

// ---------- 小说世界（角色扮演） ----------
export interface CharacterCard {
  name: string
  persona: string
  style: string
  background: string
  greeting: string
}
export const listWorldCharacters = (fileId: string, chapterUntil?: number) => {
  const query = new URLSearchParams({ file_id: fileId })
  if (chapterUntil) query.set('chapter_until', String(chapterUntil))
  return request<{ characters: string[]; cached: boolean }>(`/chat/world/characters?${query.toString()}`)
}
export const selectWorldCharacters = (fileId: string, names: string[], chapterUntil?: number) =>
  request<{ cards: CharacterCard[]; scenario: string }>('/chat/world/characters/select', 'POST', {
    file_id: fileId, names, chapter_until: chapterUntil,
  })
export const getMessages = (sessionId: string) =>
  request<{ id: number; role: string; content: string; sources: SourceItem[]; status?: string }[]>(
    `/chat/sessions/${sessionId}/messages`,
  )

export interface OutputPolicy {
  summary_only?: boolean
  show_source_text?: boolean
  allow_direct_quotes?: boolean
  show_citations?: boolean
  citation_style?: 'chapter_only' | 'hidden' | 'normal' | string
  show_agent_details?: boolean
}

export interface MemoryItem {
  id: string
  memory_type: 'user_preference' | 'novel_fact' | 'session_fact' | string
  content: string
  importance: number
  session_id?: string | null
  file_id?: string | null
  source_message_id?: number | null
  expires_at?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export interface MemoryContext {
  summary: string
  summary_id?: string | null
  memories: MemoryItem[]
  output_policy?: OutputPolicy
}

export const listMemories = (sessionId?: string, fileId?: string | null) => {
  const params = new URLSearchParams()
  if (sessionId) params.set('session_id', sessionId)
  if (fileId) params.set('file_id', fileId)
  return request<MemoryItem[]>(`/memories${params.toString() ? `?${params.toString()}` : ''}`)
}

export const getMemoryContext = (sessionId: string, fileId?: string | null) => {
  const query = fileId ? `?file_id=${encodeURIComponent(fileId)}` : ''
  return request<MemoryContext>(`/chat/sessions/${sessionId}/memory-context${query}`)
}

export const deleteMemory = (memoryId: string) => request(`/memories/${encodeURIComponent(memoryId)}`, 'DELETE')

// ---------- 流式对话（H5 浏览器下用 fetch 解析 SSE）----------
// 思考流事件负载：reasoning 原文仅在服务端 EXPOSE_RAW_REASONING=true 时下发；
// 前端"显示思考过程"开关只控制已收到内容是否渲染。
export type ThinkingStream = 'main_agent' | 'multi_agent'
export type ThinkingPhase = 'decide' | 'plan' | 'reflect' | 'reasoning'

export interface ThinkingStart {
  stream: ThinkingStream
  id: string
  agent?: string
  label?: string
  phase: ThinkingPhase
  // 主 Agent 推理轮次（1 起计）：与工具调用序号、计划步骤编号语义分离
  reasoning_round?: number
  step?: number
  retry?: number
  status?: string
}

export interface ThinkingToken {
  stream: ThinkingStream
  id: string
  agent?: string
  phase: ThinkingPhase
  delta: string
}

export interface ThinkingEnd {
  stream: ThinkingStream
  id: string
  agent?: string
  phase: ThinkingPhase
  status: 'completed' | 'error' | 'timeout' | 'cancelled' | 'corrected'
  reasoning_round?: number
  reasoning_chars?: number
  truncated?: boolean
  latency_ms?: number
}

export interface StreamHandlers {
  onSession?: (id: string) => void
  onRunStarted?: (value: any) => void
  onMemoryContext?: (value: MemoryContext & { count?: number }) => void
  onMemoryUpdated?: (value: any) => void
  onRoute?: (value: any) => void
  onMeta?: (value: any) => void
  onPlan?: (value: any) => void
  onAgentDecision?: (value: any) => void
  onThinkingStart?: (value: ThinkingStart) => void
  onThinkingToken?: (value: ThinkingToken) => void
  onThinkingEnd?: (value: ThinkingEnd) => void
  onStepStart?: (value: any) => void
  onObservation?: (value: any) => void
  onReflection?: (value: any) => void
  onValidation?: (value: any) => void
  onSources?: (s: SourceItem[]) => void
  onToken?: (t: string) => void
  onTokenReplace?: (t: string) => void
  onToolStart?: (t: any) => void
  onToolToken?: (t: any) => void
  onToolEnd?: (t: any) => void
  onDone?: () => void
  onError?: (e: any) => void
}

// 通过 fetch 读取 SSE，保持工具过程事件与最终答案 token 的独立回调。
// expert_tasks/report_validation 已随专家链移除；旧事件若出现将被静默忽略。
const JSON_EVENT_HANDLERS: Record<string, keyof StreamHandlers> = {
  run_started: 'onRunStarted',
  memory_context: 'onMemoryContext',
  memory_updated: 'onMemoryUpdated',
  route: 'onRoute',
  meta: 'onMeta',
  plan: 'onPlan',
  agent_decision: 'onAgentDecision',
  thinking_start: 'onThinkingStart',
  thinking_token: 'onThinkingToken',
  thinking_end: 'onThinkingEnd',
  step_start: 'onStepStart',
  observation: 'onObservation',
  reflection: 'onReflection',
  validation: 'onValidation',
  sources: 'onSources',
  tool_start: 'onToolStart',
  tool_token: 'onToolToken',
  tool_end: 'onToolEnd',
  error: 'onError',
}

type ParsedSseBlock = { event: string; data: string }

// 导出供单元测试与潜在复用（解析规则：event: 行 + 多行 data: 用 \n 重连）。
export const parseSseBlock = (block: string): ParsedSseBlock => {
  let event = ''
  const dataLines: string[] = []
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      let data = line.slice(5)
      if (data.startsWith(' ')) data = data.slice(1)
      dataLines.push(data)
    }
  }
  return { event, data: dataLines.join('\n') }
}

const createSseDispatcher = (handlers: StreamHandlers) => {
  let completed = false
  const dispatch = (event: string, data: string) => {
    try {
      if (event === 'session') {
        handlers.onSession?.(data)
      } else if (event === 'token') {
        handlers.onToken?.(data)
      } else if (event === 'token_replace') {
        // 输出护栏净化稿：覆盖已流式拼接的内容（去引用/截断引语后可能与原文不同）
        handlers.onTokenReplace?.(data)
      } else if (event === 'done') {
        if (!completed) {
          completed = true
          handlers.onDone?.()
        }
      } else {
        const handlerName = JSON_EVENT_HANDLERS[event]
        const handler = handlerName
          ? handlers[handlerName] as ((value: any) => void) | undefined
          : undefined
        handler?.(JSON.parse(data))
      }
    } catch (error) {
      console.error('[SSE] 事件解析失败', event, error)
    }
  }
  return {
    dispatch,
    finish: () => {
      if (!completed) {
        completed = true
        handlers.onDone?.()
      }
    },
  }
}

const consumeSseStream = async (res: Response, handlers: StreamHandlers) => {
  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  const dispatcher = createSseDispatcher(handlers)

  // 事件块以空行（\n\n 或 \r\n\r\n）分隔；同一块内多行 data: 用 \n 重连。
  // data 内容只剥掉冒号后至多一个前导空格，保留 token 自身的空格和换行。
  const drain = () => {
    const separator = /\r?\n\r?\n/
    let match: RegExpMatchArray | null
    while ((match = buffer.match(separator))) {
      const block = buffer.slice(0, match.index!)
      buffer = buffer.slice(match.index! + match[0].length)
      const parsed = parseSseBlock(block)
      if (parsed.event) dispatcher.dispatch(parsed.event, parsed.data)
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      drain()
      dispatcher.finish()
      return
    }
    buffer += decoder.decode(value, { stream: true })
    drain()
  }
}

export const streamChat = (
  payload: {
    message: string
    role: string
    domain?: 'novel'
    // 三种实际执行策略 + auto（入口路由）；旧值 multi_expert/roleplay 已废弃。
    strategy?: 'auto' | 'direct' | 'react' | 'plan_execute'
    interaction_mode?: 'qa' | 'roleplay'
    max_steps?: number
    memory_mode?: 'auto' | 'off'
    history?: any[]
    session_id?: string
    file_id?: string
    personas?: string[]
    chapter_until?: number
    [key: string]: any
  },
  handlers: StreamHandlers,
) => {
  const controller = new AbortController()
  fetch(BASE + '/chat', {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(payload),
    signal: controller.signal,
  })
    .then(async (res) => {
      await ensureSuccess(res)
      await consumeSseStream(res, handlers)
    })
    .catch((error) => {
      if (error.name !== 'AbortError') handlers.onError?.(error)
    })
  return controller
}

export default { request }
