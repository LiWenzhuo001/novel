# 小说智读 · RAG 问答

基于 FastAPI、LangChain、LangGraph、PostgreSQL 与 pgvector 构建的小说内容 RAG 问答系统：

- **小说导入与索引**：上传 PDF、DOCX、TXT 或 Markdown 小说，按章节清洗分段，记录卷/章、页码、全书片段号等定位元数据。
- **小说问答**：回答人物关系、情节发展、时间线与章节定位问题，答案附原文出处引用。

## 功能特性

- **小说导入与索引**：上传 PDF / Word(.docx) / TXT / MD；清除 BOM、控制字符和多余空白，按章节优先、字符数兜底切分，记录卷/章、页码、全书片段号、章节内片段号和字符区间；索引任务带状态、租约和重启恢复。
- **小说问答**：混合向量 + 中文词法召回（jieba/pg_trgm）+ RRF 融合 + cross-encoder/远程 rerank 重排；按策略补充同文件同章节相邻片段，答案通过独立 SSE `sources` 事件返回原文片段和章节定位。
- **Agent 执行闭环**：所有问答统一使用 `strategy=auto`、`direct`、`multi_expert`、`react` 或 `plan_execute`；复杂问题先拆成四个职责互斥的专家子任务，再并发流式分析、校验重复度、按需纠偏一次并由 Supervisor 去重汇总。
- **记忆自动闭环**：会话前召回摘要/三层记忆，回答后异步生成摘要和稳定事实记忆，并支持用户级隔离、查看和删除。
- **LLM 语义路由**：Query Preparation 每轮只调用一次 LLM，同时判断是否需要小说 RAG；Route Node 负责低置信度、强小说信号和失败场景的保守兜底。
- **正式用户级多租户**：支持 users 表注册/登录、PBKDF2 密码哈希、JWT Bearer 认证；聊天、知识库按 `user_id` 行级隔离。

## 技术栈

| 层 | 技术 |
| --- | --- |
| 后端 | Python 3.12/3.13 + FastAPI + **LangChain + LangGraph Agent Runtime**（ChatOpenAI / OpenAIEmbeddings / RecursiveCharacterTextSplitter） |
| 数据库 | **PostgreSQL 16 + pgvector**（向量与关系数据**共用同一库**：`embeddings` 表存向量，`knowledge_files` / `chat_sessions` / `chat_messages` 存业务数据） |
| 前端 | **Vue3 + Vite + vue-router** + TypeScript + TailwindCSS + 本地内联 SVG 图标 |
| 大模型 | OpenAI 兼容接口（DeepSeek / 通义千问 / 智谱 GLM 等，在 `.env` 配置 `base_url` 与 `api_key`） |
| 向量库 | **pgvector**（PostgreSQL 扩展，零独立向量服务） |

## 目录结构

```
E:/novel/
├── backend/                 # FastAPI + LangChain 后端
│   ├── app/
│   │   ├── core/            # llm / embed / rag / query_rewriter / rerank 核心封装
│   │   ├── services/        # kb_service（入库）/ novel_service（分块）/ memory_service（记忆）
│   │   ├── api/             # chat / knowledge / auth 路由
│   │   ├── models/          # Pydantic schemas
│   │   ├── db/              # PostgreSQL + pgvector 数据层（engine / session / ORM 模型）
│   │   │   ├── __init__.py  # engine、get_db、init_db（启用 vector 扩展 + 建表 + HNSW 索引）
│   │   │   └── models.py    # Embedding / KnowledgeFile / ChatSession / ChatMessage
│   │   ├── config.py        # 配置（含 PostgreSQL 连接与 EMBED_DIM）
│   │   └── main.py          # 入口（lifespan 中 init_db）
│   ├── data/                # 上传原文（已在 .gitignore）
│   ├── initdb/              # 01-extensions.sql：首次启动启用 vector 扩展
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/                # Vue3 + Vite 前端
│   ├── src/pages/           # chat / login 页面（vue-router）
│   ├── src/components/      # ChatPanel / KnowledgeManager / Icon
│   ├── src/api/client.ts    # fetch 封装 + SSE 流式（ReadableStream）
│   └── Dockerfile / nginx.conf
└── docker-compose.yml       # postgres(pgvector) + backend + frontend
```

## 快速开始（本地开发）

### 1. 后端 + PostgreSQL（含 pgvector）

```bash
# 启动带 pgvector 的 PostgreSQL（或已有的本地实例，需已启用 vector 扩展）
docker run -d --name novel_rag_postgres \
  -e POSTGRES_USER=job_agent -e POSTGRES_PASSWORD=change_me_to_a_strong_password \
  -e POSTGRES_DB=job_agent -p 5432:5432 \
  pgvector/pgvector:0.8.0-pg16

cd backend
pip install -r requirements.txt          # 建议使用虚拟环境
cp .env.example .env                     # 然后填写你的 API Key 与 PostgreSQL 连接
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`.env` 关键配置：

```ini
# 大模型
LLM_API_KEY=你的key
LLM_BASE_URL=https://api.deepseek.com/v1   # 或通义/智谱的兼容地址
LLM_MODEL=deepseek-chat
EMBEDDING_PROVIDER=api                     # api 走兼容接口；local 走本地 bge
EMBEDDING_API_KEY=你的embedding_key
EMBEDDING_BASE_URL=https://api.deepseek.com/v1
EMBEDDING_MODEL=text-embedding-3-small     # 或 Qwen/Qwen3-Embedding-0.6B
# 第三方 OpenAI-compatible 服务（如 SiliconFlow/Qwen）直接接收原始字符串，
# 不要使用 OpenAI tiktoken 的 token ID 输入；原生 OpenAI tokenizer 场景才设为 true。
EMBEDDING_CHECK_CTX_LENGTH=false

# 向量维度：必须与 EMBEDDING_MODEL 一致！
# text-embedding-3-small=1536 / text-embedding-3-large=3072 /
# Qwen/Qwen3-Embedding-0.6B=1024 / BAAI/bge-m3=1024
EMBED_DIM=1536

# 重排：local 使用本地 CrossEncoder；siliconflow 使用硅基流动远程 API
RERANKER_PROVIDER=siliconflow
RERANKER_MODEL=Qwen/Qwen3-Reranker-8B
RERANKER_BASE_URL=https://api.siliconflow.cn/v1
RERANKER_ENDPOINT=/rerank
RERANKER_API_KEY=你的硅基流动key
RERANKER_TIMEOUT=30
RERANKER_MAX_RETRIES=2
RERANKER_CANDIDATE_N=60
# 远程失败时跳过重排，继续使用原候选回答，不自动切回本地模型

# PostgreSQL + pgvector（向量与业务数据共用同一库）
POSTGRES_HOST=postgres     # 本地直连时改为 127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=job_agent
POSTGRES_USER=job_agent
POSTGRES_PASSWORD=change_me_to_a_strong_password
# 也可直接给完整连接串（优先级最高）：
# DATABASE_URL=postgresql+psycopg2://job_agent:change_me_to_a_strong_password@127.0.0.1:5432/job_agent

# 用户认证：默认开启（users 表 + JWT 登录）。JWT_SECRET 为必填，未配置将启动失败；
# 纯本地裸跑开发可显式指定 JWT_SECRET=dev-xxx python app/main.py
USER_AUTH_ENABLED=true
JWT_SECRET=请替换为高强度随机串
JWT_ACCESS_TOKEN_MINUTES=1440

# Agent 执行策略：auto / direct / multi_expert / react / plan_execute
AGENT_MAX_STEPS=6
AGENT_TOOL_TIMEOUT=20
AGENT_MAX_EXPERTS=4
AGENT_MULTI_EXPERT_TIMEOUT=45
AGENT_EXPERT_MAX_TOKENS=800
AGENT_SYNTHESIS_MAX_TOKENS=1200
AGENT_EXPERT_DISPATCH_MODE=hybrid
AGENT_DISPATCH_MAX_TOKENS=500
AGENT_REPORT_SIMILARITY_THRESHOLD=0.72
AGENT_EXPERT_CORRECTION_RETRIES=1
NOVEL_CONTEXT_K=10
NOVEL_NEIGHBOR_WINDOW=1
```

> 后端启动时会自动 `CREATE EXTENSION IF NOT EXISTS vector`（docker 环境下由 `initdb/01-extensions.sql` 以超级用户预建）、`CREATE TABLE` 并为 `embeddings.embedding` 建 HNSW 索引；在 PostgreSQL 未就绪时会**重试等待**，不会因启动顺序问题直接崩溃。

### 2. 前端（Vue3 + Vite）

```bash
cd frontend
npm install
npm run dev                               # 默认 http://localhost:5173
npm run build                             # 生产构建产物位于 dist/
```

前端通过 Vite 代理把 `/api` 转发到后端 `http://localhost:8000`（见 `vite.config.ts`）。
浏览器端使用原生 `fetch` + `ReadableStream` 解析 SSE 流式响应。

## 使用流程

1. 首次使用先注册/登录；启用 `USER_AUTH_ENABLED=true` 后，前端会通过 `/login` 获取 JWT 并自动附加 `Authorization: Bearer ...`。
2. 在小说工作台右侧「小说书库」上传作品，等待状态变为 `indexed`；直接提问；系统会根据问题复杂度自动选择 direct 或 plan_execute，也可通过 API 显式指定 react。
3. 答案下方显示文件、章节、页码、片段号与原文摘要；复杂问题可查看 Agent 计划、工具调用、观察和回退状态。
4. 对话按会话保存，不同用户只能看到自己的会话与知识库。

## LangChain / LangGraph 架构要点

```
Document Loaders → 章节感知 RecursiveCharacterTextSplitter
       → pgvector + 中文词法/FTS + RRF + reranker
       → Query Preparation（改写 + LLM 语义路由）→ LangGraph Agent Router → 条件 Shared Retrieval
              ├── direct → Supervisor
              ├── multi_expert → 专家任务分解 → 四专家并发 → 契约/重复校验 → 可选纠偏 → Supervisor
              └── react / plan_execute → Tool Runtime → Reflect → Supervisor
       → 专家 tool_token + 最终 token → SSE
```

- **解析与入库**：`services/kb_service.py` 复用 LangChain loader，交给 `services/novel_service.py` 章节感知切分，统一写入 pgvector，并记录文件状态、租约和重试次数。
- **领域检索**：`core/rag.py` 的向量与 FTS 分支都强制 `user_id` 过滤；小说命中后按 `file_id + chapter_no + chunk_no` SQL 范围查询相邻片段。
- **Agent 编排**：`app/agent/` 使用 LangGraph 管理 `route → plan → retrieve → dispatch → experts → validate/refine → supervisor`；原始会话 Query 先被改写为独立检索问题，再由 Dispatcher 生成四个专属子任务，专家仍共享同一次 RAG 证据。
- **流式与回退**：`expert_tasks` 先展示本轮分工，专家通过独立 `tool_token` 交错输出；报告不符合契约或相似度超过阈值时只纠偏一次，部分失败仍汇总成功报告，全部无效则降级到 direct。

## 数据库设计（PostgreSQL + pgvector，单库统一）

| 表 | 作用 |
| --- | --- |
| `users` | 用户账号：用户名、邮箱、PBKDF2 密码哈希、启用状态、管理员标记与最近登录时间 |
| `embeddings` | **RAG 向量片段**：内容、向量、来源、`file_id`、`user_id`，以及小说专用 `chapter/chapter_no/chunk_no/page` 定位列和其余 JSON 元数据 |
| `knowledge_files` | 知识库文件清单：文件名、状态、租约，以及 embedding 模型、维度、分块参数和索引版本 |
| `chat_sessions` | 会话：id、标题、角色视角、创建/更新时间、user_id |
| `chat_messages` | 消息：自增 id、所属 session、role（user/assistant）、内容、来源、时间（级联删除） |
| `conversation_summaries` | 会话摘要：覆盖消息范围、摘要内容和 token 估算 |
| `agent_memories` | 用户/小说长期记忆：内容、重要性、来源消息、过期时间和可选向量 |

> 向量与业务数据在同一 Postgres 实例同一库中，可用一条 SQL 同时做向量检索与关系过滤，无需跨库 JOIN 或双写一致性处理。

## 常见问题

- **缺少 API Key**：聊天与上传资料（需 embedding）都会校验 Key，请先在 `.env` 配置。
- **8000 端口被占用**：用 `uvicorn ... --port 8010` 并同步修改 `vite.config.ts` 的 proxy 目标。
- **本地 Embedding**：将 `EMBEDDING_PROVIDER` 设为 `local` 可免去 embedding 费用，但需已安装 `sentence-transformers` 且首次会下载 bge 模型。
- **远程 rerank**：设置 `RERANKER_PROVIDER=siliconflow`、`RERANKER_API_KEY`，并使用模型 `Qwen/Qwen3-Reranker-8B`。请求发送到 `https://api.siliconflow.cn/v1/rerank`；远程调用**失败**（网络/HTTP）时跳过重排继续回答；配置**缺失**（未配 `RERANKER_API_KEY`）会在首次重排调用时 fail-fast。。
- **EMBED_DIM 不匹配**：`EMBED_DIM` 必须与 `EMBEDDING_MODEL` 的维度一致。修改模型或分块参数后使用 `python scripts/reindex_file.py <file_id>` 原子重建索引；系统会拒绝混用不兼容版本。远程 rerank 模型切换不需要重建 embedding 索引。
- **登录后 401**：确认后端已设置稳定 `JWT_SECRET`，前端 localStorage 中的旧 token 可清除后重新登录。
- **PostgreSQL 连接失败**：确认 `POSTGRES_*` 配置正确；若用 docker-compose，后端会等待 `postgres` 健康检查通过后再连库。
- **数据库扩展缺失**：自建 PostgreSQL需启用 `vector`；中文词法检索建议同时启用 `pg_trgm`。`pg_trgm` 不可用时系统自动回退 simple FTS。
- **PDF 解析质量**：使用 `pypdf`，复杂排版（多栏/表格）可能需后期校对。

## LLM 辅助章节规则发现

手动重新索引会先运行确定性章节解析。仅当章节数为 0、长文本章节数异常少、未分配片段比例过高或编号异常时，系统才把最多 240 条短行候选交给模型。模型输出固定 DSL（章节单位、编号方式、字面前后缀和标题边界），程序完成数量、顺序、重复率、正文长度和误切验证后才会应用；模型不得生成或执行任意正则。

验证通过的规则按原文 SHA-256、章节解析器版本、模型和 Prompt 版本缓存。模型调用或规则验证失败时不会替换旧向量，文件继续可检索并显示失败原因。初次上传不调用该模型辅助流程。

相关配置：`ENABLE_LLM_CHAPTER_DETECTION`、`CHAPTER_DETECTION_MODEL`、`CHAPTER_DETECTION_TIMEOUT`、`CHAPTER_DETECTION_CANDIDATE_LIMIT` 和 `CHAPTER_DETECTION_CONFIDENCE_THRESHOLD`。
## RAG 质量评测与重新索引

```bash
cd backend
python scripts/preload_models.py
python scripts/reindex_file.py <file_id> --user default
python scripts/run_rag_eval.py --eval evals/rag_queries.json --output evals/results/current
```

- `preload_models.py` 在部署阶段预下载 embedding/reranker，避免首个聊天请求等待模型下载。
- `reindex_file.py` 先生成全部新向量，再在单个数据库事务中替换旧片段；失败时旧索引保持可用。
- `run_rag_eval.py` 输出 Recall@5/10、Precision@5、MRR@10、nDCG@10、无结果率和延迟 JSON/Markdown 报告；正式验收要求至少 40 条 `validated=true` 的人工标注。
- 来源分数不是 Recall。主命中展示向量/词法/混合/重排分，相邻片段展示“上下文补充”。
## Agent 策略与数据库迁移

`/api/chat` 请求使用 `strategy`：`auto`、`direct`、`multi_expert`、`react` 或 `plan_execute`，并可通过 `max_steps` 限制执行预算。流式响应包含 `route`、`plan`、`sources`、`expert_tasks`、`tool_start`、`tool_token`、`tool_end`、`validation`、`reflection`、`token` 和 `meta` 事件。

数据库新增 `conversation_summaries` 和 `agent_memories` 表。正式环境使用 Alembic：

```bash
cd backend
alembic upgrade head
```

## Docker 部署

```bash
docker compose up --build
```

将同时启动 **PostgreSQL（pgvector，:5432）**、后端（:8000）与前端（H5 构建产物由 nginx 提供并反向代理 `/api`）。
后端通过 `depends_on: postgres: condition: service_healthy` 与启动重试逻辑保证在 PostgreSQL 就绪后再建表；`embeddings` 表的 HNSW 索引会在首次启动时创建。Compose 默认把 PostgreSQL 和后端端口绑定到 `127.0.0.1`，生产环境建议只暴露前端或由反向代理统一入口。

## 数据库迁移

项目已加入 Alembic 基础配置，首个迁移位于 `backend/alembic/versions/20260802_0001_initial_pgvector_schema.py`。

```bash
cd backend
alembic upgrade head
```

当前迁移头为 `20260824_0005_rag_retrieval_quality.py`，新增中文 trigram 索引和知识文件索引版本字段。`app.db.init_db()` 仍保留幂等建表和补列逻辑；正式部署建议以 Alembic 为准管理结构变更。

## 实施假设与开源复用依据

实施基于以下假设：小说文本以章节标题和自然段为主要结构；当前规模可由单一 PostgreSQL 承载；人物关系和时间线首先要求“有原文证据的分析”，暂不要求持久化知识图谱或图可视化。

| 项目 | 许可证与维护信号 | 本项目复用或参考 |
| --- | --- | --- |
| [LangChain](https://github.com/langchain-ai/langchain) | MIT；1.3.x 持续发布 | 复用 Document、loader、递归分块、ChatOpenAI/Embeddings 接口，不重复实现格式解析与模型适配 |
| [pgvector](https://github.com/pgvector/pgvector) | PostgreSQL License；持续维护 | 复用余弦检索和 HNSW，在同一 SQL 中执行租户和章节过滤，避免引入第二套向量服务 |
| [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | MIT；持续维护 | 参考实体关系、全局/局部检索与冲突意识；未整体引入，因为其 LLM 图谱索引成本、批处理管线和配置迁移会显著扩大当前部署与数据模型 |

当前人物、情节和时间线问题采用“混合召回 + 相邻上下文 + Agent 证据分析”。若后续评测证明跨章节、多跳关系问题仍明显不足，再以离线可选模块引入实体关系表或 GraphRAG，而不是替换现有在线 RAG 主链路。
