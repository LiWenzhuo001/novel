# 《小说智读》RAG 问答系统 · 循序渐进源码教程

> 面向初学者，但保证你读到最后也不掉队。
> 每一章只引入一个新概念：先讲清楚"它是什么、解决什么问题"，再回到真实源码。
> 所有代码片段都注明文件路径与行号（以当前仓库版本为准），核心语句附逐行注释。

---

## 目录

- [第 0 章 项目全景](#第-0-章-项目全景)
- [第 1 章 配置层：一切从 `settings` 开始](#第-1-章-配置层一切从-settings-开始)
- [第 2 章 数据层：一个库同时存"向量"和"业务数据"](#第-2-章-数据层一个库同时存向量和业务数据)
- [第 3 章 索引管线：把一本书切成"可检索的卡片"](#第-3-章-索引管线把一本书切成可检索的卡片)
- [第 4 章 检索（一）：向量通道与多租户隔离](#第-4-章-检索一向量通道与多租户隔离)
- [第 5 章 检索（二）：词法通道与两路融合](#第-5-章-检索二词法通道与两路融合)
- [第 6 章 检索（三）：cross-encoder 重排与邻居上下文](#第-6-章-检索三cross-encoder-重排与邻居上下文)
- [第 7 章 Query Preparation：问句改写与语义路由](#第-7-章-query-preparation问句改写与语义路由)
- [第 8 章 Agent 编排：LangGraph 多专家流水线](#第-8-章-agent-编排langgraph-多专家流水线)
- [第 9 章 API 层：用 SSE 把思考过程"直播"出去](#第-9-章-api-层用-sse-把思考过程直播出去)
- [第 10 章 前端：用 fetch 手工解析 SSE 流](#第-10-章-前端用-fetch-手工解析-sse-流)
- [第 11 章 记忆闭环：摘要与三层长期记忆](#第-11-章-记忆闭环摘要与三层长期记忆)
- [第 12 章 评测体系：让每一次改进可被证明](#第-12-章-评测体系让每一次改进可被证明)
- [第 13 章 端到端执行链路串讲](#第-13-章-端到端执行链路串讲)
- [第 14 章 术语表](#第-14-章-术语表)

---

## 第 0 章 项目全景

### 0.1 它解决什么问题

想象你在读一本几百章的长篇小说，想知道"贾宝玉和林黛玉的关系是怎么变化的"。
翻书找答案很痛苦——于是你把这个任务交给一个 AI 助手。但直接问大语言模型（LLM）有两个致命问题：

1. **模型没读过这本书**。通用模型对具体小说的情节要么不知道，要么一本正经地编造（幻觉）。
2. **上下文装不下整本书**。一本 100 万字的小说远超模型一次能处理的长度。

这个项目的答案就叫 **RAG（Retrieval-Augmented Generation，检索增强生成）**。
用开卷考试来类比最贴切：模型不是靠背诵（参数记忆）答题，而是开卷——先从书里**检索**出最相关的几段原文，
再把这几段原文和你的问题一起交给模型，让它**只依据这几段**作答，并标注引用了哪一段。

所以《小说智读》做的事情可以概括成一条流水线：

```
上传小说 → 清洗/按章节切块 → 每块算出向量存入 PostgreSQL
用户提问 → 改写问题 → 混合检索（向量+词法）→ 重排 → 取相邻段落补上下文
        → 按复杂度路由到不同 Agent 策略 → 生成带 [S#] 引用的答案 → 流式返回
```

### 0.2 整体架构

```
┌─────────────────────────── 浏览器（Vue3 + Vite）────────────────────────────┐
│  chat.vue（页面骨架/选书/策略切换）                                            │
│    ├─ ChatPanel.vue   消息流 + 专家过程 + 引用来源                            │
│    ├─ KnowledgeManager.vue  上传/索引进度/删除                                │
│    └─ client.ts        fetch 封装 + 手工解析 SSE 流                           │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │  /api/*（Vite proxy 或 nginx 反代）
┌──────────────────────────────────▼─────────────────────────────────────────┐
│  FastAPI 后端（backend/app/main.py）                                        │
│    AuthMiddleware（JWT → user_id 注入 ContextVar）                          │
│    ├─ api/knowledge.py  上传 → 后台索引任务                                  │
│    ├─ api/chat.py       SSE 流式问答（本章的"总导演"）                        │
│    ├─ api/auth.py       注册/登录/JWT                                        │
│    └─ api/memory.py     记忆查看/删除                                        │
│                                                                            │
│  core/                     services/              agent/                    │
│   ├─ rag.py 混合检索        ├─ kb_service.py 索引   ├─ runtime.py LangGraph   │
│   ├─ rerank.py 重排        ├─ novel_service.py 切块 ├─ router.py 路由        │
│   ├─ query_rewriter.py 改写 ├─ memory_service.py   ├─ dispatcher.py 拆任务   │
│   ├─ embed.py / llm.py     │  记忆闭环              ├─ tools.py 工具注册表   │
│   └─ security.py JWT       └─ chapter_detection.py └─ contracts/validation   │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │ SQL（asyncpg）
┌──────────────────────────────────▼─────────────────────────────────────────┐
│  PostgreSQL 16 + pgvector + pg_search(ParadeDB)                             │
│   users / knowledge_files / embeddings(向量+tsvector) /                      │
│   chat_sessions / chat_messages / conversation_summaries / agent_memories    │
└────────────────────────────────────────────────────────────────────────────┘
                                   │ HTTPS（OpenAI 兼容接口）
                        ┌──────────▼──────────┐
                        │ LLM（DeepSeek/通义/智谱）│
                        │ Embedding API / 本地 bge │
                        └──────────────────────┘
```

三个值得先记住的架构决策（后面各章都会反复出现）：

1. **向量与业务数据共用一个 PostgreSQL**。不引入独立的向量数据库服务（如 Milvus），
   一条 SQL 就能同时做"向量相似 + 用户过滤 + 章节定位"，免去跨库 JOIN 和双写一致性。
2. **检索是混合的**。向量检索擅长"意思相近"，词法检索擅长"字面命中"（人名、绰号），
   两者用 RRF 融合，再用 cross-encoder 重排，最后补邻居段落保证上下文完整。
3. **所有问答统一走一个 Agent 图**。简单问题走 `direct` 两步直达；复杂问题走 `multi_expert`
   多专家并发。无论哪条路，检索只做一次，专家共享同一份证据。

### 0.3 一次提问的调用流程（先混个眼熟）

```
用户输入
  → POST /api/chat（api/chat.py）
    → 记忆召回 memory_service.build_memory_context     ……第 11 章
    → 改写 rewrite_query（指代消解 + 是否需要检索）      ……第 7 章
    → stream_agent_question（agent/runtime.py）         ……第 8 章
        route → plan → retrieve（core/rag.py 混合检索）  ……第 4/5/6 章
              → dispatch → experts ×4 → validate → refine? → supervisor → summary
        每个节点都往 asyncio.Queue 吐事件
    → event_gen 把事件翻译成 SSE 推给浏览器              ……第 9 章
  → 前端 client.ts 逐块解析 SSE，ChatPanel 渲染          ……第 10 章
  → 回答完成后后台异步更新摘要与记忆                       ……第 11 章
```

### 0.4 阅读前的前置知识

本教程会在用到的地方现讲概念，但你最好对下面这些有"听说过"级别的了解：

| 领域 | 需要的程度 | 不熟也没关系，因为 |
| --- | --- | --- |
| Python 基础 | 变量、函数、类、装饰器 | 教程会解释每个新语法 |
| `async/await` | 知道"异步=等待 IO 时让出 CPU" | 第 2/8/9 章会结合例子讲 |
| SQL 基础 | 会写 `SELECT ... WHERE ... ORDER BY LIMIT` | 第 2 章从建表讲起 |
| HTTP | 知道请求/响应、header | 第 9 章讲 SSE 时补背景 |
| Vue 3 | 知道组件、`ref`、模板绑定 | 第 10 章只讲用到的部分 |
| LLM 概念 | 知道"模型收文本、吐文本" | RAG/向量等概念全部现讲 |

推荐边读边开着仓库对照：先读每章"类比"建立直觉，再对照源码验证理解。

**本章小结**：这是一个"开卷考试"式的小说问答系统；一条主线是**把书变成可检索的向量卡片**（第 2~3 章），
另一条主线是**把问题变成带引用的答案**（第 4~10 章），外加两翼——记忆（第 11 章）和评测（第 12 章）。
下面我们从地基开始。

---

## 第 1 章 配置层：一切从 `settings` 开始

### 1.1 新概念：环境变量与配置单例

**类比**：把程序想象成一间餐厅。菜单（代码逻辑）是固定的，但今天用哪个供应商的食材（API Key）、
桌子摆几张（端口）、打烊时间（超时）不该焊死在菜单里，而应贴在后厨的黑板上，随时可换。
这块"黑板"就是**环境变量**，通常写在项目根目录的 `.env` 文件里。

Python 社区的惯用做法是：用 `python-dotenv` 把 `.env` 读进环境变量，再用一个配置类统一收口。
这个项目的收口点是 `backend/app/config.py`。

### 1.2 源码走读

先看加载 `.env` 和一个有趣的细节——代理绕行（`config.py:17-33`）：

```python
load_dotenv()            # 把 .env 文件的内容读进环境变量，之后 os.getenv 才能读到

_proxy_bypass = (
    "hf-mirror.com,huggingface.co,...,api.deepseek.com,..."   # 这些域名不走系统代理
)
_existing_np = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
_merged_np = ",".join(x for x in [_existing_np, _proxy_bypass] if x)
os.environ["NO_PROXY"] = _merged_np     # 追加而非覆盖，尊重用户已有配置
```

为什么？注释写得很清楚：本地代理（如 Clash）转发 HuggingFace 大文件下载不稳定，
而 LLM API 走代理又会"远程计算机拒绝网络连接"。把模型源和 API 域名加入 `NO_PROXY` 直连，
属于**在代码里固化环境适配经验**——这正是配置层该干的事。

接着是配置类的收口（`config.py:39-47` 和 `config.py:295-302`）：

```python
class Settings:
    """统一管理所有配置项，通过环境变量注入。"""
    def __init__(self):
        self.llm_api_key = os.getenv("LLM_API_KEY", "")          # 大模型 key
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        self.llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
        ...
        self.embed_dim = int(os.getenv("EMBED_DIM", "1536"))     # 向量维度，第 2 章主角
        ...

    def validate(self):
        if not self.llm_api_key:
            raise ValueError("LLM_API_KEY 未配置，请在 .env 中设置")
        ...

settings = Settings()   # 模块级单例：所有文件都 from app.config import settings
```

`config.py:172-175` 还有一处启动即失败（fail-fast）的例子：

```python
self.novel_chunk_size = max(200, int(os.getenv("NOVEL_CHUNK_SIZE", "650")))     # 每块目标长度
self.novel_chunk_overlap = max(0, int(os.getenv("NOVEL_CHUNK_OVERLAP", "120"))) # 相邻块重叠
if self.novel_chunk_overlap >= self.novel_chunk_size:
    raise ValueError("NOVEL_CHUNK_OVERLAP 必须小于 NOVEL_CHUNK_SIZE")
```

### 1.3 为什么这样写

- **为什么单例而不是每次 `os.getenv`？** 把"读环境变量 + 类型转换 + 边界钳制 + 校验"集中到一次，
  其余代码只面对类型安全的属性（`settings.embed_dim` 一定是 `int`）。散落的 `getenv` 会让
  "字符串 vs 整数""默认值不一致"这类 bug 到处繁殖。
- **为什么在 `__init__` 里就 `raise`？** 配置写错（比如 overlap ≥ size）属于**必然导致后续逻辑错误**的问题，
  越早暴露越好。`import` 阶段直接崩掉，比运行到第 3000 个 chunk 才出现诡异结果便宜得多。
  这也是贯穿全项目的原则：**能fail-fast 的地方不静默降级**。
- **替代方案**：`pydantic-settings` 可以用声明式字段替代手写 `__init__`，更优雅；
  本项目手写是为了少一个依赖、且能写自定义钳制逻辑（`max/min/raise`）。两者取舍见仁见智。

> **坑与注意**：改任何 `.env` 后要重启进程才会生效；`EMBED_DIM` 改了必须重建向量索引（第 2、3 章解释）。

**本章小结**：`settings` 是全项目唯一的配置出口，模块导入即完成校验。
有了地基，下一步是它指向的数据库。

---

## 第 2 章 数据层：一个库同时存"向量"和"业务数据"

### 2.1 新概念：向量、embedding、pgvector、ORM

先解决三个术语（更完整的定义见第 14 章术语表）：

- **embedding（嵌入向量）**：把一段文字变成一串数字，比如 1536 个 0~1 之间的小数。
  意思相近的两段文字，向量在 1536 维空间里的"夹角"就小。
- **向量检索**：给一个查询向量，找出库里夹角最小（最相似）的 K 条记录。
  常用指标是**余弦距离** = 1 − cos(夹角)，越小越相似。
- **pgvector**：PostgreSQL 的一个扩展，让普通表多出一列 `vector` 类型并支持相似度排序。
  类比：Excel 多了一列"数值数组"，还能按"数组有多像你手里这个"排序。

**ORM（对象关系映射）**：用 Python 类表示数据库表、用对象表示行，避免手拼 SQL 字符串。
本项目用 SQLAlchemy 2.0 的**异步**版本——等数据库返回时让出 CPU，去处理别的请求。

### 2.2 表结构：先看 `Embedding`，它是全书的主角

`backend/app/db/models.py:33-62`：

```python
class Embedding(Base):
    """RAG 向量片段（pgvector）。"""
    __tablename__ = "embeddings"

    id = Column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)  # 随机主键
    content = Column(Text, nullable=False)                     # 小说原文片段
    embedding = Column(Vector(settings.embed_dim), nullable=False)  # 向量列，维度来自配置
    source = Column(String(255), nullable=False, index=True)   # 文件名
    file_id = Column(String(32), index=True)                   # 属于哪本书
    user_id = Column(String(64), index=True, default="default")# 多租户隔离（第 4 章）
    domain = Column(String(32), index=True, default="novel")   # 业务域，目前都是 novel
    chapter = Column(String(255), index=True)                  # 章节标题（引用展示用）
    chapter_no = Column(Integer, index=True)                   # 章节序号（定位/邻居扩展用）
    chunk_no = Column(Integer, index=True)                     # 全书片段号（引用 [S#] 用）
    page = Column(Integer)                                     # PDF 页码
    meta_json = Column(Text)                                   # 其余元数据兜底
    # 全文检索向量：PostgreSQL "生成列"，写入时自动由 content 计算并持久化
    search_vector = Column(
        TSVECTOR,
        Computed("to_tsvector('simple', content)", persisted=True),
    )
    created_at = Column(DateTime, default=datetime.utcnow)
```

逐行看重点：

- `id` 用 `uuid4().hex`：分布式友好的随机字符串主键，避免自增 ID 在导入/迁移时冲突。
- `embedding` 的维度**在建模时就固定**为 `settings.embed_dim`。这带来强约束：
  换 embedding 模型（比如 1536 维换 1024 维）必须连配置一起改，否则插入直接报错。
- `search_vector` 是**生成列**：`to_tsvector('simple', content)` 把原文切成词法检索用的词袋。
  `persisted=True` 表示落盘存储——写入慢一点，检索快很多。
- `chapter_no / chunk_no` 两个整数是小说场景的灵魂：第 6 章的"邻居上下文"、答案里的"第 N 章 片段 M"
  全靠它们。

其余表一图流（`models.py:16-30, 65-111, 114-143, 146-180`）：

```
users            账号（PBKDF2 密码哈希，见第 9 章）
knowledge_files  每本书一条：状态机 pending→indexing→indexed/failed + 租约 + 索引版本指纹
chat_sessions    会话（绑定 file_id：一场对话只聊一本书）
chat_messages    消息（sources 列以 JSON 字符串存引用）
conversation_summaries  会话摘要（第 11 章）
agent_memories   长期记忆（第 11 章，也有可选向量列）
```

### 2.3 连接与启动初始化

`backend/app/db/__init__.py:29-44`：

```python
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,      # postgresql+asyncpg://...，异步驱动
    pool_pre_ping=True,      # 每次借连接前先 ping 一下，剔除被防火墙掐死的死连接
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,  # 提交后对象属性仍可读；否则一读属性就再发一条 SELECT
    autoflush=False,         # 不在查询前偷偷把未保存改动刷进库，行为更可预期
)

Base = declarative_base()
from app.db import models  # 必须在 Base 之后导入，让所有表注册进 Base.metadata
```

三个参数各挡住一类坑：

- `pool_pre_ping=True`：长连接被数据库/防火墙静默回收后，第一条查询会莫名其妙失败。ping 一次花 1ms，换稳定。
- `expire_on_commit=False`：默认行为下 `commit()` 后对象"过期"，访问字段会触发懒加载 IO——
  在异步代码里这就是一个隐藏的同步 IO 炸弹。关掉它，提交后对象就是普通字典一样的存在。
- `autoflush=False`：查询前自动 flush 会让"我只是查一下"意外写出半成品数据，显式提交才写库更符合直觉。

启动时的 `init_db`（`db/__init__.py:53-107`）做了三件事，逐段看：

```python
async def init_db(max_retries: int = 30, retry_interval: float = 2.0) -> None:
    # 1) 重试等待数据库就绪：docker-compose 里后端比 postgres 先起是常态
    for attempt in range(1, max_retries + 1):
        try:
            async with async_engine.connect() as conn:
                pass
            break
        except OperationalError as e:
            last_err = e
            if attempt < max_retries:
                await asyncio.sleep(retry_interval)
    else:
        raise last_err                       # for-else：循环没 break 说明 30 次全失败
```

`for...else` 是 Python 小众但好用的语法：`else` 在循环**没有被 break** 时执行，正好表达"重试耗尽"。

```python
    # 2) 启用扩展（幂等：IF NOT EXISTS）
    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    ...
    # 3) 建表 + 维度守卫
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)      # 建缺失的表（不动已有表）
        dimension_result = await conn.execute(text("""
            SELECT a.atttypmod FROM pg_attribute ...        # 查 embeddings.embedding 的实际维度
        """))
        actual_dimension = dimension_result.scalar_one_or_none()
        if actual_dimension is not None and int(actual_dimension) != settings.embed_dim:
            raise RuntimeError(                             # 配置与库不一致 → 启动即失败
                f"embeddings.embedding 当前为 {actual_dimension} 维，"
                f"运行配置要求 {settings.embed_dim} 维；请先执行向量维度迁移。"
            )
```

最后建索引（`db/__init__.py:207-226`）：

```python
await conn.execute(text(
    "CREATE INDEX IF NOT EXISTS embeddings_embedding_idx "
    "ON embeddings USING hnsw (embedding vector_cosine_ops)"   # 向量近似检索索引
))
await conn.execute(text(
    "CREATE INDEX IF NOT EXISTS embeddings_search_vector_idx "
    "ON embeddings USING gin (search_vector)"                  # 全文检索倒排索引
))
```

- **HNSW**（分层可导航小世界图）是向量检索的近似索引：牺牲一点点召回率，换几十上百倍的检索速度。
  不建它，每次检索都要和全表逐条算余弦——百万片段时代价不可接受。
- **GIN** 是倒排索引，服务 `search_vector` 的 `@@` 匹配，第 5 章会用到。

### 2.4 为什么这样写

- **为什么单库而不是"PostgreSQL 存业务 + 独立向量库"？**
  检索时必须带 `user_id == 当前用户 AND domain == 'novel' AND file_id == 本书` 过滤（第 4 章）。
  单库里这是 WHERE 条件；跨库就要"各自查一批再内存拼接"，既慢又有租户泄露风险。
  README 里也明确写了这个取舍：当前规模单库够用，等数据量真撑爆再拆。
- **为什么 `init_db` 里写一堆 `ADD COLUMN IF NOT EXISTS`？** `create_all` 只建缺失的**表**，
  不会给已有表**加列**。这些语句让老库无痛升级（幂等），属于轻量迁移；正式环境则用 Alembic
  （`backend/alembic/versions/`）管理结构版本。
- **替代写法**：把维度守卫去掉、插错维度时让 pgvector 自己报错也可以——但那个报错发生在
  "用户上传完一本书、跑了几分钟 embedding 之后"，而现在的守卫在**启动 3 秒内**就告诉你配置错了。

**本章小结**：一张 `embeddings` 表同时装下了原文、向量、章节定位和多租户字段；
引擎层用三个参数挡住了连接池/懒加载/隐式 flush 三类坑；启动逻辑全部幂等且 fail-fast。
书和库都有了，下一章把书"装"进库里。

---

## 第 3 章 索引管线：把一本书切成"可检索的卡片"

### 3.1 新概念：Document、分块、状态机

- **Document**：LangChain 的标准数据结构，就两个字段——`page_content`（正文）和 `metadata`（字典）。
  本项目里一本小说会被拆成上千个 Document，每个带"第几章、第几段"的元数据。
- **分块（chunking）**：模型一次读不了整本书，检索命中的也只能是"段落"而不是"整本"。
  把长文切成几百字的块，每块独立算向量。块太大→检索不精准；块太小→上下文断裂。
- **状态机**：一本书从上传到可检索要经过多个阶段。用数据库里的一个 `status` 字段记录当前阶段，
  每次只允许合法的迁移（pending→indexing→indexed/failed），这就是最朴素的状态机。

### 3.2 上传入口：立即返回，索引交给后台

`backend/app/api/knowledge.py:22-65`（有删节）：

```python
@router.post("/kb/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:                 # 1) 扩展名白名单
        raise HTTPException(status_code=400, ...)
    content = await file.read()                       # 2) 整体读入（上限 10MB）
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, ...)
    info = await kb_service.create_pending_file(filename, content)  # 3) 落盘 + 建记录
    metrics.incr("kb_uploads")
    # 4) 关键：索引丢给响应结束后的后台任务，接口立刻返回
    background_tasks.add_task(kb_service.run_indexing, info["id"], filename, ext, get_current_user())
    return {"code": 0, "data": info}                  # 前端拿到 file_id，靠轮询看进度
```

为什么用后台任务？整本书的清洗 + 调 embedding API 可能要几分钟。HTTP 请求挂几分钟必超时。
"落盘 + 建 pending 记录 + 秒回"是标准的**异步任务**模式，前端按 `file_id` 轮询
`/api/kb/files` 看进度（`KnowledgeManager.vue` 就是这么做的）。

### 3.3 文本解码：给每种候选编码打分

TXT 小说最常见的坑是**编码**：同一个文件可能是 UTF-8、GB18030、GBK、Big5。
`backend/app/services/kb_service.py:69-94` 的解法是"全部试一遍，谁最像中文小说谁赢"：

```python
def _decode_quality(text: str, encoding: str) -> float:
    """给解码结果打分：中文多加分，乱码符号重罚。"""
    cjk = sum("\u3400" <= char <= "\u9fff" for char in text)      # 中文字符数
    common = sum(char in "的一是在不了有和人这中大为上个国..." for char in text)  # 高频字加权
    private_use = sum("\ue000" <= char <= "\uf8ff" for char in text)  # 私有区≈乱码
    replacement = text.count("\ufffd")                            # 替换符≈解码失败
    ...
    utf_bonus = 8 if encoding.startswith("utf-8") else 0          # 同分时偏向 UTF-8
    return cjk + common * 2 + utf_bonus - private_use * 8 - kana * 4 - replacement * 20 - controls * 10
```

替代方案是 `chardet` 这类统计探测库，但它偶尔给出"自信的错误答案"且多一个依赖；
这里的打分器只服务"中文小说"这一种文本，40 行代码、完全可控、错了也知道为什么错。

### 3.4 清洗与章节感知切分

清洗（`backend/app/services/novel_service.py:61-78`）处理网页小说的典型残留：

```python
def clean_novel_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")   # 全角→半角、兼容字符归一
    text = _INVISIBLE_RE.sub("", text).replace("\u00a0", " ")  # 零宽字符、不间断空格
    text = text.replace("\r\n", "\n").replace("\r", "\n")      # Windows/Mac 换行统一
    text = _CONTROL_RE.sub("", text)                   # 控制字符剔除
    lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    ...                                                # 连续空行压成一个，保留段落边界
```

注意**保留**了 `\n` 和空行——它们是后面识别"整行章节标题"的依据，不能像普通空白一样压掉。

切分主函数 `split_novel_documents`（`novel_service.py:156-274`）的策略是**先按章节切大段，再在大段内细切**：

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.novel_chunk_size,        # 目标 650 字
    chunk_overlap=settings.novel_chunk_overlap,  # 相邻块重叠 120 字，防止句子被切断丢失语义
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],  # 优先按段落切，实在不行按句/字
    add_start_index=True,                        # 在 metadata 记录块在原文中的起始偏移
)
```

"Recursive"的含义：先试最重要的分隔符 `\n\n`（段落），切出来的还超长就降级用 `\n`，
再不行用句号、逗号……像剥洋葱一样层层细化。这比"每 500 字硬切"好得多——绝不把一句话腰斩。

章节识别是**三级策略**（`novel_service.py:189-202`）：

```python
strict_count = sum(len(items) for items in strict_by_doc)   # 整行标题命中数
if strict_count:
    parser_mode = "strict"                                   # ① 整行就是标题（最可靠）
else:
    fallback_by_doc = [_find_inline_headings(text) for text, _, _ in prepared]
    if fallback_count >= 2:
        parser_mode = "inline_fallback"                      # ② 正文段首出现"第N章"（保守回退）
    else:
        parser_mode = "none"                                 # ③ 放弃，全部归入"未分章"
```

每个块的元数据（`novel_service.py:243-256`）就是第 2 章 `Embedding` 表那些列的来源：

```python
doc.metadata.update({
    "domain": "novel",
    "source": filename,          # 文件名 → embeddings.source
    "file_id": file_id,          # → embeddings.file_id
    "chapter": chapter,          # 章节标题 → chapter
    "chapter_no": chapter_no,    # 章节序号 → chapter_no
    "chunk_no": chunk_no,        # 全书递增片段号 → chunk_no
    "chapter_chunk_no": chapter_chunk_counts[counter_key],  # 章节内片段号
    "char_start": char_start,    # 字符区间，TXT 没有"页"就用它定位
    "char_end": char_start + len(doc.page_content),
    ...
})
```

另有第三级兜底：手动重新索引时若确定性解析质量不达标，会把候选行交给 LLM 产出固定格式的
章节规则 DSL，程序验证通过才应用（`services/chapter_detection.py`），并且规则按"原文哈希+解析器版本+
模型+Prompt 版本"缓存。**模型永远不直接改数据**——它只提建议，代码做裁判。

### 3.5 原子替换与索引状态机

向量入库有两条路（`backend/app/core/rag.py:77-98`）：

```python
async def add_documents(chunks, user_id=None):
    """首次入库：算向量 + 追加。"""
    vectors = await _embed_documents_batched(chunks)      # 分批算，控内存峰值
    async with AsyncSessionLocal() as session:
        session.add_all(_rows_for_documents(chunks, vectors, owner))
        await session.commit()

async def replace_documents(file_id, chunks, user_id=None):
    """重建索引：新向量全部生成成功后，单个事务里"删旧+插新"。"""
    vectors = await _embed_documents_batched(chunks) if chunks else []
    rows = _rows_for_documents(chunks, vectors, owner)    # 注意：先在内存备好全部行
    async with AsyncSessionLocal() as session:
        async with session.begin():                       # 一个事务
            await session.execute(delete(Embedding).where(Embedding.file_id == file_id, ...))
            session.add_all(rows)                         # 成功一起提交，失败一起回滚
```

为什么特意做 `replace_documents`？重新索引（换了 embedding 模型/分块参数）如果"边删边插"，
中途失败用户就丢了整本书的索引。**先在内存备齐所有向量，再一个事务原子替换**——失败时旧索引完好，
用户最多得到一条警告，不会失去服务。这呼应了 `kb_service.py:427` 的注释：
"Embedding 全部成功后才替换旧索引，重建失败时旧数据仍可检索"。

外层 `run_indexing`（`kb_service.py:340-494`）是完整的后台状态机，骨架如下：

```python
async def run_indexing(file_id, filename, ext, user_id=None, use_llm_chapter_detection=False):
    lease_id = uuid.uuid4().hex                          # 本轮任务的"工作证"
    ...
    # 拿租约：防止两个进程同时索引同一本书
    record.status = "indexing"
    record.attempts = (record.attempts or 0) + 1         # 重试计数
    record.lease_id = lease_id
    record.lease_until = now + timedelta(seconds=LEASE_SECONDS)   # 15 分钟租约
    await session.commit()
    ...
    try:
        loaded = await asyncio.to_thread(_load_novel, ...)      # CPU/IO 密集 → 线程池
        split_result = await asyncio.to_thread(split_novel_documents, ...)
        ...
        await replace_documents(file_id, split_result.documents, user_id=owner)
    except Exception as exc:
        failure = _format_index_error(exc, save_path)    # 提取根因，写库展示
    # 收尾：租约匹配才允许写结果（旧任务不能覆盖新任务的状态）
    if not record or record.lease_id != lease_id:
        return
    if split_result is not None and failure is None:
        record.status = "indexed"                        # 成功
    elif previous_chunks:
        record.status = "indexed"                        # 重建失败但旧索引还在 → 继续可用
        record.index_warning = "索引失败，旧索引仍可使用"
    else:
        record.status = "failed"                         # 从未成功过 → 真失败
```

**租约（lease）**类比：图书馆的讨论室预订。拿到 15 分钟预订条（lease_id + lease_until）才能用房间；
超时未续，管理员（`recover_stale_indexing`，进程重启时调用）回收并重新派任务。
每个进度写回函数（`kb_service.py:246-283`）都会重新核对 `lease_id`，
保证崩溃恢复的旧任务不会覆盖新一轮任务的进度。

### 3.6 为什么这样写

- **为什么记录 `attempts` 并设上限 3？** 后台任务最怕"无限失败循环"：网络差 → 失败 → 重启重试 → 又失败。
  上限 + 旧索引降级保护，让最坏结果可控。
- **为什么章节识别要"确定性优先、LLM 兜底"？** LLM 有随机性，直接让它切块会导致同一本书
  每次索引结果不同（评测、缓存全部失效）。确定性规则先行，LLM 只在指标异常时介入且产出受验证的规则——
  这是"用模型但不受制于模型"的典型设计。
- **坑**：改了 `NOVEL_CHUNK_SIZE` 或 `EMBEDDING_MODEL` 后，旧文件不会自动重索引。
  检索前有版本守卫（`rag.py:344-367` 的 `_check_index_compatibility`）：
  发现文件记录的 embedding 模型/维度/分块参数与当前配置不一致，直接抛错提示"请重新索引"，
  而不是拿两种口径混着算分。

**本章小结**：上传即返回，后台状态机负责"读文件 → 清洗 → 找章节 → 细切块 → 算向量 → 原子替换"，
租约和重试上限保证失败可控。下一章开始，我们用这些卡片回答问题——先从最直观的向量检索说起。

---

## 第 4 章 检索（一）：向量通道与多租户隔离

### 4.1 新概念：Top-K 检索与"当前用户"上下文

检索的第一步：把用户问题也算成向量，然后在 `embeddings` 表里找余弦距离最小的 K 条。
但在此之前要解决一个前置问题——**每个 SQL 都要知道"现在是谁在问"**。

### 4.2 新概念：ContextVar——异步世界的"请求随身口袋"

**类比**：餐厅服务员（请求）带着一张写有桌号的点单卡在厨房里流转。厨师（业务函数）不需要
服务员每喊一句菜名都重复桌号——看一眼手里的卡就行。这张卡就是 `contextvars.ContextVar`：
每个异步请求有自己独立的副本，互不串。

最小示例：

```python
from contextvars import ContextVar
request_id: ContextVar[str] = ContextVar("request_id", default="anonymous")

request_id.set("req-42")     # 进入请求时设置
print(request_id.get())      # 业务代码任意深处读取 → "req-42"
token = request_id.set("req-43")
request_id.reset(token)      # 离开时恢复旧值，防止泄漏到下一个请求
```

项目实现只有 30 行（`backend/app/core/context.py:15-30`）：

```python
_current_user: ContextVar[str] = ContextVar("current_user", default=settings.default_user)

def get_current_user() -> str:        # 业务代码随处调用
    return _current_user.get()

def set_current_user(user_id: str):   # 鉴权中间件在请求入口调用（第 9 章）
    return _current_user.set(user_id)

def reset_current_user(token) -> None:
    _current_user.reset(token)
```

**为什么不用函数参数一路传 user_id？** `rag.py` 里几十个函数、`kb_service` 里几十个函数，
如果每个签名都加 `user_id` 参数，任何一处漏传都是租户泄露事故。ContextVar 让"谁在请求"
成为隐式环境的一部分，写错的地方反而更少。
（代价：隐式依赖可读性差一点，所以文件 docstring 里专门写了约定。）

### 4.3 向量检索本体

`backend/app/core/rag.py:147-156`，全文仅 10 行：

```python
async def _vector_search(session, qvec, k, filter_source=None, filter_user=None,
                         filter_domain=None, filter_file_id=None):
    """执行 pgvector 余弦距离检索并返回原始相似度。"""
    distance = Embedding.embedding.cosine_distance(qvec)   # 生成 <=> 距离表达式
    stmt = select(Embedding, distance.label("distance")).order_by(distance)  # 距离升序=最相似在前
    stmt = _apply_filters(stmt, filter_source, filter_user, filter_domain, filter_file_id)
    ...                                                    # 追加 WHERE 条件
    result = await session.execute(stmt.limit(k))          # 只要 Top-K
    # 距离转相似度：score = 1 - distance，负值钳到 0
    return [(row, max(0.0, 1.0 - float(distance_value))) for row, distance_value in result.all()]
```

翻译成 SQL 大致是：

```sql
SELECT *, embedding <=> :query_vector AS distance
FROM embeddings
WHERE user_id = :owner AND domain = 'novel' AND file_id = :file_id
ORDER BY distance
LIMIT :k;
```

`_apply_filters`（`rag.py:134-144`）是所有检索通道共用的过滤装配器——**四个通道
（向量/FTS/中文词法/BM25）都强制过它**，这就是"行级租户隔离"的落地点。
查询条件缺省时跳过对应 WHERE，函数签名用 `None` 表达"不过滤"而不是空字符串，避免歧义。

### 4.4 为什么这样写

- **为什么 `score = 1 - distance` 还要 `max(0.0, ...)`？** 余弦距离范围是 [0, 2]（夹角可大于 90°），
  完全相反的向量距离为 2，`1 - 2 = -1`。分数统一钳到 [0,1]，后续融合（第 5 章）才不会出现负分捣乱。
- **为什么手写 SQL 而不用 LangChain 自带的 PGVectorRetriever？** 自带封装难以塞进
  user_id/domain/file_id 过滤和 HNSW 调优，而本项目恰恰需要"同一条 SQL 里做向量 + 多条件过滤"。
  自己写 10 行换来完全的控制权。`rag.py:836-847` 保留了一个实现 LangChain 接口的
  `_PGVectorRetriever` 薄壳，供需要 LangChain 生态的场景使用。
- **性能**：`ORDER BY distance LIMIT k` 走的正是第 2 章建的 HNSW 索引；
  没有索引时这条语句是全表扫描。

**本章小结**：向量通道 = "问题向量化 → 单条 SQL（相似度排序 + 租户过滤 + Top-K）"。
但向量检索有个天生短板——它按"意思"找，人名、绰号这种**字面精确匹配**反而是弱项。
下一章请出词法通道补位。

---

## 第 5 章 检索（二）：词法通道与两路融合

### 5.1 新概念：词法检索——按"字面"而非"意思"找

**类比**：向量检索像"听起来像什么就找什么"（"大圣"能匹配"孙悟空"）；
词法检索像 Ctrl+F——字面不出现就找不到，但命中的一定含那个词。
向量会漏"冷门绰号的精确所指"，词法会漏"同义改写"，所以要两条腿走路。

PostgreSQL 生态里本项目先后用了三种词法通道，按演进顺序：

| 通道 | 原理 | 状态 |
| --- | --- | --- |
| `_fts_search` | 内置全文检索（tsvector 生成列 + GIN 索引） | 永远的兜底 |
| `_chinese_lexical_search` | jieba 分词 + ILIKE 词覆盖 + trigram 相似度 | 可配置开启 |
| `_bm25_search` | ParadeDB pg_search 扩展，真 BM25 打分 | **默认开启，fail-fast** |

### 5.2 中文词法的关键：先分词

中文没有空格，`to_tsvector('simple', ...)` 只会把整句当成一个大 token，什么都匹配不上。
`rag.py:172-198` 的 `_tokenize_chinese_query` 负责把问题拆成词：

```python
def _tokenize_chinese_query(query: str, max_terms=None) -> list[str]:
    limit = max_terms or settings.chinese_lexical_max_terms     # 最多取 8 个词
    anchor_text = query
    for phrase in sorted(_CHINESE_STOPWORDS, key=len, reverse=True):  # 先去掉"什么/为什么/人物"等
        anchor_text = anchor_text.replace(phrase, " ")
    anchor_text = re.sub(r"[和与及跟的在对把将由从向于、，。！？；：\s]+", " ", anchor_text)
    anchors = re.findall(r"[\u3400-\u9fff]{2,6}|[A-Za-z0-9_]{2,}", anchor_text)  # 正则兜底词
    try:
        import jieba
        candidates = anchors + jieba.lcut(query, cut_all=False)  # jieba 精确模式分词
    except ImportError:
        candidates = anchors + _FALLBACK_TOKEN_RE.findall(query) # 没 jieba 也不崩
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        token = raw.strip().lower()
        if len(token) < 2 or token in _CHINESE_STOPWORDS or token in seen:
            continue                       # 过滤单字、停用词、重复
        ...
        tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens
```

细节值得学：`anchors`（正则切的 2~6 字片段）和 `jieba` 结果**取并集**——jieba 偶尔会把人名
"林黛玉"切成"林/黛玉"，正则片段保证人名完整形态也在候选里。

### 5.3 三条通道的实现要点

FTS 兜底通道（`rag.py:159-169`）：

```python
tsquery = func.plainto_tsquery("simple", query)              # 问题 → 查询词袋
rank = func.ts_rank(Embedding.search_vector, tsquery)        # 词袋命中打分
stmt = select(Embedding, rank.label("rank")).where(
    Embedding.search_vector.op("@@")(tsquery)                # @@ 是 tsvector 匹配运算符
)
...
return [(row, min(1.0, float(rank_value) / 0.1)) for row, rank_value in result.all()]
```

中文词法通道（`rag.py:200-221`）核心是**词覆盖率**打分：

```python
for token in tokens:
    condition = Embedding.content.ilike(f"%{escaped}%", escape="\\")  # 内容包含该词？
    conditions.append(condition)
    coverage_parts.append(case((condition, 1.0), else_=0.0))          # 命中记 1 分
coverage = sum(coverage_parts) / float(len(tokens))                   # 命中比例，如 3/5=0.6
trigram = func.least(1.0, func.word_similarity(query, Embedding.content))
rank = coverage * 0.85 + trigram * 0.15          # 覆盖为主，字符三元组相似为辅
```

BM25 通道（`rag.py:234-271`）是评测驱动的升级（见 `docker-compose.yml:3-6` 的注释：
spike 实测候选召回 0.575 > ILIKE 0.525，延迟好约 90 倍）。它用原生 SQL：

```python
stmt = sql_text(
    f"SELECT embeddings.*, paradedb.score(embeddings.id) AS bm25_score "
    f"FROM embeddings WHERE embeddings.content @@@ :bm25_query ... "
    f"ORDER BY bm25_score DESC LIMIT :k"
)
```

BM25 是信息检索的经典算法：词频越高、词越稀有、文档越短，分越高——比"数命中词个数"聪明得多。

### 5.4 两路怎么合并：RRF 与 weighted

两通道分数**量纲不同**（余弦相似 vs BM25 无上限分），直接相加没有意义。项目给了两种融合器。

RRF（Reciprocal Rank Fusion，倒数排名融合，`rag.py:274-284`）——只看**名次**不看分数：

```python
def _rrf_fuse(vector_results, lexical_results, k, rrf_c: int = 60):
    """用 Reciprocal Rank Fusion 合并向量和词法候选。"""
    scores: dict[str, float] = {}
    for results in (vector_results, lexical_results):
        for rank_index, (row, _) in enumerate(results):      # rank_index 从 0 开始
            scores[row.id] = scores.get(row.id, 0.0) + 1.0 / (rrf_c + rank_index + 1)
            # 向量第 1 名贡献 1/61，第 2 名 1/62…… 名次越靠前贡献越大
    fused.sort(key=lambda item: item[1], reverse=True)
    return fused[:k]
```

RRF 的妙处：`1/(60+rank)` 曲线非常平缓，某通道第 1 名和第 2 名的差距被压缩，
**两通道都靠前的候选稳赢，单通道的"爆款"不能一票定音**。常数 60 是论文经验值，
越大排名差距越平滑。

weighted 融合（`rag.py:298-319`）——先归一再加权，让**分数本身**参与排序：

```python
def _weighted_fuse(vector_results, lexical_results, vector_weight, lexical_weight):
    """各通道原始分数池内 min-max 归一后加权求和。

    RRF 只用名次，会丢弃"两通道都把某候选排第 1"和"险排第 N"的差距；
    评测已定位瓶颈在池内排序，此模式让通道相关度参与排序。
    单通道候选在缺失通道计 0 分——被双通道同时召回的候选天然占优。
    """
    norm_vector = _minmax_normalize(vector_scores)     # 池内线性映射到 [0,1]
    norm_lexical = _minmax_normalize(lexical_scores)
    fused = [
        (row, vector_weight * norm_vector.get(row_id, 0.0) + lexical_weight * norm_lexical.get(row_id, 0.0))
        for row_id, row in rows_by_id.items()
    ]
```

两种模式由 `FUSION_MODE` 配置切换（`config.py:105-107`），是典型的 **A/B 开关**设计：
新策略先实现、默认关闭，评测证明收益后再切默认值。

### 5.5 串联：`_similarity_search_once` 主流程

`rag.py:403-595` 是检索的总装配线（此处摘骨架）：

```python
pool_n = (max(k, settings.reranker_candidate_n) if settings.enable_reranker
          else max(k, settings.hybrid_candidate_k) if settings.enable_hybrid_search else k)
# 先取足够大的候选池，留给去重/重排/扩展用，而不是一开始就砍到 Top-K

if settings.enable_hybrid_search:
    vector_results = await _vector_search(session, qvec, candidate_k, ...)   # 通道一
    if settings.enable_bm25_search and _CJK_RE.search(query):
        # fail-fast：BM25 开启即视为该环境已具备 pg_search 扩展 + bm25 索引。
        # 异常直接抛出，不静默降级——缺扩展/索引是必须修复的配置错误，
        # 静默回退会让词法通道悄悄退化且无人察觉。
        lexical_results = await _bm25_search(session, query, candidate_k, ...)
    else:
        try:
            ..._chinese_lexical_search(...) / ..._fts_search(...)
        except Exception as exc:                     # pg_trgm 不可用 → 回退 simple FTS
            await session.rollback()
            log.warning("lexical_search.fallback", error=str(exc))
            lexical_results = await _fts_search(...)
    rrf_fused_all = _rrf_fuse(vector_results, lexical_results, candidate_k, settings.rrf_k)
    fused_all = (_weighted_fuse(...) if settings.fusion_mode == "weighted" else rrf_fused_all)
    fused = fused_all[:pool_n]
    for row, fused_score in fused:                   # 逐条补齐分数元数据
        meta.update({
            "score": ..., "score_type": "hybrid" if vector_score is not None and lexical_score is not None else ...,
            "vector_score": ..., "fts_score": ..., "rrf_score": ..., "fusion_mode": ...,
        })
        pool.append(Document(page_content=row.content, metadata=meta))
```

注意两种**不同的失败策略**：

- BM25 开着却缺扩展 → **直接抛错**。因为这是环境配置错误，静默回退会让检索质量悄悄劣化、无人察觉
  （这正是用户确立的"fail-fast over silent fallback"原则）。
- `pg_trgm` 缺失 → **降级到 FTS 并打 warning 日志**。因为 simple FTS 仍是可用的合理兜底，
  中文词法只是"更好的替代品"，没有它系统依然正确。

"什么该崩、什么该降"不是教条，而是看**降级后结果是否仍然正确可接受**。

### 5.6 为什么这样写

- **为什么两路各取 60 个候选再融合，而不是直接各取 Top-8？** 融合需要"允许一路失手"：
  向量漏掉的冷门词法命中可能在第 30 名，池子太小它永远进不了决赛。
  `hybrid_candidate_k=60` 就是给重排（下一章）备足决赛选手。
- **为什么 trace 里要保留完整候选池？**（`rag.py:488-503` 的注释）离线评测需要回答
  "金标到底是被哪一步丢掉的"——是通道没召回，还是融合/重排挤掉的。没有全池快照，这类归因无从谈起。

**本章小结**：向量管语义、词法管字面；RRF/weighted 负责公平合并；失败策略分"必须崩"和"可以降"。
现在候选池有了且排序尚可，但"尚可"不够——下一章请出重排器精修名次。

---

## 第 6 章 检索（三）：cross-encoder 重排与邻居上下文

### 6.1 新概念：bi-encoder vs cross-encoder

**类比招聘**：
- 向量模型是 **bi-encoder**（双塔）：提前把每份简历（文档）做成一张摘要卡（向量），
  面试时只拿你的 JD 和卡片比——快，但信息压缩有损耗。
- 重排模型是 **cross-encoder**（交叉）：把 JD 和每份简历**逐字拼在一起**送进模型细读——
  精准得多，但每份都要现场读一遍，慢。

所以工程上永远是"bi-encoder 海选几百份 → cross-encoder 终面 Top 几十"，
这正是本项目 `reranker_candidate_n → Top-K` 的两级漏斗。

### 6.2 加载：健康检查不是可选项

`backend/app/core/rerank.py:25-55`：

```python
def get_reranker():
    """懒加载 CrossEncoder 单例，首次加载后执行一次健康检查。"""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        try:
            model = CrossEncoder(settings.reranker_model)    # 下载/加载（约 400MB）
        except Exception as exc:
            raise RerankerUnavailable(f"重排模型加载失败: ...") from exc

        # warmup：用最小输入触发一次真实前向
        try:
            scores = model.predict([("健康检查", "健康检查")])
            if scores is None or len(list(scores)) != 1:
                raise RerankerUnavailable(f"重排模型健康检查返回异常: {scores!r}")
        except Exception as exc:
            raise RerankerUnavailable(
                "重排模型健康检查失败（HF 缓存可能损坏，请检查 tokenizer.json 等文件是否为 0 字节）: ...")
        _model = model
    return _model
```

docstring 里记录了一次惨痛教训：HF 缓存损坏（tokenizer 文件 0 字节）时**构造函数不报错**，
只有 predict 才炸——历史上导致重排 40/40 全部静默回退，评测数据被污染而长期无人察觉。
所以加载后必须真跑一次前向。`main.py:66-73` 还在启动阶段就预热它，把"首次下载"从用户请求挪到部署时。

### 6.3 重排本体：分数融合 + 保护名额

`rerank.py:75-146` 的核心逻辑（摘录）：

```python
def rerank(query: str, docs, top_k: int):
    pairs = [[query, d.page_content] for d in docs]      # 每个 (问题, 文档) 对
    logits = list(model.predict(pairs))                  # 逐对细读打分
    ...
    # sentence-transformers 新版对单输出模型已内置 sigmoid（输出在 0~1）；
    # 再套一层 sigmoid 属双重压缩——不改排序，但压扁 blend 权重的绝对量纲。
    if logits and min(logits) >= 0.0 and max(logits) <= 1.0:
        scores = [float(value) for value in logits]      # 已是概率，直接用
    else:
        scores = [_sigmoid(float(value)) for value in logits]  # 旧版返回 logit，补一层

    for d, reranker_score in zip(docs, scores):
        ...
        if settings.enable_reranker_blend:               # 三路加权：重排为主，RRF/原始分为辅
            final_score = (
                settings.reranker_weight * reranker_score        # 0.70
                + settings.rrf_weight * _normalized(rrf_score, rrf_low, rrf_high)   # 0.20
                + settings.raw_score_weight * _normalized(raw_score, raw_low, raw_high)  # 0.10
            )
        ...
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [doc for _, doc in ranked[:top_k]]
    if settings.enable_reranker_blend and settings.reranker_protect_slots and top_k > 0:
        # 保护名额：RRF 名次最靠前却被重排挤出的少量候选，强制补回
        protected = sorted(docs, key=lambda d: float(d.metadata.get("rrf_score") or 0.0), reverse=True)[...]
        for candidate in protected:
            if candidate in selected:
                continue
            candidate.metadata["reranker_protected"] = True
            if len(selected) >= top_k:
                selected[-1] = candidate          # 挤掉队尾一个名额
            ...
```

三个设计点：

1. **sigmoid 去重**（`rerank.py:88-94`）：sigmoid 是把任意实数压到 (0,1) 的函数。新旧版
   sentence-transformers 行为不同，盲目再套一层会让分数全挤在 0.9 附近，加权融合时权重失真。
   修复来自一个专门的健全性测试脚本（`scripts/check_reranker_sanity.py`）——评测文化又一次立功。
2. **为什么还要 blend 而不是全信重排分？** cross-encoder 单模型也有口味偏差；
   混入 RRF 和原始召回分相当于"三个评委取加权平均"。
3. **保护名额**：检索阶段已确认强相关的证据若被重排全挤出 Top-K，答案会失去事实边界。
   给 RRF 头部候选留 2 个"保送名额"，是对模型不完美的一种工程对冲。

### 6.4 邻居扩展：让"命中段"变成"命中上下文"

检索命中的是孤立的 650 字块，但小说情节往往跨块。`expand_novel_context`（`rag.py:749-814`）
按 `chapter_no + chunk_no` 精确取回每个主命中前后的邻居块：

```python
stmt = select(Embedding).where(
    Embedding.user_id == owner,
    Embedding.domain == "novel",
    Embedding.file_id == current_file,
    Embedding.chapter_no == chapter_no,
    Embedding.chunk_no.between(chunk_no - window, chunk_no + window),   # 前后各 window 块
)
```

邻居块会打上 `score_type: "neighbor"`、`neighbor: True` 标记（`rag.py:802-806`），
预算上限 `neighbor_budget = max(2, window * 4)`，最终按**原文顺序**排列
（`rag.py:809-813`）——传给模型的上下文像原文一样从头读到尾，而引用里仍能区分
"主命中"和"上下文补充"。

编排入口 `retrieve_novel_context`（`rag.py:817-833`）固定了顺序：
主检索 → （可选章节内二级精排）→ 邻居扩展，并且注释强调
"二级精排必须在邻居扩展之前，否则邻居片段会被二次检索打乱，评测也无法区分主命中与上下文扩展"。

### 6.5 一个反面教材：`chapter_local_refine` 的教训

`rag.py:658-746` 的 docstring 是全项目最值得细读的注释之一。这个"章节内二次精排"功能
经过 A/B 实测是**负收益**（Recall@10 从 0.325 降到 0.275），因此默认关闭。根因分析：

> 候选里主检索文档带的是 RRF 融合序，而本函数对所有候选按 `score` 排序——该字段在主检索文档上
> 是纯向量余弦分。于是 RRF 融合序被整体丢弃，退化成纯向量检索，词法通道的贡献被抹掉。
> **真正的瓶颈不在这里**：候选池已含金标的比例是 0.575，最终 Top-10 只留下 0.325——
> 损失发生在池内排序，而非池的召回能力。

这里有三层可迁移的经验：
1. **先归因再优化**：用 trace 数据回答"损失发生在哪一步"，而不是凭直觉加模块。
2. **量纲必须统一**：不同来源的分数直接比较前要归一（第 5 章的 `_minmax_normalize` 就是为此而生）。
3. **负结果也要留档**：功能不删、留 docstring 记录实验数据，防止后人重蹈覆辙。

**本章小结**：重排 = cross-encoder 精修名次 + 三路分数融合 + 保护名额；邻居扩展补上下文；
还有一条用真实 A/B 数据换来的教训：改进前先定位瓶颈。
检索链路至此完整。但用户的问题往往是"他后来为什么走了"——"他"是谁？下一章处理这个。

---

## 第 7 章 Query Preparation：问句改写与语义路由

### 7.1 新概念：指代消解与"一次 LLM 调用干三件事"

多轮对话里用户说"他后来为什么走了"，直接拿去检索必然失败。需要先改写成
"孙悟空后来为什么离开取经队伍"。这就是**指代消解**。

本项目把"改写 + 判断是否需要检索 + 提取输出偏好"合并为**一次** LLM 调用，
称为 Query Preparation（`backend/app/core/query_rewriter.py`）。每次多调一次 LLM
就意味着几百毫秒延迟和真金白银，能合并就合并。

### 7.2 结构化输出的校验：不信任模型

模型返回 JSON，但 JSON 可能缺字段、字段值越界、甚至编造原文没有的实体。
`_validate_payload`（`query_rewriter.py:202-297`）是全部规则，摘录最精彩的三组：

```python
# 规则一：路由自洽——需要检索就必须给检索词，不需要就不许给
if needs_retrieval and not retrieval:
    return None, "missing_retrieval_query"
if not needs_retrieval and retrieval:
    return None, "unexpected_retrieval_query"
if needs_retrieval and answer_mode != "novel_evidence":
    return None, "inconsistent_answer_mode"

# 规则二：防幻觉实体——改写结果里出现的新实体，必须在"原问题+历史+记忆"里出现过
source_text = f"{original}\n{history_text}\n{memory_text}"
missing_entities = [item for item in entities if len(item) >= 2 and item not in source_text]
if any(item in standalone for item in missing_entities):
    return None, "invented_entity"

# 规则三：防止模型把"改写任务"干成"回答任务"
if any(marker in retrieval or marker in standalone for marker in _ANSWER_MARKERS):
    return None, "answer_like"          # _ANSWER_MARKERS = ("答案：", "综上", ...)
```

任何一条不过 → `_fallback`（`query_rewriter.py:300-327`）原样返回用户输入，
`needs_retrieval=True`——**宁可白检索一次，也不能漏检**。这是全模块的安全网。

还有一个容易忽略的语义区分（prompt 里专门强调，`query_rewriter.py:348-353`）：

> - "不要展示原文，只给总结"是用户输出偏好，不是原文证据请求；
> - "不要展示原文，但回答孙悟空为什么离开"仍需要 RAG，但 output_policy 必须禁止展示原文；
> - **是否调用 RAG 与是否展示原文是两个独立决策**。

"要不要检索"（`needs_retrieval`）决定证据从哪来；"要不要展示"（`output_policy`）决定答案长什么样。
把它们揉成一个开关是很多 RAG 产品的经典 bug。

### 7.3 路由：LLM 说不用检索，就真不检索吗？

改写结果作为 `routing_hint` 传给 Agent 的路由节点，`backend/app/agent/router.py:92-149`
的 `_routing_choice` 决定最终答案。核心是**保守兜底**：

```python
llm_needs = bool(routing_hint["needs_retrieval"])
confidence = float(routing_hint["confidence"])
reasons: list[str] = []
if confidence < settings.query_routing_confidence_threshold:   # 置信度 < 0.75
    reasons.append("low_confidence")
if _strong_novel_signal(f"{query} ..."):                       # 问句里有强小说信号词
    reasons.append("strong_novel_signal")
if mode == "novel_evidence":
    reasons.append("novel_evidence_mode")
if llm_needs:
    return True, ...                                           # LLM 说要 → 要
if reasons:                                                    # LLM 说过不要 → 但有兜底信号，强制要
    return True, "forced_by_" + reasons[0], "novel_evidence", ...
return False, ...                                              # 都干净 → 真不用
```

为什么这么保守？**成本不对称**：多检索一次的代价是几十毫秒；
漏检的代价是模型没证据、开始瞎编——用户体验直接归零。所以路由永远偏向"检索"。
同时 `route_query`（`router.py:181-214`）按复杂度选策略：

```python
def normalize_strategy(requested_strategy, query):
    value = (requested_strategy or "auto").strip().lower()
    if value == "auto":
        # 长度≥32 或 命中≥2 个小说信号词 → 复杂问题 → 多专家
        return Strategy.MULTI_EXPERT if is_complex_query(query) else Strategy.DIRECT
    return _STRATEGY_ALIASES.get(value, Strategy.DIRECT)      # 未知值回退 direct
```

### 7.4 为什么这样写

- **为什么校验而不是直接信 JSON？** LLM 输出本质是"概率性的文本"，任何进入业务逻辑的结构
  都必须过白名单校验（intent 枚举、confidence 范围、字符串长度上限……）。
  `_validate_payload` 返回 `(None, 失败原因)` 而不是抛异常——校验失败是**预期内**的常规事件，
  有专门的降级路径，不值得打断调用栈。
- **替代方案**：OpenAI 的 JSON mode / function calling 可以约束格式，但约束不了**语义**
  （编造实体、答非所问的改写依然可能出现），所以服务端校验不可省。
- **坑**：改 prompt 时注意 `QUERY_REWRITE_PROMPT_VERSION`（`config.py:132`）这类版本号——
  评测体系用它们区分"不同 prompt 下的行为"，忘了升版本会污染对比实验。

**本章小结**：一次 LLM 调用完成改写/路由/偏好提取；输出先过三道校验再进业务；
路由永远保守偏向检索。问题和策略都定了，下一章进入全项目最复杂的部分——Agent 编排。

---

## 第 8 章 Agent 编排：LangGraph 多专家流水线

### 8.1 新概念：LangGraph 与状态图

前面所有逻辑都是"函数 A 调函数 B"的直线。但 Agent 的执行是**分支流程图**：
路由后可能走 direct（2 步）、multi_expert（5 步带纠偏回环）、react……
手写 if-else 嵌套会迅速失控。

**LangGraph** 把流程显式建成一张图：**节点**是 async 函数（读一个共享 State、返回增量更新），
**边**定义"下一步去哪"，**条件边**按 State 字段分流。类比工厂流水线：State 是传送带上的工件箱，
节点是工位，条件边是岔路机的传感器。

### 8.2 共享状态：`AgentState`

`backend/app/agent/types.py:42-87`，一个 TypedDict 声明所有节点共享的字段（摘录）：

```python
class AgentState(TypedDict, total=False):
    """LangGraph 在各节点之间传递的共享状态。字段允许按执行路径逐步填充。"""
    query: str                       # 交给图的独立问题（已消解指代）
    original_query: str              # 用户原始输入
    needs_retrieval: bool            # 路由结论：要不要检索
    strategy: str                    # direct / multi_expert / react / plan_execute
    plan: list[dict[str, Any]]       # 展示用执行计划
    evidence: list[dict[str, Any]]   # 共享证据（所有专家共用！）
    sources: list[dict[str, Any]]    # 引用来源（给前端的 [S#] 清单）
    expert_tasks: dict[str, dict]    # 四专家任务
    reports: list[dict[str, Any]]    # 专家报告
    report_validation: dict[str, dict]  # 契约校验结果
    refine_agents: list[str]         # 需要纠偏的专家
    answer: str                      # 最终答案
    event_queue: Any                 # SSE 事件队列（贯穿全程的"直播线路"）
```

`total=False` 表示字段可缺省——不同策略只填自己用的字段。

### 8.3 图的定义：一张图看懂全流程

`backend/app/agent/runtime.py:718-756`：

```python
def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("route", _route_node)              # 路由：策略 + 是否检索
    graph.add_node("plan", _plan_node)                # 生成可展示的执行计划
    graph.add_node("retrieve", _retrieve_node)        # 共享检索（只跑一次！）
    graph.add_node("dispatch", _dispatch_node)        # 拆四个专家任务
    graph.add_node("experts", _experts_node)          # 四专家并发分析
    graph.add_node("validate_reports", _validate_reports_node)  # 契约/重复校验
    graph.add_node("refine_experts", _refine_experts_node)      # 一次纠偏
    graph.add_node("execute", _execute_node)          # react/plan_execute 的多步工具
    graph.add_node("reflect", _reflect_node)          # 检查证据是否足够
    graph.add_node("supervisor", _supervisor_node)    # 内部汇总准备
    graph.add_node("summary", _summary_node)          # 最终总结（流式）

    graph.add_edge(START, "route")
    graph.add_edge("route", "plan")
    graph.add_conditional_edges(                      # 条件边：按 needs_retrieval 分流
        "plan", _after_plan, {"retrieve": "retrieve", "supervisor": "supervisor"})
    graph.add_conditional_edges(
        "retrieve", _after_retrieve,                  # 按策略分流
        {"dispatch": "dispatch", "execute": "execute", "supervisor": "supervisor"})
    graph.add_edge("dispatch", "experts")
    graph.add_edge("experts", "validate_reports")
    graph.add_conditional_edges(
        "validate_reports", _after_validation,        # 有报告不合格 → 纠偏，否则汇总
        {"refine": "refine_experts", "supervisor": "supervisor"})
    graph.add_edge("refine_experts", "supervisor")
    graph.add_edge("execute", "reflect")
    graph.add_edge("reflect", "supervisor")
    graph.add_edge("supervisor", "summary")
    graph.add_edge("summary", END)
    return graph.compile()

agent_graph = _build_graph()      # 模块级编译一次，进程内复用
```

三条执行路径在图里一目了然：

```
direct:        route → plan → retrieve → supervisor → summary
multi_expert:  route → plan → retrieve → dispatch → experts → validate ─┬→ refine → supervisor → summary
                                                               (合格)└→ supervisor
react/plan:    route → plan → retrieve → execute → reflect → supervisor → summary
```

**为什么 retrieve 只跑一次？**（`_after_retrieve` 的注释，`runtime.py:204-212`）
四个专家如果各自调 RAG，就是四倍检索延迟 + 四倍 embedding 费用，而且四份证据不一致会互相打架。
共享证据还有个隐含好处：所有专家引用同一套 [S#] 编号，Supervisor 汇总时引用不会错位。

### 8.4 SSE 直播线路：事件队列

节点在图里执行，前端要实时看到过程。桥接方式是 `asyncio.Queue`（`runtime.py:28-31`）：

```python
async def _emit(state: AgentState, event_type: str, data: Any) -> None:
    queue = state.get("event_queue")
    if queue is not None:
        await queue.put({"type": event_type, "data": data})   # 节点只管发
```

`stream_agent_question`（`runtime.py:762-814`）把图和 HTTP 响应连起来：

```python
async def stream_agent_question(query, strategy="auto", file_id=None, ...):
    queue: asyncio.Queue = asyncio.Queue()
    initial: AgentState = {..., "event_queue": queue}

    async def run_graph() -> None:
        try:
            await agent_graph.ainvoke(initial)       # 后台任务里跑完整张图
        except Exception as exc:
            await queue.put({"type": "error", "data": {...}})
        finally:
            await queue.put(_STREAM_DONE)            # 哨兵对象：表示流结束

    graph_task = asyncio.create_task(run_graph())    # 图在后台跑
    try:
        while True:
            event = await queue.get()                # 主协程只管吐事件给 SSE
            if event is _STREAM_DONE:
                break
            yield event                              # yield 给调用方（api/chat.py）
    finally:
        # 客户端断开 SSE 时取消图任务，避免后台继续消耗模型和检索资源。
        if not graph_task.done():
            graph_task.cancel()
            await asyncio.gather(graph_task, return_exceptions=True)
```

生产者（图）与消费者（SSE）解耦，任何一边慢都不阻塞另一边；
`finally` 里的取消逻辑保证用户关掉页面后，后台不会继续烧 API 费用。

### 8.5 多专家并发：超时隔离的教科书写法

`runtime.py:343-408`，四个专家用 `asyncio` 并发跑（摘录）：

```python
task_map: dict[asyncio.Task, str] = {}
for name in names:
    ...
    task_map[asyncio.create_task(_run_specialist(contract, state, f"expert-{name}", ...))] = name
    # create_task 立即返回，四个 LLM 请求同时飞出去

# 统一超时等待：done=已完成的，pending=到点还没完的
done, pending = await asyncio.wait(task_map, timeout=settings.agent_multi_expert_timeout)
for task in done:
    reports[task_map[task]] = task.result()          # 成功的收结果
for task in pending:
    name = task_map[task]
    task.cancel()                                    # 未完成的取消
    reports[name] = {..., "status": "timeout", ...}  # 记为 timeout 而不是拖垮整轮
if pending:
    await asyncio.gather(*pending, return_exceptions=True)  # 等取消落地，避免孤儿任务
```

注意 `_run_specialist`（`runtime.py:269-340`）内部把每个 token 同时做两件事：
`parts.append(token)` 攒成完整报告 + `await _emit(state, "tool_token", {..., "delta": token})`
实时推给前端。**专家分析是流式展示的**——用户在等最终答案时能看到四个专家各自的思考在实时滚动，
这是体验上的关键设计（`types.py:17-19` 的注释解释了为什么 `show_agent_details` 默认 True）。

单个专家抛异常只影响自己（返回 `status="error"` 的报告），四个全挂才置
`fallback_reason="all_experts_failed"`（`_experts_node`，`runtime.py:411-424`）。

### 8.6 专家契约与报告校验

四个专家的职责边界是**写死的契约**（`backend/app/agent/contracts.py:18-51`，摘录人物专家）：

```python
"character": SpecialistContract(
    name="character",
    label="人物关系专家",
    focus=("人物身份与称谓", "关系边与双方立场", "人物动机", "关系变化阶段", "事实与推断的区分"),
    forbidden=("完整复述故事情节", "输出章节定位总表", "替代时间线专家排列全部事件"),
    output_format="Markdown 引用块；依次输出：关系边、变化阶段、事实与推断、证据不足。",
    required_groups=(("关系", "立场", "称谓"), ("动机", "变化", "阶段"), ("事实", "推断", "证据不足")),
),
```

`focus/forbidden/output_format` 进 prompt 约束模型；`required_groups` 给**程序校验**用——
报告必须各命中至少一个关键词组，否则判"契约不合格"（`agent/validation.py:31-67`）。

报告之间的**重复度检测**用 3-gram Jaccard（`validation.py:15-28`）：

```python
def char_ngrams(text: str, size: int = 3) -> set[str]:
    """归一化后切成字符 3-gram。"""
    normalized = _normalize(text)                    # 去引用符、去标点、转小写
    return {normalized[i:i+size] for i in range(len(normalized) - size + 1)}

def report_similarity(left: str, right: str) -> float:
    a, b = char_ngrams(left), char_ngrams(right)
    return round(len(a & b) / len(a | b), 4)         # 交集/并集 ∈ [0,1]
```

Jaccard = 两个集合的交集除以并集。用字符 3-gram 而不是整词，是因为中文分词本身有歧义，
字符片段是更稳定的相似性信号。相似度 ≥ 0.72（`AGENT_REPORT_SIMILARITY_THRESHOLD`）的两份报告
会被标记，**契约分低的一方**进入纠偏（`validation.py:87-107`）。

纠偏只做**一次**（`_refine_experts_node`，`runtime.py:452-498`）：把校验结果和原报告喂回去重写，
再不合格就标 `invalid` 丢弃。为什么只一次？专家重写很贵，而 Supervisor 本来就会去重汇总——
流程要为"大部分情况一次就好"优化，而不是为极端情况无限重试。

### 8.7 工具注册表：权限、超时、安全的计算器

`backend/app/agent/tools.py:42-67` 的 `ToolRegistry.execute` 是所有工具调用的统一闸门：

```python
async def execute(self, name, *, allowed_tools, **kwargs):
    if name not in self._specs or name not in allowed_tools:
        return ToolResult(status="denied", error_code="tool_not_allowed", tool=name)
        # 双重白名单：注册过 ≠ 允许用。路由节点给每个策略分配 allowed_tools（router.py:194-214）
    try:
        result = await asyncio.wait_for(self._handlers[name](**kwargs), timeout=spec.timeout_seconds)
    except asyncio.TimeoutError:
        return ToolResult(status="timeout", ...)     # 超时不抛异常，返回结构化结果
    except Exception as exc:
        return ToolResult(status="error", ...)       # 任何异常都变成 ToolResult
```

路由节点把 `retrieve_novel`、`get_chapter_context`、`calculator` 写进各策略的
`allowed_tools` 元组（`router.py:194/211`）——工具权限跟着策略走，而非全局开放。

计算器工具是"最小权限"的好例子（`tools.py:137-150`）：模型要算数时不让它写代码，
而是解析成 AST、只允许白名单运算符、指数超过 8 直接拒绝：

```python
_ALLOWED_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ...}

def _safe_eval(node: ast.AST) -> float:
    """递归计算受限 AST 表达式；不执行变量、函数调用或任意 Python 代码。"""
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        ...
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("exponent_too_large")   # 防 2**999999 打爆内存
    raise ValueError("unsupported_expression")
```

**为什么不用 `eval()`？** 模型输出是不可信输入，`eval` 等于让外部输入直接在服务端执行代码，
是安全红线。AST 白名单求值 20 行代码换来绝对可控。

### 8.8 最终汇总：流式 + 输出护栏

`_summary_node`（`runtime.py:616-715`）是所有路径的终点：

```python
async for token in _stream_llm([...], settings.agent_synthesis_max_tokens):
    parts.append(token)
    await _emit(state, "token", token)               # 一边生成一边推给前端
answer = _sanitize_summary("".join(parts), policy)   # 生成完做净化
if answer != "".join(parts):
    # 输出护栏净化改变了内容（去引用/截断引语/隐藏 [S#]）时，
    # 以完整净化稿覆盖已流式渲染的内容：流式体验与护栏语义同时成立。
    await _emit(state, "token_replace", answer)
```

`token_replace` 事件是个巧妙的设计：流式要快，所以先原样吐字；但用户设置了
"不要展示原文"时，`_sanitize_summary`（`runtime.py:602-613`）会把答案里的长引语截断、
引用块剥掉。两者冲突吗？不——先流式后覆盖，前端收到 `token_replace` 就整体替换
（第 10 章你会看到前端怎么接）。牺牲一次重绘，换来"快"和"合规"兼得。

证据为空时（`runtime.py:630-632`）直接返回固定话术 `_EMPTY_MESSAGE`——
"检索不到就不编"，这是 RAG 系统最重要的诚实性兜底。

**本章小结**：一张 LangGraph 图承载三种策略；共享检索、共享证据；
专家契约 + 程序校验 + 一次纠偏控制质量；队列解耦执行与直播；
工具闸门统一权限和超时。下一章看这些事件如何变成 HTTP 响应。

---

## 第 9 章 API 层：用 SSE 把思考过程"直播"出去

### 9.1 新概念：SSE——服务器单向推送

**WebSocket** 是双向长连接，功能强但重：需要协议升级、心跳、状态管理。
而"AI 逐字回答"只需要**服务器 → 浏览器**的单向推送，于是更轻的 **SSE**
（Server-Sent Events）就够了：本质是一个**不结束的 HTTP 响应**，`Content-Type: text/event-stream`，
按文本协议一段段写事件：

```
event: token
data: 林

event: token
data: 黛玉

event: done
data:
```

浏览器一边收一边渲染，用户看到"打字机效果"。

### 9.2 会话与历史：先把地基校验一遍

`backend/app/api/chat.py:154-214` 的 `_chat_stream_response` 开头做例行校验（摘录）：

```python
if not req.file_id:
    raise HTTPException(status_code=400, detail="请先从小说列表中选择当前咨询对象")

# 校验这本书属于当前用户且已索引（两处 WHERE 都带 user_id，租户隔离不留死角）
target_file = await session.execute(
    select(KnowledgeFile).where(
        KnowledgeFile.id == req.file_id,
        KnowledgeFile.user_id == user_id,
        KnowledgeFile.status == "indexed",
    ))

# 会话绑定校验：一场对话只能聊一本书，换书必须换会话
elif chat_session.file_id and chat_session.file_id != req.file_id:
    raise HTTPException(status_code=409, detail="当前会话绑定了另一部小说，请切换会话")
```

会话防串书看似小事，实则是**检索正确性**的一部分：证据过滤靠 `file_id`，
会话若能跨书，答案就会把《红楼梦》的段落引到《西游记》的问题里。

### 9.3 事件生成器：整个后端的"总导演"

`chat.py:217-378` 的 `event_gen` 是一次问答的完整编排，按时间顺序：

```python
async def event_gen():
    yield {"event": "session", "data": session_id}       # ① 先把会话 ID 发出去

    async with asyncio.timeout(settings.agent_request_timeout):   # ② 全程 240s 硬顶
        ...
        # ③ 召回记忆（摘要 + 三层记忆 + 输出偏好）——失败只告警不阻断
        memory_context = await memory_service.build_memory_context(...)
        yield _sse_event("memory_context", {...})

        # ④ Query Preparation（第 7 章）
        rewrite = await rewrite_query(req.message, history_messages, memory_context=memory_context)

        # ⑤ 用户明确表达展示偏好 → 立即落库，下一轮就生效
        if preference_update:
            await memory_service.upsert_preference(preference_update, ...)

        # ⑥ 进入 Agent 图，逐事件转发
        async for stream_event in stream_agent_question(...):
            event_type = stream_event["type"]
            if event_type == "sources":
                reply_sources = payload or []             # 引用清单（记住，落库要用）
                yield _sse_event("sources", reply_sources)
            elif event_type == "token":
                full_reply.append(token)                  # 攒最终答案
                yield _sse_event("token", token)
            elif event_type == "token_replace":
                full_reply.clear(); full_reply.append(replaced)   # 净化稿为准
                yield _sse_event("token_replace", replaced)
            ...                                           # route/plan/tool_* 原样转发

    # ⑦ 收尾：持久化答案（失败只记日志——答案已经在用户屏幕上了，别让它消失）
    reply = "".join(full_reply)
    if persist:
        assistant_message_id = await _persist_message(session_id, "assistant", reply, reply_sources)
        # ⑧ 记忆更新丢给后台任务，不占用响应
        asyncio.create_task(update_memory_background())
        yield _sse_event("memory_updated", {"status": "scheduled"})
    yield {"event": "done", "data": ""}

return EventSourceResponse(event_gen(), ping=10, ...)     # sse-starlette 负责协议细节
```

几个值得记住的取舍：

- **记忆上下文失败为什么只告警？** 没有记忆，答案质量下降但不至于错误；为了它中断问答得不偿失。
  对比第 5 章 BM25 的 fail-fast——同样是异常，处理方式取决于"降级后是否仍正确"。
- **`asyncio.timeout` 包住全程**：给"AI 问答"这类无上限的外部调用族一个硬边界，
  超时给用户明确提示（`request_timeout` 事件）而不是无限转圈。
- **每个事件处理点都检查 `request.is_disconnected()`**：客户端断开立刻 return，
  连锁触发第 8 章 `finally` 里的图任务取消，资源不泄漏。

### 9.4 鉴权中间件与用户注入

`backend/app/main.py:120-181` 的 `AuthMiddleware` 把 JWT 变成 `user_id`（摘录）：

```python
if not user_id:
    return JSONResponse(status_code=401, content={"detail": "未授权：..."})

token = set_current_user(user_id)        # 写进 ContextVar（第 4 章）
try:
    response = await call_next(request)  # 后续所有路由/业务代码都能 get_current_user()
    return response
finally:
    reset_current_user(token)            # 请求结束必须恢复，防止上下文串号
```

JWT 本体是纯标准库实现（`core/security.py:57-97`）：HMAC-SHA256 签名、恒定时间比较
（`hmac.compare_digest`，防时序攻击）、过期与类型校验。密码用 PBKDF2（26 万轮迭代 + 随机盐，
`security.py:17-28`）——不引入重型依赖也守住了安全底线。

**为什么用中间件而不是每个路由写鉴权？** 正好和第 4 章的 ContextVar 呼应：
入口一处 `set`，全链路可用；漏写一处的风险从"每个路由"缩小到"中间件本身"。

**本章小结**：SSE = 不结束的 HTTP 响应；`event_gen` 把"记忆 → 改写 → Agent 图 → 落库 → 记忆更新"
串成一条带超时、可取消的流水线；鉴权在中间件一次完成。下一章去浏览器那边接住这条流。

---

## 第 10 章 前端：用 fetch 手工解析 SSE 流

### 10.1 新概念：为什么不直接用 EventSource？

浏览器有原生 SSE 客户端 `EventSource`，但它**只支持 GET、不能带自定义 header**。
本项目需要：POST + `Authorization: Bearer` + JSON body。所以用 `fetch` + `ReadableStream`
手工解析（`frontend/src/api/client.ts:1-4` 的注释开门见山说了这个原因）。

### 10.2 SSE 解析器：缓冲、切块、分发

`client.ts:329-358` 是全文最值得逐行读的前端代码：

```typescript
const consumeSseStream = async (res: Response, handlers: StreamHandlers) => {
  const reader = res.body!.getReader()        // 拿到字节流读取器
  const decoder = new TextDecoder()
  let buffer = ''                             // 关键：网络是按"块"到的，不是按"事件"
  const dispatcher = createSseDispatcher(handlers)

  const drain = () => {                       // 把缓冲区里所有完整事件块吃干净
    const separator = /\r?\n\r?\n/            // SSE 约定：事件之间以空行分隔
    let match: RegExpMatchArray | null
    while ((match = buffer.match(separator))) {
      const block = buffer.slice(0, match.index!)     // 一个完整事件块
      buffer = buffer.slice(match.index! + match[0].length)  // 从缓冲区移除
      const parsed = parseSseBlock(block)
      if (parsed.event) dispatcher.dispatch(parsed.event, parsed.data)
    }
    // 不完整的事件块留在 buffer 里，等下一个网络块到了再拼
  }

  while (true) {
    const { done, value } = await reader.read()   // 等下一段字节
    if (done) { drain(); dispatcher.finish(); return }
    buffer += decoder.decode(value, { stream: true })  // stream:true 处理跨块的多字节字符
    drain()
  }
}
```

为什么必须缓冲？网络传输按 TCP 包分块，一个 `token` 事件的半截 JSON 可能横跨两个网络包。
不缓冲就会解析出半截数据。`decoder.decode(value, { stream: true })` 同理：
一个中文字符是 3 字节，可能被 TCP 拆开，流式解码器会记住残缺字节等下一块。

事件块解析（`client.ts:276-289`）按 SSE 规范处理 `event:` 和 `data:` 行，
多行 `data:` 用 `\n` 重连——**只剥掉冒号后至多一个前导空格**，保留 token 自身的空格和换行，
否则流式文本会"丢空格"。

事件分发（`client.ts:291-327`）是一张事件名 → 回调的路由表：

```typescript
const JSON_EVENT_HANDLERS: Record<string, keyof StreamHandlers> = {
  memory_context: 'onMemoryContext',
  route: 'onRoute',
  sources: 'onSources',
  tool_start: 'onToolStart',
  tool_token: 'onToolToken',   // 专家报告的流式 token
  ...
}
```

`token` / `token_replace` / `done` / `session` 是纯文本事件单独处理；
其余事件 `JSON.parse` 后交给对应回调。`done` 有 `completed` 标志防重复触发，
`finish()` 保证流意外断开时也回调 `onDone`，UI 不会卡在"正在思考"。

### 10.3 发起请求与 UI 状态

`streamChat`（`client.ts:360-389`）用 `AbortController` 支持取消：

```typescript
const controller = new AbortController()
fetch(BASE + '/chat', { method: 'POST', ..., signal: controller.signal })
  .then(async (res) => { await ensureSuccess(res); await consumeSseStream(res, handlers) })
  .catch((error) => { if (error.name !== 'AbortError') handlers.onError?.(error) })
return controller        // 调用方拿到 controller，stop() 时 abort()
```

`ChatPanel.vue` 的 `send`（`components/ChatPanel.vue:215-356`）把回调接进 Vue 响应式状态：

```typescript
onToken: (t) => {
  messages.value[aiIndex].content += t     // 累加到当前消息
  updateRendered(aiIndex)                  // 50ms 节流的 Markdown 重渲染
  scrollToBottom()                         // 智能跟随：仅在贴近底部时滚
},
onTokenReplace: (t) => {
  messages.value[aiIndex].content = t      // 后端净化稿整体覆盖
  updateRendered(aiIndex)
},
onToolToken: (t) => {
  const step = m.tools?.find((item) => item.id === t.id)   // 按 id 找到对应专家卡片
  if (step) step.text = (step.text || '') + (t.delta || '')  // 专家报告流式滚动
},
```

`updateRendered` 的 50ms 节流（`ChatPanel.vue:156-168`）是个小而重要的优化：
LLM 每 token 都会触发回调，逐 token 做 Markdown→HTML 转换+DOM 更新会卡；
攒 50ms 批量渲染，肉眼无感，CPU 减负大半。

滚动跟随也有讲究（`ChatPanel.vue:105-116`）：`scrollHeight - scrollTop - clientHeight < 32`
判断用户是否本就贴近底部——用户上翻阅读历史时**不**强制拉回底部，这是聊天产品的基本修养。

路由守卫（`pages/../router/index.ts:30-39`）在进入 `/chat` 前检查本地 token，
未登录跳 `/login` 并带 `redirect` 参数——登录后能回到原页面。

**本章小结**：前端不依赖任何流式库，60 行搞定 SSE 解析；关键在缓冲与节流。
至此主线全部打通。还有两条支线——记忆与评测，它们决定这个系统能否"越用越好、改得放心"。

---

## 第 11 章 记忆闭环：摘要与三层长期记忆

### 11.1 新概念：为什么要记忆？

上下文窗口有限，不可能把 100 轮对话全部塞给模型。业界的标准解法分两层：

- **摘要（summary）**：把旧对话压缩成一段话，代替原文历史。
- **长期记忆（memory）**：从对话中抽取"值得跨会话保留的事实"存起来，下次按需召回。

本项目把记忆分**三层作用域**（`db/models.py:170` 的 `memory_type`）：

| 类型 | 作用域 | 例子 |
| --- | --- | --- |
| `user_preference` | 用户全局 | "不要展示原文，只给总结" |
| `novel_fact` | 当前这本书 | "用户在关注宝玉挨打情节" |
| `session_fact` | 当前会话 | "上面说的'他'指贾政" |

### 11.2 召回：偏好永远在场，其余按相关性

`backend/app/services/memory_service.py:194-239`：

```python
async def retrieve_memories(*, query, session_id, file_id, limit=None):
    """按当前会话、当前小说、用户偏好三层作用域召回记忆。

    用户偏好不是普通语义文档：必须始终纳入上下文，避免"不要展示原文"
    因为向量相关性不足而在下一轮失效；其余记忆再按相关性排序截断。
    """
    scope = or_(                                   # 三层并集：本会话 ∪ 本书 ∪ 全局
        AgentMemory.session_id == session_id,
        AgentMemory.file_id == file_id if file_id else AgentMemory.file_id.is_(None),
        (AgentMemory.session_id.is_(None) & AgentMemory.file_id.is_(None)),
    )
    # 偏好单独查，全量带出（最多 10 条），不做相似度过滤
    preference_result = await session.execute(
        base.where(AgentMemory.memory_type == "user_preference")...)
    # 其余记忆用向量相似度排序（算过 embedding 的记忆存在 agent_memories.embedding 列）
    if query_vector:
        other_stmt = other_stmt.order_by(
            AgentMemory.embedding.cosine_distance(query_vector).asc().nullslast(),
            AgentMemory.importance.desc(), ...)    # 相似度优先，重要性次之
```

注意 scope 的语义：`novel_fact` 存库时 `session_id=None`（`maintain_conversation_memory` 里
按类型决定挂哪个作用域，`memory_service.py:383-385`），所以"本书"条件是
`file_id == 当前书`；全局偏好两个外键都是 NULL。一个 OR 条件精确表达了三层并集，
且天然带了 `user_id` 隔离（`base` 里已有）。

### 11.3 写入：抽取、去重、版本

回答完成后，后台任务 `maintain_conversation_memory`（`memory_service.py:360-415`）做三件事：

1. **抽取记忆**（`_extract_memories`，332-357 行）：一次 LLM 调用，prompt 限定
   只允许三种类型、最多 5 条、并给出负面清单（"不要保存问候、临时推断、助手自创内容"）。
   服务端再校验类型白名单和重要性下限（`memory_min_importance=0.55`），过滤模型废话。
2. **去重/更新**（`save_memory`，113-171 行）：同作用域同内容的记忆只提升 `importance`；
   同 `preference_key` 的偏好则版本号 +1 并覆盖内容——"改主意"是合法操作，留版本便于追溯。
3. **滚动摘要**：等未覆盖消息超过阈值（8 条或 5000 字，`config.py:145-146`），
   把"旧摘要 + 最近 30 条消息"交给 LLM 压缩成新摘要，记录 `covered_message_id` 水位
   （`memory_service.py:391-414`）。下次只摘要增量——像日志系统的 checkpoint。

整个函数被 `api/chat.py:350-374` 用 `asyncio.create_task` + `wait_for`（30s 超时）调度：
**记忆更新永远不阻塞用户看到答案**，失败也只是日志里一行 warning。

**为什么偏好要"立即落库"？** 对比：普通记忆可以等后台，但用户刚说完"别再贴原文了"，
如果下一轮还贴，用户会认为系统没听懂。所以 `chat.py:257-270` 在本轮请求内同步
`upsert_preference`，下一轮 `retrieve_memories` 就把它全量带出——闭环一回合内完成。

**本章小结**：记忆 = 摘要（省 token）+ 三层记忆（跨轮连贯）；偏好是"一等公民"永远注入；
写入在后台、失败可容忍。最后一章支线：评测——它让你敢改上面的一切。

---

## 第 12 章 评测体系：让每一次改进可被证明

### 12.1 新概念：为什么 RAG 必须配评测？

RAG 的每个环节（分块、embedding、融合、重排、prompt）改动都可能"感觉更好、实际更差"。
没有数字，调优就是玄学。本项目建了三套互补的评测：

| 评测 | 回答的问题 | 脚本 |
| --- | --- | --- |
| 检索评测（Recall/MRR/nDCG） | 金标片段有没有被召回、排多前 | `scripts/run_rag_eval.py`、`scripts/evaluate_rag_recall.py` |
| 答案评测（LLM-as-judge） | 最终答案是否忠实、完整、引用正确 | `scripts/evaluate_agent_answer.py` |
| 专项 spike | 单一假设的快速验证（如 BM25 值不值） | `scripts/bm25_spike.py` 等 |

### 12.2 检索指标：Recall / MRR / nDCG

金标（gold）= 人工标注的"这个问题应该命中哪些 (chapter_no, chunk_no)"。
`run_rag_eval.py:26-39` 是全部指标的实现，逐行看：

```python
def _dcg(relevances: list[int]) -> float:
    # DCG：位置越靠前权重越大（log2 折扣），命中得 1 分
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevances))

def _case_metrics(retrieved, relevant, k):
    top = retrieved[:k]
    flags = [1 if _key(item) in relevant else 0 for item in top]  # Top-K 里逐条对金标
    hits = sum(flags)
    recall = hits / len(relevant)                  # 召回率：金标被找到的比例
    precision = hits / k                           # 精确率：Top-K 里有多少是对的
    reciprocal_rank = next((1.0 / (index + 1) for index, value in enumerate(flags) if value), 0.0)
    # MRR：第一个命中的排名倒数；排第 1 得 1.0，排第 3 得 0.33，全没中得 0
    ideal = [1] * min(len(relevant), k)            # 理想排序（全部命中且靠前）
    ndcg = _dcg(flags) / _dcg(ideal) if ideal else 0.0   # nDCG：实际 DCG / 理想 DCG
    return {"recall": recall, "precision": precision, "mrr": reciprocal_rank, "ndcg": ndcg, ...}
```

四个指标各有分工：Recall 看"有没有"、MRR 看"第一个对的排多前"、
nDCG 综合看"整体排序质量"、`no_result_rate` 看"彻底空手"的比例。

防呆设计（`run_rag_eval.py:48-55`）：验收要求至少 40 条 `validated=true` 的人工标注，
不满足直接拒绝运行（`--allow-small` 才放行调试）。没有这条，"20 条样本跑出 +5%"的
噪声结论会反复误导决策。

### 12.3 冻结变量：preparation cache

A/B 对比最怕"变量不齐"：改了融合算法，但 LLM 改写结果也变了，你怎么知道是算法的功劳？
`evaluate_rag_recall.py:54-113` 引入 **preparation cache**：把 Query Preparation 的输出
（standalone_query / retrieval_query）按题固化成 JSON 快照，实验时校验快照与数据集、
file_id、原文哈希完全一致才允许复用——**LLM 成为常量，只比较检索侧变量**。
这是所有 LLM 应用做对照实验的通用技巧。

### 12.4 答案侧：LLM-as-judge + 引用落地校验

检索指标再高也不等于答案好。`evaluate_agent_answer.py` 的流程（docstring，1-16 行）：
冻结 Query → 无头运行 `agent_graph` 收集答案 → 两个判分器：

**判分器一（无 LLM，纯代码）——引用落地校验**（`evaluate_agent_answer.py:101-130`）：

```python
def citation_metrics(answer, sources, gold_chunks):
    """unknown_citations = 引用了不存在的 S#（幻觉引用）；
       gold_citation_rate = 引用中指向金标片段的比例。"""
    citations = extract_citations(answer)          # 正则抽出答案里的 [S#]
    unknown = [c for c in citations if c not in evidence_ids]
    for citation in citations:
        source = by_id.get(citation)
        if not source or source.get("neighbor"):
            continue                               # 邻居片段是上下文补充，不算证据命中
        if (source.get("chapter_no"), source.get("chunk_no")) in gold_keys:
            gold_hits += 1
```

**判分器二（LLM-as-judge）——四维 1~5 分**（`evaluate_agent_answer.py:49-63`）：

```python
class JudgeVerdict(BaseModel):
    """LLM-as-judge 的严格 JSON 结构（1~5 分制）。"""
    faithfulness: int        # 忠实性：是否只基于证据
    completeness: int        # 完整性：要点是否齐全
    relevance: int           # 相关性
    citation_support: int    # 引用支撑
    @field_validator("faithfulness", ...)
    @classmethod
    def _clamp(cls, value: int) -> int:
        return max(1, min(5, int(value)))      # 分数硬钳到 1~5，模型打 99 分也没用
```

诚实的局限声明也写在 docstring 里：judge 与被评对象同模型有自评偏差、n=20 结论看置信区间。
A/B 差值全部带**配对 bootstrap 95% CI**（对同一批题两策略的差值做有放回重采样，
估计差值的置信区间）——点值好看不算数，区间不含 0 才算。

### 12.5 评测驱动的真实案例

这个仓库的许多关键决策都有评测背书，注释里随处可见：

- **BM25 上位**（`docker-compose.yml:3-6`）：spike 实测候选召回 0.575 > ILIKE 0.525，
  延迟好约 90 倍 → 换 ParadeDB 镜像并默认开启（报告 `evals/reports/bm25_spike_20260828.md`）。
- **reranker 双重 sigmoid 修复**（`rerank.py:88-94`）：健全性测试确认新版输出恒在 [0,1]，
  再套 sigmoid 是双重压缩 → 只在越界时归一。
- **章节内二级检索下架**（`rag.py:672-694`）：A/B 全指标负收益 + 根因分析 → 默认关闭，
  数据留档。
- **weighted 融合模式**（`rag.py:300-306`）：评测定位瓶颈在池内排序 → 新模式让相关度参与排序。

**本章小结**：评测文化是这个项目最硬的软实力——金标、指标、冻结变量、bootstrap CI、
负结果留档。改任何一环之前，先问"评测会说什么"。

---

## 第 13 章 端到端执行链路串讲

把全书串起来。场景：用户登录后选择《西游记》，在输入框敲下
**"孙悟空为什么三打白骨精后被赶走？"**（当前策略 `auto`，问题超过 32 字 → 复杂）。

1. **前端**（第 10 章）：`ChatPanel.send()` → `client.ts streamChat()` POST `/api/chat`，
   带 JWT、session_id、file_id，等待 SSE。
2. **中间件**（第 9 章）：`AuthMiddleware` 解 JWT、查 users 表确认活跃 →
   `set_current_user("u_xxx")` 写入 ContextVar。
3. **入口校验**（第 9 章）：`chat.py` 确认该书属于该用户且 `status=indexed`；
   会话绑定校验；读取本会话历史消息转成 LangChain 消息。
4. **记忆召回**（第 11 章）：`build_memory_context` → 摘要 + 偏好（全量）+ 相关记忆（向量排序）。
5. **Query Preparation**（第 7 章）：一次 LLM 调用 →
   `standalone_query="孙悟空为什么三打白骨精后被唐僧赶走"`、
   `retrieval_query="孙悟空 三打白骨精 被赶走 原因"`、`intent=plot_causality`、
   `needs_retrieval=true`、校验通过。
6. **进入 LangGraph**（第 8 章）：`stream_agent_question` 启动图任务。
7. **route 节点**：`is_complex_query` 为真 → `strategy=multi_expert`，
   `allowed_tools=("retrieve_novel", "get_chapter_context")`；发 `route` 事件。
8. **plan 节点**：五步计划发 `plan` 事件 → 条件边进入 retrieve。
9. **retrieve 节点**（第 4~6 章）：
   问题向量化 → `_vector_search`（HNSW Top-60）+ `_bm25_search`（jieba 分词后 BM25 Top-60）→
   RRF 融合 → 候选池补分数元数据 → cross-encoder 重排 + 保护名额 → Top-8 主命中 →
   `expand_novel_context` 补邻居 → `_normalize_evidence` 去重并编号 [S1]~[Sn] →
   发 `observation` + `sources` 事件。
10. **dispatch 节点**：Dispatcher LLM 把问题拆成四个互斥子任务
    （3-gram 相似度 < 0.82 校验通过），发 `expert_tasks` 事件。
11. **experts 节点**：四专家并发（`asyncio.wait` 统一 45s 超时），
    各自的 `tool_token` 事件实时滚动到前端"调用过程"卡片。
12. **validate 节点**：契约校验 + 报告两两 Jaccard 相似度。假设时间线专家报告过短 →
    `refine_agents=["timeline"]`，发 `validation` 事件 → 条件边进 refine。
13. **refine 节点**：只重跑时间线专家一次（带纠偏提示与旧报告），重校验；通过则继续。
14. **supervisor 节点**：过滤无效报告，组装 `synthesis_context`（问题 + 共享证据 + 合格报告 + 摘要 + 记忆）。
15. **summary 节点**：流式生成最终答案，逐 token 发 `token`；答案中带 [S3][S7] 引用；
    `_sanitize_summary` 净化（本用户有"summary_only"偏好则截断长引语），如有改动发 `token_replace`；
    最后发 `meta` 事件（策略、步骤、路由诊断等）。
16. **收尾**（第 9 章）：`_persist_message` 把用户消息和带 sources 的助手消息写库；
    `asyncio.create_task` 启动记忆更新（抽取记忆 + 滚动摘要，第 11 章）；发 `done`。
17. **前端渲染**（第 10 章）：`onToken` 累加渲染，引用折叠面板展示"第 27 回 · 片段 143 · 重排分 0.87"，
    `onDone` 结束打字机。

一条请求，五次 LLM 调用（改写 1 + 拆分 1 + 专家 4~5 + 汇总 1 ≈ 7~8 次）、
一次混合检索、一次重排、两次记忆 LLM（后台）。每一步都有超时、降级与事件可观测。

---

## 第 14 章 术语表

| 术语 | 一句话解释 | 详见 |
| --- | --- | --- |
| RAG | 检索增强生成：先检索原文再让模型作答，附引用 | 第 0 章 |
| embedding / 向量 | 把文本映射成固定长度的数字串，语义相近则向量相近 | 第 2 章 |
| 余弦距离 / 相似度 | 向量夹角的度量；`score = 1 - distance` | 第 4 章 |
| pgvector | PostgreSQL 的向量扩展，支持 `vector` 列与相似度排序 | 第 2 章 |
| HNSW | 向量近似最近邻的图索引，牺牲少量召回换数量级提速 | 第 2 章 |
| GIN 索引 | 倒排索引，加速 tsvector 全文匹配与 trigram 相似度 | 第 2 章 |
| tsvector / 生成列 | 全文检索词袋列；由 `content` 自动计算并持久化 | 第 2 章 |
| chunk / 分块 | 把长文切成几百字的检索单元，带章节/序号元数据 | 第 3 章 |
| RecursiveCharacterTextSplitter | LangChain 分块器：按段落→句→字递归细化 | 第 3 章 |
| 状态机 / 租约 | 后台任务的阶段记录；租约防止两个进程同时索引一本书 | 第 3 章 |
| 原子替换 | 新向量全部备好后单事务"删旧+插新"，失败不损旧索引 | 第 3 章 |
| ContextVar | 异步请求级上下文，让"当前用户"隐式贯穿全链路 | 第 4 章 |
| 混合检索 | 向量（语义）+ 词法（字面）两路召回后融合 | 第 5 章 |
| FTS / pg_trgm / BM25 | 三种词法通道：全文检索 / 三元组相似 / 概率化相关性打分 | 第 5 章 |
| RRF | 倒数排名融合：只看各通道名次，`Σ 1/(60+rank)` | 第 5 章 |
| min-max 归一 / weighted 融合 | 池内把分数线性映射到 [0,1] 后加权求和 | 第 5 章 |
| bi-encoder / cross-encoder | 双塔（快、粗）vs 交叉（逐对细读、准、慢） | 第 6 章 |
| 重排（rerank） | cross-encoder 对候选精修名次，带三路融合与保护名额 | 第 6 章 |
| 邻居扩展 | 按章节+片段号取主命中前后的块补全上下文 | 第 6 章 |
| 指代消解 / standalone_query | 把"他后来怎么了"改写成独立可理解的问题 | 第 7 章 |
| Query Preparation | 一次 LLM 调用完成改写+路由+偏好提取，输出强校验 | 第 7 章 |
| fail-fast / 静默降级 | 配置错误立刻崩 vs 降级到次优路径——按"降级后是否仍正确"选择 | 第 1/5 章 |
| LangGraph / StateGraph | 把 Agent 流程显式建成"节点+条件边"的状态图 | 第 8 章 |
| Agent 策略 | direct（直答）/ multi_expert（四专家）/ react / plan_execute | 第 8 章 |
| 专家契约 | 每个专家固定的 focus/forbidden/输出格式，供 prompt 与程序校验共用 | 第 8 章 |
| Jaccard 相似度 | 集合交并比；本项目用字符 3-gram 检测专家报告重复 | 第 8 章 |
| 工具注册表 / allowed_tools | 工具的统一闸门：双重白名单 + 超时 + 异常转结构化结果 | 第 8 章 |
| SSE | 服务器单向推送：不结束的 HTTP 响应，事件以空行分隔 | 第 9 章 |
| token_replace | 后端净化稿整体覆盖已流式渲染内容的补充事件 | 第 8/9 章 |
| JWT / PBKDF2 | 无状态登录令牌 / 密码加盐慢哈希 | 第 9 章 |
| 三层记忆 | user_preference / novel_fact / session_fact 三种作用域 | 第 11 章 |
| 会话摘要 / covered_message_id | 压缩旧对话省 token；水位记录已摘要到哪条 | 第 11 章 |
| Recall / MRR / nDCG | 召回率 / 首命中倒数均值 / 折扣累计增益（排序质量） | 第 12 章 |
| LLM-as-judge | 用模型按四维给答案打分（1~5，带范围钳制） | 第 12 章 |
| 配对 bootstrap CI | 对 A/B 差值重采样估置信区间，区间不含 0 才算显著 | 第 12 章 |
| preparation cache | 固化 LLM 改写输出做对照实验，让 LLM 成为常量 | 第 12 章 |

---

## 附录：贯穿全书的十条工程原则

1. **能 fail-fast 的配置错误不在运行时静默**（启动即校验维度、overlap、API Key；BM25 缺扩展直接抛）。
2. **降级与硬崩的分界线**：降级后结果仍正确可接受才降级（pg_trgm→FTS），否则崩（BM25）。
3. **不信任 LLM 输出**：结构化输出必过白名单校验，失败走保守回退（宁可多检索不可漏检）。
4. **LLM 只建议，代码裁决**（章节规则 DSL 由程序验证后才应用）。
5. **外部调用必有超时与取消**（工具 20s、专家组 45s、请求 240s、SSE 断开取消图任务）。
6. ** expensive 资源单例 + 预热**（embedding/reranker 进程级单例，lifespan 预热）。
7. **写路径原子化**（向量替换单事务；先备料后切换，失败不损旧数据）。
8. **多租户隔离在 SQL 层强制**（四个检索通道共用 `_apply_filters`，ContextVar 注入）。
9. **一切改进评测先行**（spike → A/B → bootstrap CI → 默认值翻转；负结果留档）。
10. **流式体验与护栏语义可以兼得**（先流式后 `token_replace` 覆盖）。

祝你读得顺利。任何一章想深入，直接打开对应文件——本教程引用的路径和行号就是你的地图。
