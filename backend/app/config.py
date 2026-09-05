"""应用全局配置——从 .env 读取所有运行参数。

必须配置项：
  LLM_API_KEY         大模型 API Key
  EMBED_DIM           向量维度（必须与 EMBEDDING_MODEL 一致）

可选覆盖：
  所有参数均有合理的默认值，可通过 .env 覆盖。
  完整配置清单参见 backend/.env.example。
"""

import os
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import load_dotenv

# 自动加载项目根目录下的 .env 文件
load_dotenv()

# 本地代理（如 Clash，127.0.0.1:7897）对 HF 大文件/重定向下载不稳定，会导致
# 模型下载失败；而 LLM/Embedding 等 OpenAI 兼容接口走代理会出现
# "远程计算机拒绝网络连接"（代理未开启时）。这里让这些域名绕过代理直连
# （仅在系统已设置代理时才有意义）。
_proxy_bypass = (
    "hf-mirror.com,huggingface.co,cdn-lfs.huggingface.co,cdn-lfs.hf-mirror.com,"
    "xethub.hf.co,cas-bridge.xethub.hf.co,transfer.xethub.hf.co,"
    "api.deepseek.com,dashscope.aliyuncs.com,open.bigmodel.cn,"
    "api.openai.com,api.moonshot.cn,ark.cn-beijing.volces.com,api.siliconflow.cn"
)
_existing_np = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
_merged_np = ",".join(x for x in [_existing_np, _proxy_bypass] if x)
os.environ["NO_PROXY"] = _merged_np
os.environ["no_proxy"] = _merged_np

# backend/ 目录（config.py 位于 backend/app/config.py）
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    """统一管理所有配置项，通过环境变量注入。

    设计原则：
    - 所有配置项均有默认值，开箱可跑（除必须的 API Key）
    - 敏感信息（Key/Password）仅从环境变量读取，不硬编码
    - 兼容 OpenAI 兼容接口体系（DeepSeek / 通义千问 / 智谱 GLM 等）
    """
    def __init__(self):
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        self.llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
        # 超时与重试（秒 / 次）：防止 LLM/网络抖动导致请求长时间挂起
        self.llm_timeout = float(os.getenv("LLM_TIMEOUT", "60"))
        self.llm_max_retries = int(os.getenv("LLM_MAX_RETRIES", "1"))
        # DeepSeek V4 等推理模型默认先输出思维链（reasoning_content）再给答案，content 常为 null，
        # 且思考导致延迟与 token 显著上升。关闭思考（true）后直接返回 content，响应更快。
        # 仅对支持 thinking 参数的模型生效；对不支持该参数的服务端不报错、不影响。
        self.llm_disable_thinking = (
            os.getenv("LLM_DISABLE_THINKING", "true").lower() == "true"
        )
        # LLM-as-judge 评审模型：默认沿用聊天模型。同一模型自评存在自评偏差风险，
        # 可通过 JUDGE_MODEL 指定其他模型（需在同一 base_url 下可用）。
        self.judge_model = os.getenv("JUDGE_MODEL", "").strip() or self.llm_model

        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "api")  # local | api
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", self.llm_api_key)
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", self.llm_base_url)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        # 可选本地模型目录。设置后只改变加载路径，knowledge_files 仍记录 EMBEDDING_MODEL，
        # 便于离线部署和缓存迁移；为空时按模型名从 HuggingFace 缓存/网络加载。
        local_model_path = os.getenv("EMBEDDING_LOCAL_PATH", "").strip()
        if local_model_path:
            path = Path(local_model_path)
            self.embedding_local_path = str((path if path.is_absolute() else BASE_DIR / path).resolve())
        else:
            self.embedding_local_path = ""
        # embedding API 调用的超时与重试（local 本地模型不涉及网络）
        self.embedding_timeout = float(os.getenv("EMBEDDING_TIMEOUT", "60"))
        self.embedding_max_retries = int(os.getenv("EMBEDDING_MAX_RETRIES", "2"))
        # OpenAIEmbeddings 默认会用 tiktoken 将文本转成 token ID 数组。
        # 第三方 OpenAI-compatible 服务（如 SiliconFlow/Qwen）未必使用 OpenAI tokenizer，
        # 因此默认直接发送原始字符串；原生 OpenAI tokenizer 场景可显式开启长度安全处理。
        raw_check_ctx = os.getenv(
            "EMBEDDING_CHECK_CTX_LENGTH",
            "true" if self.embedding_provider == "local" else "false",
        ).strip().lower()
        if raw_check_ctx not in {"true", "false"}:
            raise ValueError("EMBEDDING_CHECK_CTX_LENGTH 仅支持 true 或 false")
        self.embedding_check_ctx_length = raw_check_ctx == "true"
        # 本地模型运行设备：auto 会优先使用 CUDA；显式 cuda 时若 CUDA 不可用则启动即报错，避免悄悄退回 CPU。
        self.embedding_device = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
        if self.embedding_device == "gpu":
            self.embedding_device = "cuda"
        elif self.embedding_device not in {"auto", "cpu", "cuda"} and not self.embedding_device.startswith("cuda:"):
            raise ValueError("EMBEDDING_DEVICE 仅支持 auto、cpu、cuda 或 cuda:<GPU编号>")
        # 本地大模型按小批次生成向量，避免一次性处理整本小说造成内存峰值。
        default_batch = "4" if self.embedding_provider == "local" and self.embedding_model == "BAAI/bge-m3" else "32"
        self.embedding_batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", default_batch)))

        raw_dir = Path(os.getenv("RAW_DIR", str(BASE_DIR / "data" / "raw")))
        self.raw_dir = str((raw_dir if raw_dir.is_absolute() else BASE_DIR / raw_dir).resolve())
        self.top_k = int(os.getenv("TOP_K", "5"))

        # ===== RAG 检索质量参数 =====
        # 相似度阈值：低于此分数的结果会被过滤掉（cosine distance → score = 1 - distance）
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
        # 混合检索：向量 + 全文（PostgreSQL FTS），用 RRF 融合
        self.enable_hybrid_search = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
        # RRF 参数：k 越大，排名差异对最终分数影响越平滑
        self.rrf_k = int(os.getenv("RRF_K", "60"))
        # ===== 池内融合模式（评测驱动的 A/B 开关）=====
        # rrf：现状，Reciprocal Rank Fusion 只用通道排名；weighted：各通道分数在合并
        # 候选集内 min-max 归一后加权求和。评测数据显示瓶颈在池内排序（evals/README §四.3），
        # weighted 让向量/词法的原始相关度参与排序，而不是只看名次。
        self.fusion_mode = os.getenv("FUSION_MODE", "rrf").strip().lower()
        if self.fusion_mode not in {"rrf", "weighted"}:
            raise ValueError(f"FUSION_MODE 必须是 rrf 或 weighted，当前: {self.fusion_mode}")
        self.vector_weight = float(os.getenv("VECTOR_WEIGHT", "0.5"))
        self.lexical_weight = float(os.getenv("LEXICAL_WEIGHT", "0.5"))
        # 混合检索时各路的候选数量（取 Top-N 做融合）
        self.hybrid_candidate_k = max(1, int(os.getenv("HYBRID_CANDIDATE_K", "60")))
        # 中文查询优先使用分词后的词法覆盖召回；关闭后回退 PostgreSQL simple FTS。
        self.enable_chinese_lexical_search = (
            os.getenv("ENABLE_CHINESE_LEXICAL_SEARCH", "true").lower() == "true"
        )
        self.chinese_lexical_max_terms = max(
            1, min(16, int(os.getenv("CHINESE_LEXICAL_MAX_TERMS", "8")))
        )
        # BM25 词法通道（ParadeDB pg_search，docker-compose 已换 paradedb/paradedb 镜像）。
        # spike 实测：候选召回 0.575 > ILIKE 词法 0.525，延迟约 90 倍优，默认开启。
        # 需要 pg_search 扩展 + 迁移 20260828_0012 的 bm25 索引；fail-fast：缺失时
        # 检索直接报错（不静默降级），属于必须修复的环境配置问题。
        self.enable_bm25_search = os.getenv("ENABLE_BM25_SEARCH", "true").lower() == "true"

        # ===== 面向 RAG 的 Query 改写 =====
        # 首轮问题也执行一次改写；模型同时产出自然语言 standalone_query 和一条检索 Query。
        self.enable_query_rewrite = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"
        self.query_rewrite_history_messages = max(0, int(os.getenv("QUERY_REWRITE_HISTORY_MESSAGES", "6")))
        self.query_rewrite_max_tokens = max(128, min(1000, int(os.getenv("QUERY_REWRITE_MAX_TOKENS", "400"))))
        self.query_rewrite_timeout = max(5.0, float(os.getenv("QUERY_REWRITE_TIMEOUT", "30")))
        self.query_rewrite_retries = max(0, min(1, int(os.getenv("QUERY_REWRITE_RETRIES", "1"))))
        self.query_rewrite_prompt_version = os.getenv("QUERY_REWRITE_PROMPT_VERSION", "rag-query-v2")
        self.query_rewrite_max_chars = max(80, min(600, int(os.getenv("QUERY_REWRITE_MAX_CHARS", "400"))))
        self.enable_llm_query_routing = os.getenv("ENABLE_LLM_QUERY_ROUTING", "true").lower() == "true"
        self.query_routing_prompt_version = os.getenv("QUERY_ROUTING_PROMPT_VERSION", "rag-routing-v1")
        self.query_routing_confidence_threshold = min(
            1.0, max(0.0, float(os.getenv("QUERY_ROUTING_CONFIDENCE_THRESHOLD", "0.75")))
        )

        # ===== 自动会话记忆 =====
        self.memory_enabled = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
        self.memory_max_context_items = max(1, min(50, int(os.getenv("MEMORY_MAX_CONTEXT_ITEMS", "8"))))
        self.memory_candidate_limit = max(1, min(200, int(os.getenv("MEMORY_CANDIDATE_LIMIT", "50"))))
        self.memory_summary_trigger_messages = max(2, int(os.getenv("MEMORY_SUMMARY_TRIGGER_MESSAGES", "8")))
        self.memory_summary_trigger_chars = max(500, int(os.getenv("MEMORY_SUMMARY_TRIGGER_CHARS", "5000")))
        self.memory_extract_max_items = max(1, min(10, int(os.getenv("MEMORY_EXTRACT_MAX_ITEMS", "5"))))
        self.memory_extract_max_tokens = max(128, int(os.getenv("MEMORY_EXTRACT_MAX_TOKENS", "600")))
        self.memory_task_timeout = max(5.0, float(os.getenv("MEMORY_TASK_TIMEOUT", "30")))
        self.memory_min_importance = min(1.0, max(0.0, float(os.getenv("MEMORY_MIN_IMPORTANCE", "0.55"))))
        # 偏好与普通记忆分池召回：偏好占独立预算（默认 3），不再挤占小说/会话事实（默认 5）。
        self.memory_preference_top_k = max(0, min(10, int(os.getenv("MEMORY_PREFERENCE_TOP_K", "3"))))
        self.memory_fact_pool_size = max(1, min(50, int(os.getenv("MEMORY_FACT_POOL_SIZE", "5"))))
        # 摘要触发增加 token 感知阈值（按 字符数/4 估算；参照 Letta 模型感知阈值，避免硬编码魔法数。）
        self.memory_summary_trigger_tokens = max(128, int(os.getenv("MEMORY_SUMMARY_TRIGGER_TOKENS", "6000")))
        # B2：会话级记忆 TTL——session_fact 默认 30 天过期；novel_fact/偏好默认永不过期。
        self.memory_session_fact_ttl_days = max(0, int(os.getenv("MEMORY_SESSION_FACT_TTL_DAYS", "30")))
        self.memory_ttl_sweeper_enabled = os.getenv("MEMORY_TTL_SWEEPER_ENABLED", "false").lower() == "true"
        self.memory_ttl_sweeper_interval_hours = max(1, int(os.getenv("MEMORY_TTL_SWEEPER_INTERVAL_HOURS", "24")))

        # ===== 请求可靠性边界 =====
        self.agent_request_timeout = float(os.getenv("AGENT_REQUEST_TIMEOUT", "240"))

        # ===== Agent 执行预算 =====
        self.agent_max_steps = max(2, min(12, int(os.getenv("AGENT_MAX_STEPS", "6"))))
        self.agent_tool_timeout = max(1.0, float(os.getenv("AGENT_TOOL_TIMEOUT", "20")))
        self.agent_max_experts = max(1, min(4, int(os.getenv("AGENT_MAX_EXPERTS", "4"))))
        self.agent_multi_expert_timeout = max(5.0, float(os.getenv("AGENT_MULTI_EXPERT_TIMEOUT", "45")))
        self.agent_expert_max_tokens = max(128, int(os.getenv("AGENT_EXPERT_MAX_TOKENS", "800")))
        self.agent_synthesis_max_tokens = max(256, int(os.getenv("AGENT_SYNTHESIS_MAX_TOKENS", "1200")))
        self.agent_expert_dispatch_mode = os.getenv("AGENT_EXPERT_DISPATCH_MODE", "hybrid").lower()
        if self.agent_expert_dispatch_mode not in {"hybrid", "template"}:
            raise ValueError("AGENT_EXPERT_DISPATCH_MODE 仅支持 hybrid / template")
        self.agent_dispatch_max_tokens = max(128, int(os.getenv("AGENT_DISPATCH_MAX_TOKENS", "500")))
        self.agent_report_similarity_threshold = min(1.0, max(0.0, float(os.getenv("AGENT_REPORT_SIMILARITY_THRESHOLD", "0.72"))))
        self.agent_expert_correction_retries = max(0, min(1, int(os.getenv("AGENT_EXPERT_CORRECTION_RETRIES", "1"))))
        self.novel_context_k = max(self.top_k, int(os.getenv("NOVEL_CONTEXT_K", "8")))
        self.novel_neighbor_window = max(0, min(3, int(os.getenv("NOVEL_NEIGHBOR_WINDOW", "1"))))

        # 小说索引切分参数。索引完成时写入 knowledge_files，避免静默混用不同版本。
        self.novel_chunk_size = max(200, int(os.getenv("NOVEL_CHUNK_SIZE", "650")))
        self.novel_chunk_overlap = max(0, int(os.getenv("NOVEL_CHUNK_OVERLAP", "120")))
        if self.novel_chunk_overlap >= self.novel_chunk_size:
            raise ValueError("NOVEL_CHUNK_OVERLAP 必须小于 NOVEL_CHUNK_SIZE")

        # ===== 手动重新索引时的 LLM 章节规则发现 =====
        self.enable_llm_chapter_detection = (
            os.getenv("ENABLE_LLM_CHAPTER_DETECTION", "true").lower() == "true"
        )
        self.chapter_detection_model = os.getenv("CHAPTER_DETECTION_MODEL", "").strip() or self.llm_model
        self.chapter_detection_max_tokens = max(256, int(os.getenv("CHAPTER_DETECTION_MAX_TOKENS", "800")))
        self.chapter_detection_timeout = max(5.0, float(os.getenv("CHAPTER_DETECTION_TIMEOUT", "45")))
        self.chapter_detection_retries = max(0, min(1, int(os.getenv("CHAPTER_DETECTION_RETRIES", "1"))))
        self.chapter_detection_candidate_limit = max(
            20, min(500, int(os.getenv("CHAPTER_DETECTION_CANDIDATE_LIMIT", "240")))
        )
        self.chapter_detection_confidence_threshold = min(
            1.0, max(0.0, float(os.getenv("CHAPTER_DETECTION_CONFIDENCE_THRESHOLD", "0.75")))
        )
        self.chapter_detection_prompt_version = os.getenv(
            "CHAPTER_DETECTION_PROMPT_VERSION", "chapter-discovery-v1"
        )

        # ===== Cross-encoder 重排（默认开启，纯重排模式）=====
        # 在 RRF 融合候选后用 cross-encoder 对 (query, doc) 逐对重打分排序。
        # 2026-08-29 A/B（evals/reports/*_rerank_v2m3_*）：换 bge-reranker-v2-m3 +
        # BM25 新池子后翻身——红楼梦 R@1 +0.275 / MRR +0.217（均显著），西游记持平，
        # 纯重排（blend 关）优于 blend 保护模式，故默认开启且关闭 blend。
        self.enable_reranker = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
        # 默认走 SiliconFlow 远程重排（部署免下 ~2.3GB 本地模型）；
        # local 仅供离线/无 RERANKER_API_KEY 的环境显式选用。远程模式未配 key 会在首次重排调用时 fail-fast。
        self.reranker_provider = os.getenv("RERANKER_PROVIDER", "siliconflow").strip().lower()
        default_reranker_model = (
            "Qwen/Qwen3-Reranker-8B"
            if self.reranker_provider == "siliconflow"
            else "BAAI/bge-reranker-v2-m3"
        )
        self.reranker_model = os.getenv("RERANKER_MODEL", default_reranker_model).strip()
        self.reranker_base_url = os.getenv(
            "RERANKER_BASE_URL", "https://api.siliconflow.cn/v1"
        ).strip().rstrip("/")
        self.reranker_endpoint = os.getenv("RERANKER_ENDPOINT", "/rerank").strip()
        self.reranker_api_key = os.getenv("RERANKER_API_KEY", "").strip()
        self.reranker_timeout = float(os.getenv("RERANKER_TIMEOUT", "30"))
        self.reranker_max_retries = int(os.getenv("RERANKER_MAX_RETRIES", "2"))
        # 交给重排器的候选数量（>= top_k）
        self.reranker_candidate_n = max(
            self.novel_context_k, int(os.getenv("RERANKER_CANDIDATE_N", str(self.hybrid_candidate_k)))
        )
        # 开启重排后候选阶段使用更宽松阈值，最终相关性由 cross-encoder 决定。
        self.reranker_candidate_threshold = min(
            1.0, max(0.0, float(os.getenv("RERANKER_CANDIDATE_THRESHOLD", "0.15")))
        )
        # 重排保护（blend）：2026-08-29 A/B 中纯重排优于保护模式，默认关闭。
        self.enable_reranker_blend = os.getenv("ENABLE_RERANKER_BLEND", "false").lower() == "true"
        self.reranker_weight = float(os.getenv("RERANKER_WEIGHT", "0.70"))
        self.rrf_weight = float(os.getenv("RRF_WEIGHT", "0.20"))
        self.raw_score_weight = float(os.getenv("RAW_SCORE_WEIGHT", "0.10"))
        self.reranker_protect_top_n = max(0, int(os.getenv("RERANKER_PROTECT_TOP_N", "3")))
        self.reranker_protect_slots = max(0, int(os.getenv("RERANKER_PROTECT_SLOTS", "2")))

        # 可选章节内二级检索：默认关闭，避免改变现有聊天链路的延迟和排序。
        self.enable_chapter_local_retrieval = os.getenv("ENABLE_CHAPTER_LOCAL_RETRIEVAL", "false").lower() == "true"
        self.chapter_local_candidate_k = max(10, int(os.getenv("CHAPTER_LOCAL_CANDIDATE_K", "40")))

        # 向量维度：必须与 EMBEDDING_MODEL 匹配！
        #   text-embedding-3-small = 1536
        #   text-embedding-3-large = 3072
        #   BAAI/bge-m3            = 1024
        #   BAAI/bge-base-zh-v1.5  = 768
        # 修改 EMBEDDING_MODEL 后务必同步修改 EMBED_DIM（且需清空 embeddings 表）。
        self.embed_dim = int(os.getenv("EMBED_DIM", "1536"))
        known_local_dimensions = {
            "BAAI/bge-small-zh-v1.5": 512,
            "BAAI/bge-base-zh-v1.5": 768,
            "BAAI/bge-m3": 1024,
        }
        expected_dim = known_local_dimensions.get(self.embedding_model)
        if self.embedding_provider == "local" and expected_dim and self.embed_dim != expected_dim:
            raise ValueError(
                f"本地 Embedding 模型 {self.embedding_model} 输出 {expected_dim} 维，"
                f"但 EMBED_DIM={self.embed_dim}，请同步修改配置。"
            )

        # ===== PostgreSQL + pgvector 配置（向量与业务数据共用同一库）=====
        self.pg_host = os.getenv("POSTGRES_HOST", "postgres")
        self.pg_port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.pg_user = os.getenv("POSTGRES_USER", "job_agent")
        self.pg_password = os.getenv("POSTGRES_PASSWORD", "job_agent123")
        self.pg_database = os.getenv("POSTGRES_DB", "job_agent")
        # 若直接给定完整连接串则优先使用（覆盖上面的分项配置）
        self._database_url = os.getenv("DATABASE_URL")

        # CORS 允许的前端来源（逗号分隔；默认仅允许本地开发端口）
        self.cors_origins = [
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
            if o.strip()
        ]

        # ===== 鉴权与多租户数据隔离 =====
        # 默认开启：users 表 + JWT 登录。JWT_SECRET 为必填，未配置将启动失败；
        # 纯本地裸跑开发可显式指定 JWT_SECRET=dev-xxx python app/main.py。
        # 兼容：API_TOKENS 仍可作为本地/脚本调用的静态 token 白名单。

        self.user_auth_enabled = os.getenv("USER_AUTH_ENABLED", "true").lower() == "true"
        self.jwt_secret = os.getenv("JWT_SECRET", "")
        # 开启 JWT 鉴权时密钥是必需项，fail-fast 阻止"重启全下线/多实例验签失败"的随机兜底。

        if self.user_auth_enabled and not self.jwt_secret:
            raise ValueError("USER_AUTH_ENABLED=true 时必须配置固定 JWT_SECRET")
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        if self.jwt_algorithm != "HS256":
            raise ValueError("当前内置 JWT 实现仅支持 HS256")
        self.jwt_access_token_minutes = int(os.getenv("JWT_ACCESS_TOKEN_MINUTES", "1440"))
        tokens_raw = os.getenv("API_TOKENS", "")
        self.api_tokens: dict[str, str] = {}
        for pair in tokens_raw.split(","):
            pair = pair.strip()
            if ":" in pair:
                uid, tok = pair.split(":", 1)
                self.api_tokens[uid.strip()] = tok.strip()
        self.auth_enabled = self.user_auth_enabled or bool(self.api_tokens)
        # 系统租户：拥有内置默认小说（如红楼梦）的只读共享索引。
        # 检索时普通用户的候选池额外包含该租户的向量（file_id 过滤保证不跨书混入）；
        # 删除/重建保持 owner 等值过滤，系统书对所有普通用户只读。
        self.system_user = os.getenv("SYSTEM_USER", "system")
        self.default_user = os.getenv("DEFAULT_USER", "default")

    @property
    def database_url(self) -> str:
        if self._database_url:
            return self._database_url
        return (
            f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def async_database_url(self) -> str:
        """异步连接串（asyncpg）。若 DATABASE_URL 已给定则替换驱动前缀。"""
        if self._database_url:
            return self._database_url.replace(
                "postgresql+psycopg2://", "postgresql+asyncpg://"
            ).replace("postgresql://", "postgresql+asyncpg://")
        return (
            f"postgresql+asyncpg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    def validate(self):
        if self.user_auth_enabled and not self.jwt_secret:
            raise ValueError("USER_AUTH_ENABLED=true 时必须配置固定 JWT_SECRET")
        if not self.llm_api_key:
            raise ValueError("LLM_API_KEY 未配置，请在 .env 中设置")
        if self.embedding_provider == "api" and not self.embedding_api_key:
            raise ValueError("EMBEDDING_PROVIDER=api 但未配置 EMBEDDING_API_KEY")
        # reranker 配置由 get_reranker() 在启用重排时独立校验，不能让远程
        # rerank 配置问题阻断不依赖重排的 LLM 请求。

    @property
    def reranker_url(self) -> str:
        """返回远程 reranker 的完整 URL。"""
        return f"{self.reranker_base_url}/{self.reranker_endpoint.lstrip('/')}"

    def validate_reranker(self) -> None:
        """校验 reranker provider 配置，避免首个检索请求才暴露错误。"""
        if self.reranker_provider not in {"local", "siliconflow"}:
            raise ValueError(
                "RERANKER_PROVIDER 仅支持 local 或 siliconflow，"
                f"当前: {self.reranker_provider}"
            )
        if not self.reranker_model:
            raise ValueError("RERANKER_MODEL 未配置")
        if self.reranker_timeout <= 0:
            raise ValueError("RERANKER_TIMEOUT 必须大于 0")
        if self.reranker_max_retries < 0:
            raise ValueError("RERANKER_MAX_RETRIES 不能为负数")
        if self.reranker_provider != "siliconflow":
            return
        parsed = urlsplit(self.reranker_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("RERANKER_BASE_URL 必须是完整的 http/https URL")
        if not self.reranker_endpoint or not self.reranker_endpoint.startswith("/"):
            raise ValueError("RERANKER_ENDPOINT 必须以 / 开头")
        endpoint = urlsplit(self.reranker_endpoint)
        if endpoint.scheme or endpoint.netloc:
            raise ValueError("RERANKER_ENDPOINT 只能是相对路径")
        if not self.reranker_api_key:
            raise ValueError("RERANKER_PROVIDER=siliconflow 但未配置 RERANKER_API_KEY")


settings = Settings()
