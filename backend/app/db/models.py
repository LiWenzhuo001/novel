"""ORM 模型：用户、知识库文件、聊天会话、消息和 RAG 向量片段。

向量与业务表共用 PostgreSQL，依靠 user_id、domain 和 file_id 完成数据隔离。
"""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, Computed, DateTime, ForeignKey, Integer, String, Text, Float
from sqlalchemy.dialects.postgresql import TSVECTOR
from pgvector.sqlalchemy import Vector

from app.config import settings
from app.db import Base


class User(Base):
    """应用用户：正式用户级多租户的身份来源。"""

    __tablename__ = "users"

    id = Column(String(64), primary_key=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(100), default="")
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime)


class Embedding(Base):
    """RAG 向量片段（pgvector）。

    与业务表（knowledge_files / chat_sessions / chat_messages）共享同一个
    PostgreSQL 库，实现「向量 + 关系」单库统一。embedding 列维度由
    settings.embed_dim 决定（须与 embedding 模型维度一致）。

    search_vector 为 PostgreSQL 生成的 tsvector 列，用于全文检索（混合检索）。
    """

    __tablename__ = "embeddings"

    id = Column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embed_dim), nullable=False)
    source = Column(String(255), nullable=False, index=True)  # 文件名
    file_id = Column(String(32), index=True)
    user_id = Column(String(64), index=True, default="default")  # 多租户隔离
    domain = Column(String(32), index=True, default="novel", nullable=False)  # novel
    chapter = Column(String(255), index=True)
    chapter_no = Column(Integer, index=True)
    chunk_no = Column(Integer, index=True)
    page = Column(Integer)
    meta_json = Column(Text)  # 其它 metadata 的 JSON 字符串
    # 全文检索向量（PostgreSQL 生成列，自动从 content 生成）
    search_vector = Column(
        TSVECTOR,
        Computed("to_tsvector('simple', content)", persisted=True),
    )
    created_at = Column(DateTime, default=datetime.utcnow)


class KnowledgeFile(Base):
    """已上传并入库的知识库文件（替代原 raw/index.json）。

    status 状态机：pending（已落盘待索引）→ indexing（索引中）→ indexed（完成）/ failed（失败）。
    error 记录失败原因（便于前端展示与排查）。
    """

    __tablename__ = "knowledge_files"

    id = Column(String(32), primary_key=True)
    filename = Column(String(255), nullable=False)
    filetype = Column(String(16), default="")
    size = Column(Integer, default=0)
    chunks = Column(Integer, default=0)
    status = Column(String(16), default="pending")  # pending|indexing|indexed|failed
    index_stage = Column(String(32))  # pending|loading|parsing|analyzing_chapters|building_embeddings|switching|completed|failed
    index_progress = Column(Integer)  # 0-100, stage estimate
    index_message = Column(String(255))  # user-facing progress message
    error = Column(Text)  # 索引失败原因
    user_id = Column(String(64), index=True, default="default")  # 多租户隔离
    domain = Column(String(32), index=True, default="novel", nullable=False)  # novel
    created_at = Column(DateTime, default=datetime.utcnow)
    attempts = Column(Integer, default=0, nullable=False)
    lease_id = Column(String(64))
    lease_until = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    index_version = Column(String(255))
    embedding_model = Column(String(255))
    embed_dim = Column(Integer)
    chunk_size = Column(Integer)
    chunk_overlap = Column(Integer)
    indexed_at = Column(DateTime)
    chapter_count = Column(Integer)
    unassigned_chunk_count = Column(Integer)
    chapter_parse_status = Column(String(32))
    chapter_parser_mode = Column(String(32))
    chapter_parser_version = Column(String(64))
    detected_encoding = Column(String(32))
    index_warning = Column(Text)
    source_hash = Column(String(64))
    chapter_rule_json = Column(Text)
    chapter_rule_confidence = Column(Float)
    chapter_rule_validated = Column(Boolean, default=False)
    chapter_detection_model = Column(String(255))
    chapter_detection_prompt_version = Column(String(64))
    chapter_detection_error = Column(Text)
    chapter_detection_requested = Column(Boolean, default=False)


class ChatSession(Base):
    """一次连续对话的会话。"""

    __tablename__ = "chat_sessions"

    id = Column(String(32), primary_key=True)
    title = Column(String(255), default="新对话")
    role = Column(String(32), default="student")
    user_id = Column(String(64), index=True, default="default")  # 多租户隔离
    domain = Column(String(32), index=True, default="novel", nullable=False)  # novel
    file_id = Column(String(32), index=True)  # 当前会话绑定的小说
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatMessage(Base):
    """会话中的单条消息（user / assistant）。sources 以 JSON 字符串存储。"""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(32),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        index=True,
    )
    role = Column(String(16), nullable=False)
    content = Column(Text, default="")
    sources = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)


class ConversationSummary(Base):
    """压缩后的会话上下文，避免每轮把完整历史重新注入模型。"""

    __tablename__ = "conversation_summaries"

    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    session_id = Column(String(32), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True, nullable=False)
    user_id = Column(String(64), index=True, nullable=False)
    summary = Column(Text, nullable=False, default="")
    covered_message_id = Column(Integer, default=0, nullable=False)
    token_estimate = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AgentMemory(Base):
    """可检索的用户/小说长期记忆；第一版仅保存稳定事实和偏好。"""

    __tablename__ = "agent_memories"

    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    user_id = Column(String(64), index=True, nullable=False)
    session_id = Column(String(32), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    file_id = Column(String(32), index=True)
    memory_type = Column(String(32), nullable=False, default="session_fact")
    preference_key = Column(String(64), nullable=True, index=True)
    memory_version = Column(Integer, nullable=False, default=1)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embed_dim), nullable=True)
    importance = Column(Float, default=0.5, nullable=False)
    source_message_id = Column(Integer)
    expires_at = Column(DateTime)
    meta_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
