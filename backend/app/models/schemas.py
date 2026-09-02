"""Pydantic 请求/响应模型——前后端数据契约。

所有 API 入参与出参均在此定义，包括：
- ChatRequest / ChatResponse：小说问答请求与响应
- KBFileInfo：知识库文件信息
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field


# ===== 用户认证 / 多租户 =====
class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    email: Optional[str] = Field(default=None, max_length=255)
    display_name: str = Field(default="", max_length=100)


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class UserInfo(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    display_name: str = ""
    is_admin: bool = False


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserInfo


# ===== 小说问答 =====
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    role: Literal["student"] = "student"
    domain: Literal["novel"] = "novel"
    strategy: Literal["auto", "direct", "multi_expert", "react", "plan_execute"] = "auto"
    max_steps: Optional[int] = Field(default=None, ge=2, le=12)
    memory_mode: Literal["auto", "off"] = "auto"
    history: Optional[List[dict]] = None  # [{"role": "user/assistant", "content": ""}]
    session_id: Optional[str] = None  # 会话 ID，不传则服务端新建
    file_id: Optional[str] = Field(default=None, min_length=1, max_length=32)  # 当前咨询小说


class SourceDoc(BaseModel):
    source: str
    snippet: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    role: str


# ===== 知识库 =====
class KBFileInfo(BaseModel):
    id: str
    filename: str
    filetype: str
    size: int
    chunks: int
    domain: Literal["novel"] = "novel"
    status: Optional[Literal["pending", "indexing", "indexed", "failed"]] = None
    error: Optional[str] = None
    chapter_count: Optional[int] = None
    unassigned_chunk_count: Optional[int] = None
    chapter_parse_status: Optional[Literal["ok", "unrecognized"]] = None
    chapter_parser_mode: Optional[Literal["strict", "inline_fallback", "llm_assisted", "none"]] = None
    chapter_parser_version: Optional[str] = None
    chapter_index_stale: bool = False
    detected_encoding: Optional[str] = None
    index_warning: Optional[str] = None
    chapter_rule_confidence: Optional[float] = None
    chapter_rule_validated: Optional[bool] = None
    chapter_detection_model: Optional[str] = None
    chapter_detection_error: Optional[str] = None
    created_at: str
