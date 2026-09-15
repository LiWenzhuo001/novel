"""Pydantic 请求/响应模型——前后端数据契约。

所有 API 入参与出参均在此定义，包括：
- ChatRequest / ChatResponse：小说问答请求与响应
- KBFileInfo：知识库文件信息
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
# 过渡版本仍接受旧策略值（multi_expert/roleplay），进入服务后立即映射为新契约
# 并在 meta.deprecations 中提示；下一个主版本从 Literal 中移除这两个值。
ChatStrategy = Literal["auto", "direct", "react", "plan_execute", "multi_expert", "roleplay"]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    role: Literal["student"] = "student"
    domain: Literal["novel"] = "novel"
    strategy: ChatStrategy = "auto"
    interaction_mode: Literal["qa", "roleplay"] = "qa"
    max_steps: Optional[int] = Field(default=None, ge=2, le=12)
    memory_mode: Literal["auto", "off"] = "auto"
    history: Optional[List[dict]] = None  # [{"role": "user/assistant", "content": ""}]
    session_id: Optional[str] = None  # 会话 ID，不传则服务端新建
    file_id: Optional[str] = Field(default=None, min_length=1, max_length=32)  # 当前咨询小说
    personas: Optional[List[str]] = Field(default=None, min_length=1, max_length=3)  # 角色扮演：在场人物（1~3）
    chapter_until: Optional[int] = Field(default=None, ge=1, le=10000)  # 角色扮演：剧情截至章节；不传=全书
    deprecations: List[str] = Field(default_factory=list, exclude=True)  # 旧值映射提示，随 meta 返回

    @model_validator(mode="after")
    def _normalize_strategy_and_mode(self) -> "ChatRequest":
        """旧策略值映射 + 交互模式约束。映射结果通过 deprecations 透出。"""
        if self.strategy == "multi_expert":
            self.strategy = "plan_execute"
            self.deprecations.append(
                "strategy=multi_expert 已弃用，已映射为 plan_execute；下个版本将拒绝该值"
            )
        elif self.strategy == "roleplay":
            self.strategy = "auto"
            self.interaction_mode = "roleplay"
            self.deprecations.append(
                "strategy=roleplay 已弃用，已映射为 interaction_mode=roleplay + strategy=auto；下个版本将拒绝该值"
            )
        if self.interaction_mode == "qa" and self.personas:
            raise ValueError("interaction_mode=qa 时不应传 personas；角色扮演请使用 interaction_mode=roleplay")
        if self.interaction_mode == "roleplay" and not self.personas:
            raise ValueError("interaction_mode=roleplay 需要 1~3 个 personas")
        if self.chapter_until is not None and self.interaction_mode != "roleplay":
            # 章节边界只对角色扮演和显式时间边界查询生效；普通问答传入直接忽略。
            self.chapter_until = None
        return self


class WorldSelectRequest(BaseModel):
    """「进入小说世界」选中人物后的角色卡生成请求。"""
    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(min_length=1, max_length=32)
    names: List[str] = Field(min_length=1, max_length=3)
    chapter_until: Optional[int] = Field(default=None, ge=1, le=10000)


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
