"""检索通道：向量、FTS、中文词法、BM25 四路候选生成及其公共过滤/元数据工具。



!!! 业务代码请从 app.core.rag 导入，勿直接 import 本模块（保持门面兼容。"""

from __future__ import annotations

import json
import re

from sqlalchemy import case, func, or_, select

from app.config import settings
from app.core.visibility import visible_user_filter, visible_users
from app.db.models import Embedding


_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_FALLBACK_TOKEN_RE = re.compile(r"[\u3400-\u9fff]{2,}|[A-Za-z0-9_]{2,}")
_CHINESE_STOPWORDS = {
    "什么", "为什么", "怎么", "如何", "哪些", "这个", "那个", "人物", "小说", "文中",
    "进行", "以及", "其中", "一个", "后来", "相关", "主要", "分析", "说明", "请问",
}


def _parse_meta(row: Embedding) -> dict:
    base = {
        "source": row.source,
        "file_id": row.file_id,
        "domain": getattr(row, "domain", "novel"),
        "chapter": getattr(row, "chapter", None),
        "chapter_no": getattr(row, "chapter_no", None),
        "chunk_no": getattr(row, "chunk_no", None),
        "page": getattr(row, "page", None),
    }
    if row.meta_json:
        try:
            base.update(json.loads(row.meta_json))
        except (json.JSONDecodeError, TypeError):
            pass
    return base


def _apply_filters(stmt, filter_source=None, filter_user=None, filter_domain=None, filter_file_id=None):
    """把来源、用户、领域和文件条件附加到 SQL 查询。"""
    if filter_source:
        stmt = stmt.where(Embedding.source == filter_source)
    if filter_user:
        # 候选池额外包含系统租户的向量（内置默认小说）；跨书混入被 file_id
        # 过滤挡住——选书检索时 file_id 必传，系统行仅在该书被选中时进入。
        stmt = stmt.where(visible_user_filter(Embedding.user_id, filter_user))
    if filter_domain:
        stmt = stmt.where(Embedding.domain == filter_domain)
    if filter_file_id:
        stmt = stmt.where(Embedding.file_id == filter_file_id)
    return stmt


async def _vector_search(
    session, qvec, k: int, filter_source: str = None,
    filter_user: str = None, filter_domain: str = None, filter_file_id: str = None,
):
    """执行 pgvector 余弦距离检索并返回原始相似度。"""
    distance = Embedding.embedding.cosine_distance(qvec)
    stmt = select(Embedding, distance.label("distance")).order_by(distance)
    stmt = _apply_filters(stmt, filter_source, filter_user, filter_domain, filter_file_id).limit(k)
    result = await session.execute(stmt)
    return [(row, max(0.0, 1.0 - float(distance_value))) for row, distance_value in result.all()]


async def _fts_search(
    session, query: str, k: int, filter_source: str = None,
    filter_user: str = None, filter_domain: str = None, filter_file_id: str = None,
):
    """执行 PostgreSQL 全文检索并返回词法相关度。"""
    tsquery = func.plainto_tsquery("simple", query)
    rank = func.ts_rank(Embedding.search_vector, tsquery)
    stmt = select(Embedding, rank.label("rank")).where(Embedding.search_vector.op("@@")(tsquery))
    stmt = _apply_filters(stmt, filter_source, filter_user, filter_domain, filter_file_id)
    result = await session.execute(stmt.order_by(rank.desc()).limit(k))
    return [(row, min(1.0, float(rank_value) / 0.1)) for row, rank_value in result.all()]


def _tokenize_chinese_query(query: str, max_terms: int | None = None) -> list[str]:
    """Return stable Chinese lexical anchors while preserving uncommon person names."""
    limit = max_terms or settings.chinese_lexical_max_terms
    anchor_text = query
    for phrase in sorted(_CHINESE_STOPWORDS, key=len, reverse=True):
        anchor_text = anchor_text.replace(phrase, " ")
    anchor_text = re.sub(r"[和与及跟的在对把将由从向于、，。！？；：\s]+", " ", anchor_text)
    anchors = re.findall(r"[\u3400-\u9fff]{2,6}|[A-Za-z0-9_]{2,}", anchor_text)
    try:
        import jieba
        candidates = anchors + jieba.lcut(query, cut_all=False)
    except ImportError:
        candidates = anchors + _FALLBACK_TOKEN_RE.findall(query)

    tokens: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        token = raw.strip().lower()
        if len(token) < 2 or token in _CHINESE_STOPWORDS or token in seen:
            continue
        if not (_CJK_RE.search(token) or re.search(r"[a-z0-9]", token)):
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


async def _chinese_lexical_search(
    session, query: str, k: int, filter_source: str = None,
    filter_user: str = None, filter_domain: str = None, filter_file_id: str = None,
):
    """Chinese lexical lane: token coverage + pg_trgm word similarity."""
    tokens = _tokenize_chinese_query(query)
    if not tokens:
        return []
    conditions = []
    coverage_parts = []
    for token in tokens:
        escaped = token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        condition = Embedding.content.ilike(f"%{escaped}%", escape="\\")
        conditions.append(condition)
        coverage_parts.append(case((condition, 1.0), else_=0.0))
    coverage = sum(coverage_parts) / float(len(tokens))
    trigram = func.least(1.0, func.word_similarity(query, Embedding.content))
    rank = coverage * 0.85 + trigram * 0.15
    stmt = select(Embedding, rank.label("rank")).where(or_(*conditions))
    stmt = _apply_filters(stmt, filter_source, filter_user, filter_domain, filter_file_id)
    result = await session.execute(stmt.order_by(rank.desc()).limit(k))
    return [(row, max(0.0, min(1.0, float(rank_value)))) for row, rank_value in result.all()]


_BM25_SCORE_SQL = "paradedb.score(embeddings.id)"


def _tokenize_for_bm25(query: str) -> str:
    """BM25 查询侧必须与索引 tokenizer 对齐：中文整句无空格，默认 parse 会把它当成
    一个超长 token 而无法命中 ngram 索引（spike 实测 0 命中）。复用 jieba 分词空格连接。"""
    tokens = _tokenize_chinese_query(query)
    return " ".join(tokens) if tokens else query


async def _bm25_search(
    session, query: str, k: int, filter_source: str = None,
    filter_user: str = None, filter_domain: str = None, filter_file_id: str = None,
):
    """ParadeDB pg_search BM25 词法通道（需要 pg_search 扩展与 embeddings_bm25_idx 索引）。

    返回与其他词法通道一致的 (Embedding, score) 列表；BM25 分数无量纲上限，
    RRF 模式只用名次不受影响，weighted 模式做池内 min-max 归一。
    """
    from sqlalchemy import text as sql_text

    conditions = ["embeddings.content @@@ :bm25_query"]
    params: dict[str, object] = {"bm25_query": _tokenize_for_bm25(query), "k": k}
    if filter_user:
        # 与 _apply_filters 对齐：候选池额外包含系统租户（内置默认小说）。
        users = visible_users(filter_user)
        placeholders = ", ".join(f":visible_user_{index}" for index in range(len(users)))
        conditions.append(f"embeddings.user_id IN ({placeholders})")
        for index, user in enumerate(users):
            params[f"visible_user_{index}"] = user
    if filter_domain:
        conditions.append("embeddings.domain = :filter_domain")
        params["filter_domain"] = filter_domain
    if filter_source:
        conditions.append("embeddings.source = :filter_source")
        params["filter_source"] = filter_source
    if filter_file_id:
        conditions.append("embeddings.file_id = :filter_file_id")
        params["filter_file_id"] = filter_file_id
    stmt = sql_text(
        f"SELECT embeddings.*, {_BM25_SCORE_SQL} AS bm25_score "
        f"FROM embeddings WHERE {' AND '.join(conditions)} "
        f"ORDER BY bm25_score DESC LIMIT :k"
    )
    result = await session.execute(stmt, params)
    rows: list[tuple[Embedding, float]] = []
    for mapping in result.mappings():
        mapping = dict(mapping)
        score = float(mapping.pop("bm25_score"))
        # text 查询返回行映射，还原为 ORM 实例以复用 _parse_meta 等元数据逻辑。
        rows.append((Embedding(**mapping), score))
    return rows
