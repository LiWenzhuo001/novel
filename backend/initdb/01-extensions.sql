-- 首次启动数据库时由 postgres 超级用户执行（仅一次）。
-- 启用 pgvector 扩展，供 RAG 向量检索使用。
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- ParadeDB pg_search（BM25 词法通道）。pgvector/pgvector 镜像没有该扩展：
-- 失败只降级（ENABLE_BM25_SEARCH 保持关闭），不阻断首次初始化。
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_search;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pg_search unavailable, BM25 lexical lane stays disabled: %', SQLERRM;
END $$;
