"""检索通道、融合、入库与上下文扩展的职责拆分包。

构图（2026-08-30 拆分自 app/core/rag.py，rag.py 保留为门面并 re-export 全部旧符号）：

- channels.py：词法/向量检索通道（_vector_search、_fts_search、_chinese_lexical_search、_bm25_search 等）
- fusion.py：RRF / min-max 归一化 / 加权融合
- indexer.py：向量入库（add_documents、replace_documents、delete_by_file_id 等）
- context.py：章节局部精排与邻居扩展（chapter_local_refine、expand_novel_context）

!!! 勿从这里直接导入新符号到业务代码——统一走 app.core.rag（向后兼容。
"""