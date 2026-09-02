# RAG 召回评测说明

本目录是小说 RAG 问答系统的离线评测资产：数据集、评测脚本入口、以及历次实验报告。

> **仓库策略（2026-08-30）**：`evals/reports/` 已整体移出版本库（`.gitignore`，仅存本地），
> 历史报告不再入库。做回归对比请用 `evals/baselines/` 下的**轻量基线**（当前默认配置
> BM25 + v2-m3 纯重排的逐题指标，约 12KB/份，剥离了 trace 大字段）；新基线的生成方式：
> 跑完评测后从 `*_cases.jsonl` 抽取指标字段（参考 `evals_stats.paired_bootstrap_ci` 所需列）。

## 一、跑一次评测

统一入口是 `backend/scripts/evaluate_rag_recall.py`（在 `backend` 目录下执行）：

```bash
cd E:/novel/backend
POSTGRES_HOST=127.0.0.1 ./.venv/Scripts/python.exe scripts/evaluate_rag_recall.py \
  --file-id c959fb4c1a30 \
  --user-id c1e78368550f498787e3871ed9291b63 \
  --name <实验名> \
  --k 10 --with-context --neighbor-window 1 \
  --preparation-cache ../evals/datasets/xiyouji_query_preparation_20260826.json
```

产物在 `evals/reports/<实验名>.json` 与同名 `.md`。Markdown 报告会自动包含“指标如何计算”章节：说明每项指标的公式、实际命中题数/总题数、候选阶段的判定、延迟 percentile 的计算方式、失败归因差异，以及逐题明细文件的回查路径。

Embedding 使用第三方 OpenAI-compatible 服务（例如 SiliconFlow/Qwen）时，必须让服务端 tokenizer 处理原始字符串；项目通过 `EMBEDDING_CHECK_CTX_LENGTH=false` 关闭 LangChain 的 OpenAI `tiktoken` token-ID 输入路径。修改该设置或更换 Embedding 模型后，必须重建目标文件的全部向量，不能继续混用旧模型生成的向量。

| 参数 | 说明 |
|---|---|
| `--dataset` | 评测集 JSONL，默认 `evals/datasets/xiyouji_recall.jsonl` |
| `--file-id` / `--user-id` | 目标索引；`user-id` 必须与 `knowledge_files.user_id` 一致（多租户行级隔离） |
| `--k` | 最终 Top-K |
| `--with-context` / `--neighbor-window` | 是否做邻块上下文扩展及其窗口 |
| `--chapter-local` | 启用章节内二级精排（实验性，A/B 验证为负收益，见下） |
| `--rewrite` | 对每题先跑一次 LLM Query Preparation |
| `--preparation-cache` | 复用已冻结的改写输出，**做 A/B 时必须带**，否则 LLM 抖动会淹没实验信号 |
| `--write-preparation-cache` | 保存本次改写输出（须与 `--rewrite` 同用，否则报"缺少题目输出"） |
| `--baseline-report` | 指定历史报告做同口径对比 |

## 二、数据集

| 文件 | 作品 | 题数 | 片段数 | 构造方式 | 原始问题 R@10 | 改写后 R@10 |
|---|---|---:|---:|---|---:|---:|
| `xiyouji_recall.jsonl` | 西游记 | 40 | 1595 | story-grounded（先命题后定位） | **0.250** | **0.325** |
| `hongloumeng_recall.jsonl` | 红楼梦 | 40 | 1982 | chunk-grounded（先读片段后命题） | **0.750** | **0.775** |

> ⚠️ 对比两部作品时**必须对齐改写开关**。`--preparation-cache` 会载入并**应用**改写输出
> （报告里 `use_rewrite=true` / `query_preparation_mode=cache`），不加才是真正的原始问题。
> 曾因 `--preparation-cache` 被误当作"冻结用、不生效"而产生过一次误判。

两部作品的索引参数一致（bge-m3 / 1024d / chunk 650 / overlap 120）。

### ⚠️ 两个数据集的绝对分数不可直接比较

`0.325` 与 `0.750` 的差距**不是**"系统在红楼梦上更强"，而是构造方式不同造成的难度差：

- **chunk-grounded**（红楼梦）：读过片段再命题，题目与金标片段在语义上天然接近。
- **story-grounded**（西游记）：凭对故事的了解命题，再反查金标，语义距离更远。

差异在**候选池阶段**就已出现，而非排序阶段：

| 候选池召回 | 西游记 | 红楼梦 |
|---|---:|---:|
| 向量 | 0.375 | 0.850 |
| 词法 | 0.550 | 0.775 |
| RRF 融合 | 0.575 | 0.900 |

即：42.5% 的西游记题目，其金标片段根本没进 60 条候选池。

已排除一个常见替代解释——**原文重复导致的金标过严**。用 2-gram Jaccard 检查所有未命中题
（`near_dup_check`），西游记最高 0.211、红楼梦最高 0.097，均未达到"近似等价片段"水平
（阈值 0.25），故不存在"召回了同样正确的片段却被判错"的系统性误判。

**结论**：任何单一数据集的 R@10 都不能代表系统通用能力。跨作品比较时，必须标注构造方式。
若要把红楼梦数据集提升到可比水平，需要由未读过片段的人重新按 story-grounded 方式命题。

## 三、构造新数据集的硬性检查

命题必然在读过片段之后，极易把片段里的独特措辞抄进题目，让词法检索一步命中。
本项目已有过一次真实教训：红楼梦初版有 5 题的最长公共子串达 8~16 字（多为直引回目
与人物原话），去泄漏后 R@10 从 0.825 降到 0.750——**泄漏贡献了约 7.5 个百分点**。

```bash
cd E:/novel/backend
POSTGRES_HOST=127.0.0.1 ./.venv/Scripts/python.exe scripts/check_dataset_leak.py \
  ../evals/datasets/<你的数据集>.jsonl --max-leak 6 --verbose
```

该脚本同时校验两件事：

1. **金标正确性**：每个 `gold_chunk` 必须存在于库，且 `evidence_quote` 必须出现在该片段原文中。
2. **泄漏程度**：题目与金标片段原文的最长公共子串长度，默认阈值 6 字（现有两个数据集均为 5 字）。

退出码 0 = 通过，1 = 存在金标错误或泄漏超标。

## 四、当前基线（2026-08-28）

配置：`HYBRID_CANDIDATE_K=60` + `ENABLE_RERANKER=false` + `NOVEL_NEIGHBOR_WINDOW=1`。

### 4.1 原始问题（未启用 Query Rewriter）

| 指标 | 西游记 | 红楼梦 |
|---|---:|---:|
| Recall@1 | 0.125 | 0.550 |
| Recall@5 | 0.150 | 0.625 |
| **Recall@10** | **0.250** | **0.750** |
| MRR@10 | 0.150 | 0.596 |
| nDCG@10 | 0.172 | 0.631 |
| 章节命中率 | 0.500 | 0.875 |
| P95 延迟 | — | 602 ms |

西游记失败归因：`chapter_hit_but_gold_chunk_missed` 10、`candidate_generation_missed` 8、
`final_top_k_cut` 10、`rrf_fusion_loss` 2（命中 10）。

### 4.2 Query Rewriter 的增益：难题收益是简单题的 3 倍

| 作品 / 构造 | 未改写 | 改写后 | 绝对增益 | 相对增益 |
|---|---:|---:|---:|---:|
| 西游记 story-grounded（难） | 0.250 | 0.325 | **+0.075** | **+30.0%** |
| 红楼梦 chunk-grounded（易） | 0.750 | 0.775 | +0.025 | +3.3% |

改写器注入的 `retrieval_query` 是原文词（如 xyj-001「孙悟空最初是怎样诞生的？」→
「孙悟空 诞生 花果山 **仙石 孕育 石猴 出世**」，后四个词正是金标片段里的词），
确实在把 story-grounded 难题向"贴着原文的简单题"拉近。

**但它只补上了约 1/10 的差距**：未改写时差 0.500，改写后仍差 0.450。

**增益来自池内排序，不是池的召回能力**（这再次印证 §四.3 的瓶颈判断）：

| 西游记指标 | 未改写 | 改写后 |
|---|---:|---:|
| 向量候选召回 | 0.475 | **0.375** ↓ |
| 词法候选召回 | 0.525 | 0.550 ↑ |
| RRF 候选池召回 | 0.600 | **0.575** ↓ |
| `final_top_k_cut`（在池内但被 Top-10 截断） | 10 | **4** ↓ |
| 命中数 | 10 | **13** ↑ |

改写把查询拉长成一串关键词，**稀释了语义向量**（向量召回反降），但把金标片段的
**排名显著前移**（Recall@1 +80%、MRR +62%），于是"在池内却排第 11~30 名"的题被救了回来。

**代价**：改写要调一次 LLM，P95 从 602ms 升到 **2549ms**（红楼梦实测）。线上是否值得，
取决于你要的是召回率还是响应速度。

### 4.3 真正的瓶颈：池内排序

RRF 候选池已含金标的比例是 0.575~0.600，最终 Top-10 只留下 0.250~0.325。
要提升应改融合与池内排序，而不是继续加二级检索或加重排。

## 五、已验证的负结论（不要重复尝试）

| 结论 | 证据 |
|---|---|
| **Cross-encoder 重排：base 恒负，v2-m3 + BM25 新池后翻案（2026-08-29）** | **旧结论（bge-reranker-base + ILIKE 词法池）**：29 次受控实验无一胜出、延迟高 2~6 倍；健全性测试证明非截断问题（XLM-R 514 硬上限），双重 sigmoid 已修。**重测（bge-reranker-v2-m3 + BM25 池，两数据集 40 题、预注册标准）**：红楼梦 R@1 **+0.275**、MRR **+0.217**、nDCG +0.183（均显著），R@10 +0.075；西游记无显著变化；**纯重排优于 blend 保护**。**已默认开启**（`ENABLE_RERANKER=true` + `ENABLE_RERANKER_BLEND=false` + v2-m3，质量优先不设延迟限，检索 P95 约 130ms→3.9s）。教训：负结论也有保质期——池子构成（换 BM25）或模型（v2-m3 8K 上下文）变化后应重跑关键 A/B |
| **章节内二级检索为负收益** | A/B（`xiyouji_local_off_20260828` vs `xiyouji_local_on_20260828`）：R@10 0.325→**0.275**、MRR 0.242→**0.155**、P95 599→652ms。根因是该函数按 `score` 字段排序，而该字段在主检索文档上是纯向量余弦分，导致 RRF 融合序被丢弃、退化成纯向量检索。函数保留但默认关闭，详见 `rag.py` 的 docstring |
| **Query 改写：有效，但对简单题几乎无用** | 对 story-grounded 难题 R@10 +0.075（相对 +30%）；对 chunk-grounded 简单题仅 +0.025（+3.3%）。增益机制是**池内排序前移**（Recall@1 +80%、MRR +62%），不是扩大候选池——向量候选召回反而从 0.475 降到 0.375。代价是 P95 从 602ms 升到 2549ms。若只用 chunk-grounded 数据集评估，会严重低估改写的价值 |
| **multi_expert 未能兑现质量增益（西游记难题集）** | 答案侧 A/B（`xiyouji_answer_ab_20260828`，20 题 × 2 策略，LLM-as-judge 4 维）：multi_expert 相对 direct 的 faithfulness +0.10、completeness +0.05、relevance −0.15、citation_support +0.20，**全部 95% CI 含 0**；而 P95 延迟 7.4s→26.8s（显著）、LLM 调用 1→6。以现有证据，multi_expert 不值得 3~4 倍成本；如需复审可换 judge 模型重评（原始输出已落盘） |

## 六、答案侧评测（2026-08-28）

`evaluate_agent_answer.py`：检索评测之上的端到端评测，两阶段设计。

- **run 阶段**：冻结 Query（preparation cache）→ 无头运行 agent_graph（`direct` / `multi_expert`，逐字同查询）→ 收集最终答案、`[S#]` 引用、证据与 meta → `*_raw.jsonl` 落盘，judge 可重跑而不重跑 agent。
- **judge 阶段**：LLM-as-judge 四维打分（faithfulness / completeness / relevance / citation_support，1~5），judge 缓存冻结；`--judge-model` / `JUDGE_MODEL` 可换模型重评。
- **引用落地校验（无 LLM）**：未知引用率（幻觉引用）、引用指向金标片段比例。
- 所有差值带配对 bootstrap 95% CI（`eval_stats.py`）。

```bash
cd E:/novel/backend
# 金标答案生成（LLM 起草 + key_point 溯源校验，决策：不人工复核）
../backend/.venv/Scripts/python.exe ../evals/datasets/build_answer_dataset.py \
  --dataset ../evals/datasets/xiyouji_recall.jsonl --user-id <owner> --file-id <file_id>
# 端到端 A/B
./.venv/Scripts/python.exe scripts/evaluate_agent_answer.py \
  --dataset ../evals/datasets/xiyouji_answer.jsonl --user-id <owner> \
  --stage all --preparation-cache ../evals/datasets/xiyouji_query_preparation_20260826.json \
  --name <实验名>
```

**结果**：西游记（难题集）multi_expert 增益全部不显著、成本显著更高，见 §五 表。红楼梦组因 DeepSeek API 余额耗尽中断（`402 Insufficient Balance`），multi_expert 全部 20 题与 judge 阶段失败，需充值后重跑同命令。

**已知局限**：judge 与 agent 同模型（自评偏差）；n=20/策略；金标答案为 LLM 起草（经 key_point 溯源校验，18~20/20 通过），未人工复核。

## 七、检索融合与词法通道实验（2026-08-28）

- **重排器健全性测试**（`scripts/check_reranker_sanity.py`）：见 §五 重排行。
- **归一化加权融合**（`FUSION_MODE=weighted` + `VECTOR_WEIGHT`/`LEXICAL_WEIGHT`，默认关闭）：各通道分数池内 min-max 归一后加权，替代"只看名次"的 RRF；RRF 分数仍写入 metadata（重排保护依赖）。网格结果（40 题/作品，冻结 prep cache，配对 bootstrap 95% CI）：

  | 数据集 | rrf R@10/MRR | weighted 0.7向/0.3词 | weighted 0.5/0.5 | weighted 0.3向/0.7词 |
  |---|---|---|---|---|
  | 西游记（难） | 0.325 / 0.242 | 0.225 / 0.140（**显著变差**） | 0.300 / 0.173（MRR 显著变差） | 0.375 / 0.200（均不显著） |
  | 红楼梦（易） | 0.775 / 0.582 | 0.850 / 0.668（**MRR 显著更好**） | 0.850 / 0.642（MRR 显著更好） | 0.775 / 0.580（持平） |

  **结论**：加权融合的收益取决于向量分数在该语料上的可靠性——语义近距（chunk-grounded）语料上 MRR 显著提升，语义远距难题上向量权重反而显著有害（余弦分被压缩，名次比分数更稳健）。**默认保持 `rrf`**；语义相近型语料可尝试 `FUSION_MODE=weighted VECTOR_WEIGHT=0.7 LEXICAL_WEIGHT=0.3`。
- **BM25 词法通道**（`ENABLE_BM25_SEARCH`，**默认开启**，ParadeDB pg_search）：docker-compose 与 CI 均已换 `paradedb/paradedb:0.18.6-pg16` 镜像（数据卷兼容，pg16 tag 匹配现有卷）；迁移 `20260828_0012` 建 ngram 2-3 bm25 索引。**fail-fast 语义：BM25 报错直接抛出、不静默降级**——缺扩展/索引属于必须修复的环境配置错误；中文词法 → FTS 的回退链仅保留给 BM25 关闭的场景。

  **实库 A/B**（重建后的 ParadeDB 库，两数据集 40 题、冻结 prep cache、RRF 融合、Top-10，配对 bootstrap）：

  | 数据集 | 通道 | R@10 | MRR@10 | 词法候选召回 | P95 延迟 |
  |---|---|---:|---:|---:|---:|
  | 西游记（难） | BM25 **开** | 0.275 | 0.212 | **0.65** | **129ms** |
  | 西游记（难） | 旧词法（关） | 0.325 | 0.242 | 0.550 | 574ms |
  | 红楼梦（易） | BM25 **开** | **0.825** | 0.571 | **0.825** | **134ms** |
  | 红楼梦（易） | 旧词法（关） | 0.775 | 0.582 | 0.750 | 590ms |

  **结论**：BM25 的单通道候选召回显著更高（+0.10 / +0.075，验证 spike）、端到端延迟约 4 倍优；端到端 R@10 红楼梦 +0.05（不显著）、西游记 -0.05（不显著），但西游记 MRR 显著 -0.031（胜/平/负 0/38/2）——BM25 的 IDF 排序信号在难集的 RRF 融合中略逊于旧词法的"多 token 命中覆盖"排序。**默认开启**（召回与延迟收益为主，MRR 代价小且可随时 `ENABLE_BM25_SEARCH=false` 回退）；若难集排序质量优先于延迟，可关闭。

  重建记录（2026-08-29）：换镜像前 `pg_dumpall` 全库备份 → 旧数据实际不在 `novel_pg_data` 命名卷（新容器挂载后为空库触发全新 initdb）→ 用备份完整恢复（embeddings 7165 行 / knowledge_files 8 / users 275）→ `alembic stamp 20260826_0011`（恢复库的 alembic_version 落后于真实 schema，`init_db` 双轨制历史问题）→ `upgrade head` 建索引。备份留存 `backend/data/pg_backup_before_paradedb_20260829_114921.sql`。教训：**确认运行容器实际挂载的卷**是换存储引擎前最容易翻车的一步。

  spike 教训（`bm25_spike_20260828.md`）：①查询侧必须用 jieba 分词空格连接，中文整句默认 parse 无法命中 ngram 索引（实测 0 命中）；②`paradedb.score(embeddings.id)` 按 key_field 引用；③0.18.6 默认 PG17，须用 `-pg16` tag 才能复用 pg16 数据卷。

