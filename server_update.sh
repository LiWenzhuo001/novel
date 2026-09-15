#!/usr/bin/env bash
# 小说智读 · 服务器一键更新（2026-09-15 版，目标迁移 0019）
# 用法：把本脚本与 /tmp/novel_src.tar.gz 上传到服务器后执行：
#   sudo bash /opt/novel/server_update.sh 2>&1 | tee /tmp/update.log
# 前提：/opt/novel 为既有部署；脚本全程幂等，失败即停，日志已含排障所需全部信息。
set -euo pipefail
cd /opt/novel

step() { echo; echo "======== $1 ========"; }

step "0. 数据库备份"
set -a; . ./.env 2>/dev/null; set +a
PGU="${POSTGRES_USER:-job_agent}"; PGD="${POSTGRES_DB:-job_agent}"
BACKUP="/opt/novel/backup_$(date +%F_%H%M).sql"
sudo docker compose exec -T postgres pg_dump -U "$PGU" "$PGD" > "$BACKUP"
echo "备份完成：$(ls -lh "$BACKUP" | awk '{print $5, $9}')"

step "1. 解包新代码（不触碰服务器本地 .env 与 data）"
sudo tar -xzf /tmp/novel_src.tar.gz -C /opt/novel

step "2. 构建镜像（backend 约 8-10 分钟，缓存命中会快很多）"
sudo docker compose build backend frontend

step "3. 数据库现状盘点"
INV=$(sudo docker compose exec -T postgres psql -U "$PGU" -d "$PGD" -At -c "
SELECT 'novel_characters=' || (SELECT count(*) FROM pg_tables WHERE tablename='novel_characters') || E'\n' ||
       'ttl_minutes=' || (SELECT count(*) FROM information_schema.columns WHERE table_name='agent_memories' AND column_name='ttl_minutes') || E'\n' ||
       'bm25=' || (SELECT count(*) FROM pg_indexes WHERE indexname='embeddings_bm25_idx') || E'\n' ||
       'emb_dim=' || COALESCE((SELECT atttypmod FROM pg_attribute WHERE attrelid='embeddings'::regclass AND attname='embedding')::text,'?')")
echo "$INV"
STAMP_BEFORE=$(sudo docker compose run --rm backend alembic current 2>/dev/null | tail -1 | awk '{print $1}')
echo "当前版本戳：${STAMP_BEFORE:-(空)}"

step "4. 结构签名判定与版本对齐"
READY=$(echo "$INV" | awk '/novel_characters=1/{a=1} /ttl_minutes=1/{b=1} /bm25=1/{c=1} END{print (a&&b&&c)?"yes":"no"}')
DIM=$(echo "$INV" | grep '^emb_dim=' | cut -d= -f2)
if [ "$READY" != "yes" ] || [ "$DIM" != "1024" ]; then
  echo "!! 结构签名不匹配（ready=$READY dim=$DIM）——停止。请把本日志发给开发确认，勿强行迁移。"
  exit 1
fi
# 版本对齐：0017/0018 已是 Alembic 权威链，直接升级；仅 0015 之前的
# create_all 塑形旧库才需要先 stamp 0015。0019 目标：memory_jobs 表 +
# chat_messages.status 列（新表 + 加列带 server_default，对存量数据安全）。
case "$STAMP_BEFORE" in
  20260912_0017|20260912_0018|20260912_0019)
    echo "版本戳 $STAMP_BEFORE 在权威链上，直接升级。"
    ;;
  20260905_0015|20260912_0016)
    echo "旧版本戳 $STAMP_BEFORE，先 stamp 20260905_0015 对齐。"
    sudo docker compose run --rm backend alembic stamp 20260905_0015
    ;;
  *)
    echo "!! 未知的迁移版本戳：$STAMP_BEFORE ——停止，请人工确认。"
    exit 1
    ;;
esac

step "5. 迁移到 0019（memory_jobs 任务表 + chat_messages.status 终态列）"
sudo docker compose run --rm backend alembic upgrade head
sudo docker compose run --rm backend alembic upgrade head   # 幂等复跑应无输出
sudo docker compose run --rm backend python scripts/check_schema.py

step "6. 重启全部服务"
sudo docker compose up -d --build

step "7. 验收"
sleep 8
sudo docker compose ps
echo "---- 前端(80) ----"; curl -sI http://127.0.0.1/ | head -1
echo "---- 后端健康 ----"; curl -s http://127.0.0.1:8000/health; echo
echo "---- 记忆 worker 就绪（日志应出现 memory_worker.started）----"
sudo docker compose logs backend --since 2m 2>/dev/null | grep -m1 memory_worker.started || echo "（未捕获到，稍后人工复查）"
echo "======== 更新完成：把本日志尾部贴回给开发确认 ========"
