# -*- coding: utf-8 -*-
"""Build a multi-turn follow-up recall test set that actually exercises Query Rewriter.

Each case carries `history` (prior turns) + a follow-up `query` that starts with a
pronoun (so query_rewriter._needs_rewrite matches) but omits the subject, so the
rewriter MUST resolve the reference to retrieve the right chapter. Gold chapter/no
is auto-resolved from the bge-m3 index by searching a distinctive phrase under a
target chapter, guaranteeing the gold chunk really exists.

Run:  cd backend && POSTGRES_HOST=127.0.0.1 ./.venv/Scripts/python.exe scripts/build_multiturn_dataset.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from app.db import AsyncSessionLocal

FILE_ID = "c959fb4c1a30"
USER_ID = "c1e78368550f498787e3871ed9291b63"
OUT = Path(__file__).resolve().parents[2] / "evals" / "datasets" / "xiyouji_recall_multiturn.jsonl"

# id, category, history(role/content), follow-up query, gold_chapter, phrases(any-of)
CHAINS = [
    {
        "id": "xyj-mt-01", "category": "character",
        "history": [
            {"role": "user", "content": "孙悟空是怎么学成本领的？"},
            {"role": "assistant", "content": "孙悟空漂洋过海到西牛贺洲，拜菩提祖师为师，在灵台方寸山学艺多年。"},
        ],
        "query": "他主要都学了哪些本事？",
        "gold_chapter": 2, "phrases": ["七十二般变化", "七十二变", "筋斗云", "腾云驾雾"],
    },
    {
        "id": "xyj-mt-02", "category": "plot",
        "history": [
            {"role": "user", "content": "孙悟空是怎么大闹天宫的？"},
            {"role": "assistant", "content": "他偷吃蟠桃、搅乱蟠桃会、偷吃金丹，与天兵天将大战，被二郎神擒住后投进了太上老君的八卦炉。"},
        ],
        "query": "他在炉子里炼出了什么本事？",
        "gold_chapter": 7, "phrases": ["火眼金睛", "开炉取丹"],
    },
    {
        "id": "xyj-mt-03", "category": "timeline",
        "history": [
            {"role": "user", "content": "孙悟空为什么被压在五行山下？"},
            {"role": "assistant", "content": "他大闹天宫，和如来佛祖打赌输了，被如来翻手化作五行山压住，一压就是五百年。"},
        ],
        "query": "后来是谁把他救出来的？",
        "gold_chapter": 14, "phrases": ["揭去金字压帖", "揭了压帖", "六字真言", "太保刘伯钦"],
    },
    {
        "id": "xyj-mt-04", "category": "character",
        "history": [
            {"role": "user", "content": "唐僧在高老庄收了个徒弟，是谁？"},
            {"role": "assistant", "content": "是猪八戒。他本是天蓬元帅，因调戏嫦娥被贬下凡，错投了猪胎，在高老庄被悟空收服。"},
        ],
        "query": "他以前是做什么官的？",
        "gold_chapter": 19, "phrases": ["天蓬元帅", "总督水兵", "天河"],
    },
    {
        "id": "xyj-mt-05", "category": "character",
        "history": [
            {"role": "user", "content": "师徒三人在流沙河遇到一个妖怪，最后怎么解决的？"},
            {"role": "assistant", "content": "那妖怪被观音派来的惠岸行者劝化，剃度做了沙和尚，帮助唐僧师徒渡过了流沙河，一同西行。"},
        ],
        "query": "他原来是做什么的？",
        "gold_chapter": 22, "phrases": ["卷帘大将", "蟠桃会上", "失手打碎玻璃盏"],
    },
    {
        "id": "xyj-mt-06", "category": "plot",
        "history": [
            {"role": "user", "content": "白骨精是怎么害人的？"},
            {"role": "assistant", "content": "白骨精先后变成村姑、老妇人、老公公来欺骗唐僧，但都被孙悟空的火眼金睛识破。"},
        ],
        "query": "那个妖怪最终落得什么下场？",
        "gold_chapter": 27, "phrases": ["白骨", "打死", "一堆粉骷髅", "脊梁上有一行字"],
    },
    {
        "id": "xyj-mt-07", "category": "plot",
        "history": [
            {"role": "user", "content": "红孩儿捉走了唐僧，还火烧了孙悟空，后来怎么样了？"},
            {"role": "assistant", "content": "孙悟空去请观音菩萨，用净瓶甘露灭了三昧真火，降伏了红孩儿。"},
        ],
        "query": "这个妖怪最后被谁收做了徒弟？",
        "gold_chapter": 42, "phrases": ["善财童子", "观音", "莲花台", "金箍"],
    },
    {
        "id": "xyj-mt-08", "category": "plot",
        "history": [
            {"role": "user", "content": "火焰山挡住了去路，孙悟空向谁借扇子？"},
            {"role": "assistant", "content": "他先找铁扇公主借芭蕉扇，公主因红孩儿的恩怨不肯借，还把他扇飞了。"},
        ],
        "query": "他最后是怎么把火灭掉的？",
        "gold_chapter": 61, "phrases": ["芭蕉扇", "连扇四十九扇", "火焰山", "琉璃盏"],
    },
    {
        "id": "xyj-mt-09", "category": "plot",
        "history": [
            {"role": "user", "content": "真假美猴王那段，为什么连观音都分辨不出？"},
            {"role": "assistant", "content": "六耳猕猴变化成孙悟空的样子，本领相当，观音、照妖镜都辨别不出真假，两个猴王一路打到如来面前。"},
        ],
        "query": "最后是谁分出真假的？",
        "gold_chapter": 58, "phrases": ["如来", "六耳猕猴", "金钵盂", "谛听"],
    },
    {
        "id": "xyj-mt-10", "category": "plot",
        "history": [
            {"role": "user", "content": "唐僧在女儿国被女王挽留，本国太师来提亲。国王非要招他做驸马，怎么办？"},
            {"role": "assistant", "content": "他装病推脱，又用了缓兵之计让徒弟们先出城，最后借机脱身。"},
        ],
        "query": "他们是怎么从女儿国脱身的？",
        "gold_chapter": 54, "phrases": ["倒换关文", "关文", "女国", "假意应承"],
    },
    {
        "id": "xyj-mt-11", "category": "character",
        "history": [
            {"role": "user", "content": "孙悟空头上戴的金箍到底怎么回事？"},
            {"role": "assistant", "content": "那是如来做的一个紧箍，观音把它当骗孙悟空戴上去的，戴上便摘不下来。"},
        ],
        "query": "这紧箍咒是谁传下来的？",
        "gold_chapter": 8, "phrases": ["紧箍儿", "紧箍咒", "三个箍", "金紧禁"],
    },
    {
        "id": "xyj-mt-12", "category": "object",
        "history": [
            {"role": "user", "content": "孙悟空的金箍棒是从哪儿弄来的？"},
            {"role": "assistant", "content": "他去东海龙宫寻宝器，龙王给他看了很多兵器都不称心，最后发现了那根如意金箍棒。"},
        ],
        "query": "这根棒子原本是谁留下的？",
        "gold_chapter": 3, "phrases": ["大禹治水", "定海神针", "天河定底", "定海之珍"],
    },
    {
        "id": "xyj-mt-13", "category": "timeline",
        "history": [
            {"role": "user", "content": "师徒四人历经多少难才到灵山？"},
            {"role": "assistant", "content": "他们一路上经历了九九八十一难，最终到达灵山圣境，拜见如来佛祖。"},
        ],
        "query": "他们最后拿到真经了吗？",
        "gold_chapter": 98, "phrases": ["阿傩", "优钵罗", "真经", "如来"],
    },
    {
        "id": "xyj-mt-14", "category": "plot",
        "history": [
            {"role": "user", "content": "托塔天王和哪吒为什么来捉孙悟空？"},
            {"role": "assistant", "content": "因为孙悟空嫌弼马温官小，反下天宫自封齐天大圣，又偷吃了蟠桃和金丹，玉帝派天兵天将来征讨。"},
        ],
        "query": "他偷的蟠桃是从哪场盛会偷的？",
        "gold_chapter": 5, "phrases": ["蟠桃胜会", "蟠桃会", "齐天府", "偷桃"],
    },
]


async def resolve() -> list[dict]:
    async with AsyncSessionLocal() as s:
        # Preload all (chapter_no, chunk_no, content) to resolve in memory.
        rows = (await s.execute(text(
            "select chapter_no, chunk_no, content from embeddings where file_id=:f and user_id=:u"
        ), {"f": FILE_ID, "u": USER_ID})).all()
        print(f"index chunks loaded: {len(rows)}")
        idx: dict[tuple[int, int], str] = {(r.chapter_no, r.chunk_no): r.content for r in rows}
    cases: list[dict] = []
    missing: list[tuple[str, int]] = []
    for c in CHAINS:
        hits = []
        for (chapter_no, chunk_no), content in sorted(idx.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if chapter_no != c["gold_chapter"]:
                continue
            for phrase in c["phrases"]:
                if phrase in content:
                    hits.append((chapter_no, chunk_no, phrase, content))
                    break
            if hits:
                break
        if not hits:
            missing.append((c["id"], c["gold_chapter"]))
            continue
        chapter_no, chunk_no, phrase, content = hits[0]
        cases.append({
            "id": c["id"],
            "category": c["category"],
            "query": c["query"],
            "history": c["history"],
            "file_id": FILE_ID,
            "gold_chapters": [chapter_no],
            "gold_chunks": [{
                "chapter_no": chapter_no,
                "chunk_no": chunk_no,
                "relevance": 2,
                "evidence_phrase": phrase,
                "evidence_preview": content[:120].replace("\n", " ").strip(),
            }],
        })
        print(f"  {c['id']} -> ch{chapter_no}/chunk{chunk_no}  phrase={phrase!r}")
    print(f"\nresolved {len(cases)}/{len(CHAINS)}")
    if missing:
        print("MISSING phrases (need manual fix):", missing)
    out_dir = OUT.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print("wrote", OUT)
    return cases


if __name__ == "__main__":
    asyncio.run(resolve())