"""把待判定 JSONL 转成便于人工快速判定的紧凑文本。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "evals" / "pooling" / "judging"
OUT_DIR = ROOT / "evals" / "pooling" / "judging_txt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for path in sorted(IN_DIR.glob("batch_*.jsonl")):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines = []
    for r in rows:
        lines.append(f"[{r['id']}] {r['category']} | {r['query']}")
        lines.append(f"    判定标准: {r['narrative']}")
        for i, c in enumerate(r["candidates"], start=1):
            lines.append(f"    ({i}) c{c['chunk_no']} ch{c['chapter_no']} | {c['text']}")
        lines.append("")
    out = OUT_DIR / path.name.replace(".jsonl", ".txt")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"{out.name}: {len(rows)} 题, {out.stat().st_size} 字节")
