# -*- coding: utf-8 -*-
"""把 Qwen 采集结果解析成结构化「思维逻辑」条目,灌进 PG thinking_corpus。

**刻意不进 brain_v2.pkl**:0824 已实测那套 SDR 编码器只看文本末 16 个字符
(CONTEXT_LEN=16),几百上千字的任务简报一律被截成尾巴,recall@1=0%。
思维逻辑走独立的词法检索层(brain/thinking_index.py)。

采集端提示词固定输出五段(见 tools/qwen_batch_plan.py),这里按段落解析:
  ## 任务类型 / ## 拆解步骤 / ## 工具计划 / ## 坑与假设 / ## 验收标准

用法:
  python scripts/thinking_ingest.py <harvest目录> [--src qwen_plan]
"""
import argparse
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.db_config import get_pg_conn
from brain.thinking_parse import parse_answer

DDL = """
CREATE TABLE IF NOT EXISTS thinking_corpus (
    id            serial PRIMARY KEY,
    src           text NOT NULL,
    src_id        integer NOT NULL,
    task_text     text NOT NULL,
    task_type     text,
    steps         jsonb NOT NULL DEFAULT '[]'::jsonb,
    tool_plan     jsonb NOT NULL DEFAULT '[]'::jsonb,
    pitfalls      jsonb NOT NULL DEFAULT '[]'::jsonb,
    acceptance    jsonb NOT NULL DEFAULT '[]'::jsonb,
    qwen_thinking jsonb NOT NULL DEFAULT '[]'::jsonb,
    answer_text   text,
    elapsed       real,
    created_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (src, src_id)
);
CREATE INDEX IF NOT EXISTS thinking_corpus_src_idx ON thinking_corpus (src);
"""

UPSERT = """
INSERT INTO thinking_corpus
    (src, src_id, task_text, task_type, steps, tool_plan, pitfalls,
     acceptance, qwen_thinking, answer_text, elapsed)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (src, src_id) DO UPDATE SET
    task_text=EXCLUDED.task_text, task_type=EXCLUDED.task_type,
    steps=EXCLUDED.steps, tool_plan=EXCLUDED.tool_plan,
    pitfalls=EXCLUDED.pitfalls, acceptance=EXCLUDED.acceptance,
    qwen_thinking=EXCLUDED.qwen_thinking, answer_text=EXCLUDED.answer_text,
    elapsed=EXCLUDED.elapsed
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("harvest_dir")
    ap.add_argument("--src", default="qwen_plan")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.harvest_dir)
                   if f.startswith("task_") and f.endswith(".json"))
    if not files:
        print("no task_*.json in", args.harvest_dir)
        return 1

    recs, n_skip = [], 0
    for fn in files:
        with io.open(os.path.join(args.harvest_dir, fn), encoding="utf-8") as f:
            d = json.load(f)
        tt, steps, tools, pit, acc = parse_answer(d.get("answer_text") or "")
        if not steps and not tools:
            print("  [skip] id=%s 解析不出步骤/工具计划" % d.get("id"))
            n_skip += 1
            continue
        recs.append((args.src, int(d["id"]), d.get("task_text") or "", tt,
                     json.dumps(steps, ensure_ascii=False),
                     json.dumps(tools, ensure_ascii=False),
                     json.dumps(pit, ensure_ascii=False),
                     json.dumps(acc, ensure_ascii=False),
                     json.dumps(d.get("thinking_steps") or [], ensure_ascii=False),
                     d.get("answer_text") or "", d.get("elapsed")))

    print("解析出 %d 条,跳过 %d 条" % (len(recs), n_skip))
    if args.dry_run:
        for r in recs[:2]:
            print("--- id=%s type=%s" % (r[1], r[3]))
            print("  steps:", r[4][:300])
            print("  tools:", r[5][:300])
        return 0

    conn = get_pg_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(DDL)
                for r in recs:
                    cur.execute(UPSERT, r)
                cur.execute(
                    "SELECT src, count(*), sum(jsonb_array_length(steps)), "
                    "sum(jsonb_array_length(tool_plan)) "
                    "FROM thinking_corpus GROUP BY src ORDER BY src")
                rows = cur.fetchall()
    finally:
        conn.close()
    for src, cnt, st, tp in rows:
        print("  %-12s 条数=%-5d 拆解步=%-5s 工具计划行=%s" % (src, cnt, st, tp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
