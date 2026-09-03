# -*- coding: utf-8 -*-
"""把 thinking_corpus 里**来自本系统自己对话记录**的条目划成评测集,不再参与学习。

规矩:对话记录只当评测集,不能当学习材料 —— 拿它训练就是用评测集训练,
分数会虚高。早期批量采集时这类条目以 src='qwen_plan' 混进了库,这个脚本
负责把它们摘出来改标记。

识别办法:src='qwen_plan' 且 task_text 不是任何一条 thinking_gaps.query
(gap 学来的那些是真实提问,可以留在学习集里,只是当时 src 打错了,顺手改成 'gap')。

改完 ThinkingIndex 只加载 src <> 'transcript_eval',数据仍在库里供评测用。

用法:
  python scripts/split_eval_corpus.py            # 只看会怎么改
  python scripts/split_eval_corpus.py --apply
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.db_config import get_pg_conn

EVAL_SRC = "transcript_eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    conn = get_pg_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM thinking_corpus")
            print("thinking_corpus 总数:", cur.fetchone()[0])
            cur.execute("SELECT src, count(*) FROM thinking_corpus GROUP BY src")
            for s, c in cur.fetchall():
                print("  src=%-16s %d" % (s, c))

            # gap 学来的:task_text 能在 thinking_gaps.query 里找到
            cur.execute("""
                SELECT c.id, g.id FROM thinking_corpus c
                JOIN thinking_gaps g ON g.query = c.task_text
                WHERE c.src = 'qwen_plan'
            """)
            gap_rows = cur.fetchall()
            # 其余 qwen_plan 的都来自对话记录
            cur.execute("""
                SELECT count(*) FROM thinking_corpus c
                WHERE c.src='qwen_plan'
                  AND NOT EXISTS (SELECT 1 FROM thinking_gaps g
                                  WHERE g.query = c.task_text)
            """)
            n_eval = cur.fetchone()[0]
            print("\n判定结果:")
            print("  来自真实提问(gap),改 src='gap' 留在学习集:", len(gap_rows))
            print("  来自对话记录,改 src='%s' 移出学习集: %d" % (EVAL_SRC, n_eval))

            if not args.apply:
                print("\n(未加 --apply,什么都没改)")
                return 0

            for cid, gid in gap_rows:
                cur.execute("UPDATE thinking_corpus SET src='gap', src_id=%s "
                            "WHERE id=%s", (gid, cid))
            cur.execute("""
                UPDATE thinking_corpus SET src=%s
                WHERE src='qwen_plan'
                  AND NOT EXISTS (SELECT 1 FROM thinking_gaps g
                                  WHERE g.query = thinking_corpus.task_text)
            """, (EVAL_SRC,))
            print("  已改 %d 条为评测集" % cur.rowcount)

            cur.execute("SELECT src, count(*) FROM thinking_corpus GROUP BY src "
                        "ORDER BY src")
            print("\n改后分布:")
            for s, c in cur.fetchall():
                mark = "  ← 评测集,不参与学习" if s == EVAL_SRC else ""
                print("  src=%-16s %d%s" % (s, c, mark))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
