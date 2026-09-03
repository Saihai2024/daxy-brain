"""
从 PostgreSQL 把已学习的语料灌入 CognitiveBrain。

每行一个记忆:topic 是检索 key,content 是回答。
"""
import os
import sys
import time

# add project root to sys.path so we can run via `python scripts/...`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.db_config import get_pg_conn
from brain.cognitive_brain import CognitiveBrain


def load(only_status="learned", min_credibility=0.5, max_topic_len=80, limit=None):
    # 文件锁 — consolidator_daemon 看到这个文件存在就跳过本轮 brain 写,避免 race
    lock = os.path.join(ROOT, "logs", "BRAIN_RELOAD_IN_PROGRESS")
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    with open(lock, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()} script=load_corpus_to_brain\n")
    try:
        _load(only_status, min_credibility, max_topic_len, limit)
    finally:
        try: os.remove(lock)
        except OSError: pass


def _load(only_status="learned", min_credibility=0.5, max_topic_len=80, limit=None,
          reset=True):
    brain = CognitiveBrain()
    print("brain stats before:", brain.stats())
    if reset:
        # 清空 memory list + feature_index,W1 保留(让灾后重灌得干净的 1:1 映射)
        brain.memory = []
        brain.feature_index = {}
        brain._key_matrix = None
        print(f"  [reset] cleared memory (W1 kept). brain ready to ingest from scratch.")

    conn = get_pg_conn()
    cur = conn.cursor()

    # Load sentence_corpus
    cur.execute(
        "SELECT topic, content, credibility FROM sentence_corpus "
        "WHERE status=%s AND credibility>=%s AND topic IS NOT NULL "
        "ORDER BY id" + (f" LIMIT {int(limit)}" if limit else ""),
        (only_status, min_credibility),
    )
    rows = cur.fetchall()
    print(f"sentence_corpus: {len(rows)} rows")
    t0 = time.time()
    for i, (topic, content, cred) in enumerate(rows):
        if not topic or not content:
            continue
        if len(topic) > max_topic_len:
            continue
        brain.add_memory(topic, content, source="sentence_corpus", credibility=float(cred), dedup_threshold=None)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(rows)}] {(i+1)/(time.time()-t0):.1f}/s  mem={len(brain.memory)}")
        if (i + 1) % 5000 == 0:
            brain.save()
            print(f"  [checkpoint] saved at i={i+1}")

    # Load web_corpus
    cur.execute(
        "SELECT topic, content, credibility FROM web_corpus "
        "WHERE status=%s AND credibility>=%s AND topic IS NOT NULL "
        "ORDER BY id" + (f" LIMIT {int(limit)}" if limit else ""),
        (only_status, min_credibility),
    )
    rows = cur.fetchall()
    print(f"web_corpus: {len(rows)} rows")
    for i, (topic, content, cred) in enumerate(rows):
        if not topic or not content:
            continue
        if len(topic) > max_topic_len:
            continue
        brain.add_memory(topic, content, source="web_corpus", credibility=float(cred), dedup_threshold=None)
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(rows)}]  mem={len(brain.memory)}")
        if (i + 1) % 5000 == 0:
            brain.save()
            print(f"  [checkpoint] saved at i={i+1}")

    cur.close()
    conn.close()

    brain.save()
    print("brain stats after:", brain.stats())

    # Quick demo
    for q in ["量子纠缠是什么", "光速", "黑洞", "你叫什么名字", "DNA"]:
        ans, dbg = brain.answer(q)
        if ans:
            preview = ans[:80].replace("\n", " ")
            print(f"\nQ: {q}\n  -> [{dbg['source']} sim={dbg['similarity']:.2f} cred={dbg['credibility']:.2f}]")
            print(f"     topic: {dbg['topic']}")
            print(f"     {preview}...")
        else:
            print(f"\nQ: {q}\n  -> NO MATCH ({dbg})")


if __name__ == "__main__":
    load()
