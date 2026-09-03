# -*- coding: utf-8 -*-
"""思维逻辑路的阈值标定 + 回归风险评估。

两组 query 一起扫,才知道阈值该定在哪:
  正例 = 从每条语料自己的任务原文里截出的短问法(模拟用户用更短的话问同一件事)
  负例 = 百科提问(web_corpus 词条) + 与语料无关的任务类提问
目标:正例 R@1 高、且负例几乎全部被 rel 阈值挡在门外(不能抢大脑的百科主路径)。
"""
import io
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.db_config import get_pg_conn
from brain.thinking_index import ThinkingIndex, looks_like_task

random.seed(11)
idx = ThinkingIndex(conn_factory=get_pg_conn)
print("语料 %d 条" % len(idx.rows))
if not idx.rows:
    sys.exit("thinking_corpus 是空的,先跑 thinking_ingest.py")


def short_form(text, n):
    """截出前 n 个非空白字符当作"更短的问法"。"""
    s = " ".join((text or "").split())
    return s[:n]


pos = []           # (query, 期望的 src_id)
for r in idx.rows:
    for n in (60, 120, 200):
        q = short_form(r["task_text"], n)
        if len(q) >= 20:
            pos.append((q, r["src_id"]))

conn = get_pg_conn()
with conn.cursor() as cur:
    cur.execute("SELECT topic FROM web_corpus WHERE status='learned' "
                "AND char_length(topic) BETWEEN 2 AND 12 ORDER BY random() LIMIT 150")
    topics = [t[0] for t in cur.fetchall()]
conn.close()

TPL = ["%s是什么", "介绍一下%s", "%s有什么用", "什么是%s", "%s怎么样"]
neg_baike = [TPL[i % len(TPL)] % t for i, t in enumerate(topics)]
neg_task = [
    "帮我把家里的路由器换成 mesh 组网,应该怎么规划",
    "我想学做红烧肉,详细步骤是什么",
    "帮我写一封请年假的邮件,语气客气一点",
    "怎么给猫做绝育术后护理,需要注意什么",
    "我要报名马拉松,三个月训练计划怎么排",
    "帮我规划一趟五天的京都行程,预算八千",
    "公司要做等保三级测评,流程和材料有哪些",
    "怎么把 excel 里的两列数据做匹配去重",
    "我想给孩子选一个编程启蒙课程,怎么挑",
    "帮我排查一下汽车怠速抖动的原因",
]

print("\n正例 %d 条 / 百科负例 %d 条 / 无关任务负例 %d 条"
      % (len(pos), len(neg_baike), len(neg_task)))

# 先看有多少负例连"像任务"这道门都过不了
gated_baike = [q for q in neg_baike if looks_like_task(q)]
gated_task = [q for q in neg_task if looks_like_task(q)]
print("looks_like_task 门:百科负例通过 %d/%d,无关任务负例通过 %d/%d"
      % (len(gated_baike), len(neg_baike), len(gated_task), len(neg_task)))


def rels(queries):
    out = []
    for q in queries:
        h = idx.search(q, top_k=1)
        out.append((q, h[0][0]["src_id"] if h else None, h[0][2] if h else 0.0))
    return out


pos_r = rels([q for q, _ in pos])
gold = [g for _, g in pos]
nb_r = rels(gated_baike)
nt_r = rels(gated_task)

# 难负例:**没采集过的**工程任务简报。它们和语料同域(同一批项目、同样的
# 路径/术语),最容易被 BM25 误判成"我见过类似的"。这才是真正的回归风险。
HELDOUT = os.environ.get("THINKING_HELDOUT", "")
hard = []
if HELDOUT and os.path.exists(HELDOUT):
    import json as _json
    seen = {r["src_id"] for r in idx.rows}
    pool = []
    for line in io.open(HELDOUT, encoding="utf-8"):
        d = _json.loads(line)
        if d.get("id") in seen:
            continue
        t = " ".join((d.get("text") or "").split())
        if len(t) >= 60:
            pool.append(t[:150])
    hard = random.sample(pool, min(60, len(pool)))
    print("难负例(未采集过的同域工程任务)%d 条" % len(hard))
else:
    print("难负例语料缺失,跳过:%s" % HELDOUT)
hd_r = rels([q for q in hard if looks_like_task(q)])

print("\n%-8s %-13s %-12s %-12s %s"
      % ("阈值", "正例R@1", "百科误触发", "无关误触发", "难负例误触发(同域)"))
for thr in (0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.25, 0.30, 0.40):
    hit = sum(1 for (q, sid, rel), g in zip(pos_r, gold) if sid == g and rel >= thr)
    fb = sum(1 for _, _, rel in nb_r if rel >= thr)
    ft = sum(1 for _, _, rel in nt_r if rel >= thr)
    fh = sum(1 for _, _, rel in hd_r if rel >= thr)
    print("%-8.2f %-13s %-12s %-12s %s"
          % (thr, "%d/%d (%.0f%%)" % (hit, len(pos_r), 100 * hit / len(pos_r)),
             "%d/%d" % (fb, len(nb_r) or 1), "%d/%d" % (ft, len(nt_r) or 1),
             "%d/%d (%.0f%%)" % (fh, len(hd_r) or 1,
                                 100 * fh / max(len(hd_r), 1))))

prs = sorted(rel for _, _, rel in pos_r)
print("\n正例 rel 分位: min=%.3f p10=%.3f p50=%.3f"
      % (prs[0], prs[len(prs) // 10], prs[len(prs) // 2]))
if hd_r:
    hrs = sorted((rel for _, _, rel in hd_r), reverse=True)
    print("难负例 rel 最高 5: %s" % ", ".join("%.3f" % v for v in hrs[:5]))

wrong = [(q, sid, g, rel) for (q, sid, rel), g in zip(pos_r, gold) if sid != g]
print("\n正例里检索到别条的 %d 例(前 5):" % len(wrong))
for q, sid, g, rel in wrong[:5]:
    print("  rel=%.3f 期望 id=%s 实得 id=%s | %s" % (rel, g, sid, q[:60]))

print("\n负例 rel 最高的 5 条(阈值必须高于它们):")
for q, sid, rel in sorted(nb_r + nt_r, key=lambda x: -x[2])[:5]:
    print("  rel=%.3f -> id=%s | %s" % (rel, sid, q[:60]))
