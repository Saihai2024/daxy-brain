# -*- coding: utf-8 -*-
"""「思维逻辑」检索层 —— BM25,独立于 SDR 大脑。

为什么不用 brain_v2.pkl 那套 SDR:0824 实测 `CONTEXT_LEN=16`,编码器只看
文本最后 16 个字符,长任务简报 recall@1 = 0%,全库余弦均值 0.9982(空间坍缩)。
同一轮评测里 BM25 在 hard 组 67.5% / precision@5 83.3%,是当时最好的词法基线。
所以这一层直接从 BM25 起步,不掺 SDR。

分词:中文取字符 2-gram(不依赖 jieba,对路径/代码标识符稳),
ASCII 取整词 + 按非字母数字切出的子词(`apiAdapter.ts` → `apiadapter.ts`/`apiadapter`/`ts`)。
"""
import math
import os
import re
import threading
import time

_CJK = re.compile(r"[\u3400-\u9fff]")
_ASCII_TOK = re.compile(r"[a-z0-9][a-z0-9_./\-]*")
_SUB_TOK = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75
# 相对分阈值:top1 分数要达到「query 全部词都被这条语料解释掉」的这个比例才算命中。
# 用相对值而不是绝对分,是因为 BM25 绝对分随 query 长度线性涨,长短问题没法共用一个数。
#
# 0.30 是标定出来的(scripts/thinking_eval.py,24 条语料):
#   正例 R@1 100%(rel 最低 0.485)/ 百科 0/150 / 无关任务 0/8 / **同域难负例 1/60**。
#   降到 0.16 的话同域难负例会误触发 17% —— 那才是真正的风险(答一个像但不对的方案)。
MIN_REL_SCORE = float(os.environ.get("THINKING_MIN_REL", "0.30"))
# 落在 [MIN_REL_SCORE, HEDGE_REL) 之间的命中要说清楚"可能不完全对上"
HEDGE_REL = float(os.environ.get("THINKING_HEDGE_REL", "0.45"))
# 索引热重载间隔。语料是百来条量级,重载 = 一个小 PG 查询 + 建倒排,毫秒级,
# 没必要拖 10 分钟 —— 守护刚学会的东西要能尽快答得上,这是闭环的体感所在。
RELOAD_INTERVAL = float(os.environ.get("THINKING_RELOAD_SEC", "120"))

# 判「这是不是一个任务类提问」的词面信号
TASK_WORDS = (
    "怎么做", "如何", "怎么办", "该怎么", "帮我", "我要", "我想", "需要",
    "实现", "修复", "排查", "定位", "优化", "重构", "迁移", "部署", "接入",
    "调研", "评估", "梳理", "盘点", "设计", "方案", "拆解", "步骤",
    "为什么会", "根因", "报错", "失败", "不生效", "跑不起来",
)
MIN_TASK_LEN = int(os.environ.get("THINKING_MIN_LEN", "24"))

# 我们自己的对话记录在 thinking_corpus 里用这个 src 标记,**只做评测不做学习**。
# 用它当学习材料 = 用评测集训练,评出来的分虚高(典型的循环论证)。
EVAL_SRC = "transcript_eval"

# 学习缺口表:问了但思维逻辑路答不上的任务类问题,按出现次数排队等着去学
GAP_DDL = """
CREATE TABLE IF NOT EXISTS thinking_gaps (
    id         serial PRIMARY KEY,
    query      text UNIQUE NOT NULL,
    hits       integer NOT NULL DEFAULT 1,
    best_rel   real,
    status     text NOT NULL DEFAULT 'pending',
    first_seen timestamptz NOT NULL DEFAULT now(),
    last_seen  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS thinking_gaps_pick_idx
    ON thinking_gaps (status, hits DESC, last_seen DESC);
"""
GAP_UPSERT = """
INSERT INTO thinking_gaps (query, best_rel) VALUES (%s, %s)
ON CONFLICT (query) DO UPDATE SET
    hits      = thinking_gaps.hits + 1,
    best_rel  = GREATEST(COALESCE(thinking_gaps.best_rel, 0), EXCLUDED.best_rel),
    last_seen = now(),
    -- 学过一轮还是答不上,重新排队(可能是当时抓失败了)
    status    = CASE WHEN thinking_gaps.status = 'learned' THEN 'pending'
                     ELSE thinking_gaps.status END
"""


def tokenize(text):
    if not text:
        return []
    t = text.lower()
    toks = []
    for m in _ASCII_TOK.finditer(t):
        w = m.group(0)
        toks.append(w)
        subs = _SUB_TOK.findall(w)
        if len(subs) > 1:
            toks.extend(subs)
    cjk = "".join(_CJK.findall(t))
    toks.extend(cjk[i:i + 2] for i in range(len(cjk) - 1))
    return toks


def looks_like_task(msg):
    """够长、或带任务动词 —— 才让思维逻辑路参与。
    短的百科提问(「量子纠缠是什么」)一律不进来,避免抢掉大脑的主路径。"""
    if not msg:
        return False
    if len(msg) >= MIN_TASK_LEN:
        return True
    return any(w in msg for w in TASK_WORDS)


class ThinkingIndex:
    def __init__(self, rows=None, conn_factory=None):
        self._lock = threading.RLock()
        self._conn_factory = conn_factory
        self.rows = []
        self._loaded_at = 0.0
        if rows is not None:
            self._build(rows)
        elif conn_factory is not None:
            self.reload()

    # ---------------- 载入 ----------------
    def reload(self):
        if self._conn_factory is None:
            return
        conn = self._conn_factory()
        try:
            with conn.cursor() as cur:
                # ★ src='transcript_eval' 是我们自己的对话记录,**只当评测集**,
                #   绝不进学习/检索集 —— 否则评测就是拿训练数据自己考自己。
                cur.execute(
                    "SELECT id, src, src_id, task_text, task_type, steps, "
                    "tool_plan, pitfalls, acceptance FROM thinking_corpus "
                    "WHERE src <> %s", (EVAL_SRC,))
                cols = ("id", "src", "src_id", "task_text", "task_type",
                        "steps", "tool_plan", "pitfalls", "acceptance")
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception as e:
            # 表还没建 / 库不通:降级成空索引,绝不能拖垮 gRPC 主服务
            print(f"[thinking] 载入失败,按空索引处理: {e}")
            rows = []
        finally:
            try:
                conn.close()
            except Exception:
                pass
        self._build(rows)

    def maybe_reload(self):
        if self._conn_factory is None:
            return
        if time.time() - self._loaded_at < RELOAD_INTERVAL:
            return
        try:
            self.reload()
        except Exception as e:
            print(f"[thinking] reload error: {e}")

    def _build(self, rows):
        docs, dfs = [], {}
        for r in rows:
            # 检索面 = 任务原文 + 拆解动作 + 任务类型。刻意不放 tool args:
            # 里面全是本机路径,会让"仓库路径像"的任务互相串味。
            parts = [r.get("task_text") or "", r.get("task_type") or ""]
            for s in (r.get("steps") or []):
                parts.append(s.get("action") or "")
            tf = {}
            for w in tokenize(" ".join(parts)):
                tf[w] = tf.get(w, 0) + 1
            docs.append(tf)
            for w in tf:
                dfs[w] = dfs.get(w, 0) + 1
        n = len(docs)
        lens = [sum(d.values()) for d in docs]
        avg = (sum(lens) / n) if n else 1.0
        idf = {w: math.log(1 + (n - df + 0.5) / (df + 0.5)) for w, df in dfs.items()}
        # 语料里根本没有的词 = 最稀有,idf 取 df=0 的值。
        # ⚠️ 必须有这个默认值:否则算相对分时未知词按 idf=0 不进分母,
        # 「只有少数几个词碰巧匹配上」的无关问题会拿到接近满分(实测 rel 0.617)。
        idf_default = math.log(1 + (n + 0.5) / 0.5) if n else 1.0
        with self._lock:
            self.rows, self._docs, self._lens = rows, docs, lens
            self._avg, self._idf, self._n = avg or 1.0, idf, n
            self._idf_default = idf_default
            self._loaded_at = time.time()

    # ---------------- 检索 ----------------
    def _score(self, qtf, i):
        d, dl, s = self._docs[i], self._lens[i], 0.0
        for w, qn in qtf.items():
            f = d.get(w)
            if not f:
                continue
            s += self._idf.get(w, 0.0) * qn * (f * (K1 + 1)) / (
                f + K1 * (1 - B + B * dl / self._avg))
        return s

    def search(self, query, top_k=3):
        """返回 [(row, score, rel)],rel = score / query 的理论满分。"""
        with self._lock:
            if not self._n:
                return []
            qtf = {}
            for w in tokenize(query):
                qtf[w] = qtf.get(w, 0) + 1
            if not qtf:
                return []
            # 满分参照:每个 query term(**包括语料里没有的词**)都满额命中。
            # rel 因此读作「这个问题有多大比例被这条语料解释掉了」。
            ceiling = sum(self._idf.get(w, self._idf_default) * qn * (K1 + 1) / K1
                          for w, qn in qtf.items()) or 1.0
            scored = [(self._score(qtf, i), i) for i in range(self._n)]
            scored.sort(key=lambda x: -x[0])
            return [(self.rows[i], s, s / ceiling) for s, i in scored[:top_k] if s > 0]

    # ---------------- 学习缺口 ----------------
    def _record_gap(self, query, rel):
        """把「像任务但答不上」的问题记下来,交给 thinking_daemon 去学。

        这是持续学习闭环的入口:系统先承认自己不会,才谈得上去补。
        写库失败一律吞掉 —— 记缺口是锦上添花,绝不能影响回答。
        """
        if self._conn_factory is None:
            return
        try:
            conn = self._conn_factory()
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(GAP_DDL)
                        cur.execute(GAP_UPSERT, (query[:2000], float(rel)))
            finally:
                conn.close()
        except Exception as e:
            print(f"[thinking] 记录缺口失败(忽略): {e}")

    def answer(self, query, min_rel=None):
        """够像才答。返回 (文本, dbg) 或 (None, dbg)。"""
        self.maybe_reload()
        if not looks_like_task(query):
            return None, {"reason": "not_task_like"}
        hits = self.search(query, top_k=3)
        if not hits:
            self._record_gap(query, 0.0)
            return None, {"reason": "no_hit"}
        row, score, rel = hits[0]
        thr = MIN_REL_SCORE if min_rel is None else min_rel
        if rel < thr:
            self._record_gap(query, rel)
            return None, {"reason": "below_rel", "rel": round(rel, 4),
                          "top": (row.get("task_text") or "")[:40]}
        return format_row(row, hedge=rel < HEDGE_REL), {
            "src": row.get("src"), "src_id": row.get("src_id"),
            "rel": round(rel, 4), "score": round(score, 2),
            "alternatives": [((r.get("task_text") or "")[:30], round(v, 3))
                             for r, _, v in hits[1:]],
        }


def format_row(row, hedge=False):
    """把一条思维逻辑渲染成回答。hedge=True 时明说匹配没那么确定。"""
    out = []
    task = (row.get("task_text") or "").strip().replace("\n", " ")
    if hedge:
        out.append("不太确定对不对得上,我见过的最接近的一件是:%s…" % task[:60])
    else:
        out.append("这类任务我见过类似的:%s…" % task[:60])
    if row.get("task_type"):
        out.append("【任务类型】%s" % row["task_type"])
    steps = row.get("steps") or []
    if steps:
        out.append("【拆解步骤】")
        for s in steps:
            line = "  %s. %s" % (s.get("n"), s.get("action") or "")
            if s.get("why"):
                line += " —— %s" % s["why"]
            out.append(line)
    tools = row.get("tool_plan") or []
    if tools:
        out.append("【每步用什么工具】")
        for t in tools:
            out.append("  步骤%s → %s:%s" % (t.get("step"), t.get("tool"),
                                             (t.get("args") or "")[:70]))
    pit = row.get("pitfalls") or []
    if pit:
        out.append("【坑】")
        out.extend("  - %s" % p for p in pit[:4])
    acc = row.get("acceptance") or []
    if acc:
        out.append("【验收】" + ";".join(a for a in acc[:3]))
    return "\n".join(out)
