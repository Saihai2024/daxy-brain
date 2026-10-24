# -*- coding: utf-8 -*-
"""思维链持续学习守护 —— 让认知大脑「知道自己不会什么,然后去学」。

闭环:
    用户问任务类问题 → 思维逻辑路 rel 不够 → 记进 PG thinking_gaps
                    ↓
    本守护挑「问得最多又答不上」的,调用推理后端(深度思考模式)
                    ↓
    解析五段(任务类型/拆解步骤/工具计划/坑/验收)→ 入 thinking_corpus
                    ↓
    ThinkingIndex 每 120s 自动热重载 → 下次同类问题能答

推理后端支持两种模式:
  1. OpenAI 兼容接口(配置环境变量 THINKING_LLM_URL,如 http://127.0.0.1:8010/v1/chat/completions)
  2. 外部 QwenFetcher 爬虫模块(兼容原外部工具环境)
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.thinking_parse import parse_answer
from utils.db_config import PG_DSN, get_pg_conn

LOG_PATH = Path(os.environ.get("THINKING_LOG", str(ROOT / "logs" / "thinking.log")))
LLM_URL = os.environ.get("THINKING_LLM_URL", "")
LLM_MODEL = os.environ.get("THINKING_LLM_MODEL", "Qwen3.6-35B-A3B")
LLM_API_KEY = os.environ.get("THINKING_LLM_API_KEY", "")
MAX_FAILS = int(os.environ.get("THINKING_MAX_FAILS", "4"))

log = logging.getLogger("thinking")

PROMPT = """{task}

────────────────────────
以上是派给一个资深工程 agent 的真实任务。现在请你扮演这个 agent 的**决策器**。

硬约束(必须遵守):
1. 你**没有**这台机器的文件访问权限。任务里出现的路径、仓库、文件一律当作
   "确实存在但你现在看不到",**不要去验证、不要列目录、不要因为找不到而中断或改口**。
2. 不要执行代码。你的产出是"我会怎么干",不是"我干完了"。
3. 严格按下面五段输出,不要加别的段落。

## 任务类型
一行:调研 / 实现 / 修复 / 优化 / 迁移 / 评估 中的一个 + 一句话理由。

## 拆解步骤
3-8 步,每步一行,格式:`N. 动作 —— 为什么这一步排在这个位置`

## 工具计划
对上面每一步写出具体要执行什么,一行一条,格式:
`步骤N | 工具 | 具体参数`
工具**只能**从这个清单里选:
read_file / list_dir / grep / bash / git / web_search / run_python / ask_user
示例:`步骤2 | grep | 在 src/ 下搜 "rate.?limit",定位 header 解析处`

## 坑与假设
2-5 条。这类任务最容易踩的坑,以及你为了推进做了哪些假设。

## 验收标准
2-4 条。怎么算这个任务做完了。
"""

SEED_PROMPT = """请列举 {n} 个**真实软件工程场景**下会派给资深工程师的任务,
主题限定在:{topic}。

每条要求:
- 120-250 字,写成"派活"的口吻(背景 + 要做什么 + 约束),不要写成问句
- **具体**:带上技术栈、文件/目录、数据量级、时限或不停机之类的真实约束
- 彼此不重复,覆盖不同子场景

严格用下面的格式输出,不要加编号以外的任何解释:
===TASK===
(第一条任务正文)
===TASK===
(第二条任务正文)
...
"""

SEED_TOPICS = [
    "数据库迁移与 schema 变更(不停机、大表、回滚)",
    "线上故障排查(超时、内存泄漏、连接池耗尽、死锁)",
    "接口与服务性能优化(P99 延迟、缓存、并发)",
    "CI/CD 与发布流程(门禁、灰度、回滚、制品)",
    "前端问题定位(渲染卡顿、状态错乱、打包体积)",
    "爬虫与反爬对抗(风控、代理、验证码、限速)",
    "数据管道与一致性(双写、对账、幂等、补数)",
    "可观测性建设(日志、指标、链路追踪、告警)",
    "安全加固与权限(越权、密钥泄漏、审计)",
    "代码重构与技术债(拆巨型文件、去重复、加测试)",
]

SEED_DDL = """
CREATE TABLE IF NOT EXISTS thinking_seeds (
    id         serial PRIMARY KEY,
    topic      text,
    task_text  text UNIQUE NOT NULL,
    status     text NOT NULL DEFAULT 'pending',
    fails      integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS thinking_seeds_pick_idx ON thinking_seeds (status, id);
ALTER TABLE thinking_seeds ADD COLUMN IF NOT EXISTS fails integer NOT NULL DEFAULT 0;
"""

GAP_DDL = """
CREATE TABLE IF NOT EXISTS thinking_gaps (
    id         serial PRIMARY KEY,
    query      text UNIQUE NOT NULL,
    hits       integer NOT NULL DEFAULT 1,
    best_rel   real,
    status     text NOT NULL DEFAULT 'pending',
    fails      integer NOT NULL DEFAULT 0,
    first_seen timestamptz NOT NULL DEFAULT now(),
    last_seen  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS thinking_gaps_pick_idx
    ON thinking_gaps (status, hits DESC, last_seen DESC);
ALTER TABLE thinking_gaps ADD COLUMN IF NOT EXISTS fails integer NOT NULL DEFAULT 0;
"""

CORPUS_DDL = """
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

_db_lock = threading.Lock()


class OpenAICompatibleFetcher:
    """基于 OpenAI /v1/chat/completions 接口的推理器(如本地 llama-server)。"""
    def __init__(self, url: str, model: str, api_key: str = ""):
        self.url = url
        self.model = model
        self.api_key = api_key

    def fetch_sync(self, prompt: str, _unused1=None, timeout: int = 300, _unused2=None):
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        thinking_steps = [s.strip() for s in reasoning.splitlines() if s.strip()] if reasoning else []
        return {
            "parsed": {
                "answer_text": content,
                "thinking_steps": thinking_steps,
            }
        }


def setup_log():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(str(LOG_PATH), encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])


def pg():
    return get_pg_conn()


def ensure_tables():
    conn = pg()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(CORPUS_DDL)
            cur.execute(GAP_DDL)
            cur.execute(SEED_DDL)
            for tbl in ("thinking_seeds", "thinking_gaps"):
                cur.execute(
                    f"UPDATE {tbl} SET status='pending' "
                    "WHERE status='failed' AND fails < %s",
                    (MAX_FAILS,),
                )
                if cur.rowcount:
                    log.info("[daemon] %s 放回 %d 条历史失败题", tbl, cur.rowcount)
    finally:
        conn.close()


def pick_tasks(n):
    out = []
    conn = pg()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, query FROM thinking_gaps WHERE status='pending' "
                "ORDER BY fails, hits DESC, last_seen DESC LIMIT %s", (n,))
            out += [("gap", r[0], r[1]) for r in cur.fetchall()]
            if len(out) < n:
                cur.execute(
                    "SELECT id, task_text FROM thinking_seeds "
                    "WHERE status='pending' ORDER BY fails, id LIMIT %s",
                    (n - len(out),))
                out += [("selfgen", r[0], r[1]) for r in cur.fetchall()]
    finally:
        conn.close()
    return out


def seed_pool_size():
    conn = pg()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM thinking_seeds WHERE status='pending'")
            return cur.fetchone()[0]
    finally:
        conn.close()


def generate_seeds(fetcher, topic, n=20):
    q = SEED_PROMPT.format(n=n, topic=topic)
    try:
        res = fetcher.fetch_sync(q, None, 300, None) or {}
    except Exception as e:
        log.error("[seed] 生成失败 topic=%s: %s", topic, e)
        return 0
    text = ((res.get("parsed") or {}).get("answer_text") or "")
    parts = [p.strip() for p in text.split("===TASK===")]
    tasks = [p for p in parts if 80 <= len(p) <= 1200]
    if not tasks:
        log.warning("[seed] topic=%s 解析不出任务(答案 %d 字)", topic, len(text))
        return 0
    added = 0
    conn = pg()
    try:
        with conn, conn.cursor() as cur:
            for t in tasks:
                cur.execute(
                    "INSERT INTO thinking_seeds (topic, task_text) VALUES (%s,%s) "
                    "ON CONFLICT (task_text) DO NOTHING", (topic, t))
                added += cur.rowcount
    finally:
        conn.close()
    log.info("[seed] topic=%s 新增 %d 条题目(解析出 %d)", topic, added, len(tasks))
    return added


def _mark(cur, src, src_id, status):
    if src == "gap":
        cur.execute("UPDATE thinking_gaps SET status=%s WHERE id=%s",
                    (status, int(src_id)))
    elif src == "selfgen":
        cur.execute("UPDATE thinking_seeds SET status=%s WHERE id=%s",
                    (status, int(src_id)))


def store(src, src_id, task_text, rec):
    tt, steps, tools, pit, acc = parse_answer(rec["answer_text"])
    if not steps and not tools:
        return False, "解析不出步骤/工具计划"
    with _db_lock:
        conn = pg()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(UPSERT, (
                    src, int(src_id), task_text, tt,
                    json.dumps(steps, ensure_ascii=False),
                    json.dumps(tools, ensure_ascii=False),
                    json.dumps(pit, ensure_ascii=False),
                    json.dumps(acc, ensure_ascii=False),
                    json.dumps(rec.get("thinking_steps") or [], ensure_ascii=False),
                    rec["answer_text"], rec.get("elapsed")))
                _mark(cur, src, src_id, "learned")
        finally:
            conn.close()
    return True, "%d 步 / %d 条工具计划" % (len(steps), len(tools))


def mark_failed(src, src_id):
    tbl = {"gap": "thinking_gaps", "selfgen": "thinking_seeds"}.get(src)
    if not tbl:
        return
    try:
        with _db_lock:
            conn = pg()
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE {tbl} SET fails = fails + 1, "
                        "  status = CASE WHEN fails + 1 >= %s THEN 'failed' "
                        "                ELSE 'pending' END "
                        "WHERE id = %s", (MAX_FAILS, int(src_id)))
            finally:
                conn.close()
    except Exception as e:
        log.error("[%s#%s] 标记失败次数出错: %s", src, src_id, e)


def learn_one(fetcher, item):
    src, sid, text = item
    log.info("[%s#%s] 开抓 (%d 字): %s", src, sid, len(text),
             " ".join(text.split())[:60])
    t0 = time.time()
    try:
        res = fetcher.fetch_sync(PROMPT.format(task=text), None, 300, None) or {}
    except Exception as e:
        log.error("[%s#%s] 异常: %s", src, sid, e)
        mark_failed(src, sid)
        return False
    parsed = res.get("parsed") or {}
    rec = {"answer_text": parsed.get("answer_text") or "",
           "thinking_steps": parsed.get("thinking_steps") or [],
           "elapsed": round(time.time() - t0, 1)}
    if len(rec["answer_text"]) < 200:
        log.warning("[%s#%s] 答案过短(%d 字),留待重跑", src, sid, len(rec["answer_text"]))
        mark_failed(src, sid)
        return False
    try:
        saved, why = store(src, sid, text, rec)
    except Exception as e:
        log.error("[%s#%s] 入库失败: %s", src, sid, e)
        return False
    if saved:
        log.info("[%s#%s] OK %.1fs %s", src, sid, rec["elapsed"], why)
        return True
    log.warning("[%s#%s] %s", src, sid, why)
    mark_failed(src, sid)
    return False


def one_round(fetcher, per_round, concurrency, seed_low_water, seed_batch):
    try:
        if seed_pool_size() < seed_low_water:
            topic = SEED_TOPICS[int(time.time() // 3600) % len(SEED_TOPICS)]
            generate_seeds(fetcher, topic, n=seed_batch)
    except Exception as e:
        log.error("[seed] 补题异常: %s", e)

    tasks = pick_tasks(per_round)
    if not tasks:
        log.info("本轮没有待学题目")
        return 0, 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(lambda it: learn_one(fetcher, it), tasks))
    ok = sum(1 for r in results if r)
    return ok, len(results) - ok


def _build_fetcher():
    if LLM_URL:
        log.info("[daemon] 使用 OpenAI 兼容接口: %s (model=%s)", LLM_URL, LLM_MODEL)
        return OpenAICompatibleFetcher(LLM_URL, LLM_MODEL, LLM_API_KEY)
    from utils.crawler.qwen import QwenFetcher
    return QwenFetcher()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=120, help="轮间隔秒")
    ap.add_argument("--per-round", type=int, default=12, help="每轮采几条")
    ap.add_argument("--concurrency", type=int,
                    default=int(os.environ.get("THINKING_CONCURRENCY", "2")),
                    help="并发请求数")
    ap.add_argument("--seed-batch", type=int, default=20,
                    help="一次生成多少道题")
    ap.add_argument("--seed-low-water", type=int, default=24,
                    help="待学题目少于这个数就自动生成一批")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    setup_log()
    log.info("[daemon] starting interval=%ss per_round=%s concurrency=%s",
             args.interval, args.per_round, args.concurrency)
    ensure_tables()

    fetcher = _build_fetcher()
    while True:
        t0 = time.time()
        try:
            ok, fail = one_round(fetcher, args.per_round, args.concurrency,
                                 args.seed_low_water, args.seed_batch)
            tot = ok + fail
            rate = (fail / tot) if tot else 0.0
            log.info("[daemon] 本轮 成功%d 失败%d(失败率%.0f%%) 用时%.0fs 并发%d",
                     ok, fail, rate * 100, time.time() - t0, args.concurrency)
        except Exception as e:
            log.error("[daemon] 轮次异常: %s\n%s", e, traceback.format_exc())
        if args.once:
            return 0
        time.sleep(max(30, args.interval))


if __name__ == "__main__":
    sys.exit(main())
