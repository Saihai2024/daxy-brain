"""
FastAPI 仪表盘:
  - 浏览 PG 各表(web_corpus / sentence_corpus / vision_corpus / training_log)
  - 看认知大脑状态 + 记忆库
  - 在线测试问答(转发到本地 gRPC 服务)
"""
import json
import os
import sys
import time
import shutil
import tempfile
import threading
from typing import Optional
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.db_config import get_pg_conn, load_dotenv_if_present
load_dotenv_if_present()

# 把 ffmpeg 加进 PATH(Whisper 转码音频要)
FFMPEG_DIR = os.environ.get("FFMPEG_DIR", "")
if FFMPEG_DIR and os.path.isdir(FFMPEG_DIR):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ["PATH"]

from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

import grpc
from chatbot_grpc import chatbot_pb2, chatbot_pb2_grpc
from webapp import auth

app = FastAPI(title="Cognitive Brain Dashboard")

GRPC_TARGET = os.environ.get("GRPC_TARGET", "127.0.0.1:9999")
INDEX_PATH = Path(__file__).parent / "index.html"
LOGIN_PATH = Path(__file__).parent / "login.html"

# 无需登录即可访问的路径
PUBLIC_PATHS = {"/login", "/api/login", "/favicon.ico"}


@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    if path not in PUBLIC_PATHS:
        token = request.cookies.get(auth.COOKIE_NAME)
        if not auth.verify_token(token):
            if path.startswith("/api/"):
                return JSONResponse({"detail": "未登录或会话已过期"}, status_code=401)
            return RedirectResponse("/login", status_code=302)
    return await call_next(request)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    # 已登录就直接回首页
    if auth.verify_token(request.cookies.get(auth.COOKIE_NAME)):
        return RedirectResponse("/", status_code=302)
    return LOGIN_PATH.read_text(encoding="utf-8")


class LoginReq(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def login(req: LoginReq):
    if not auth.check_credentials(req.username, req.password):
        time.sleep(0.5)  # 减缓暴力破解
        raise HTTPException(401, "账号或密码错误")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        auth.COOKIE_NAME, auth.make_token(req.username),
        max_age=auth.SESSION_TTL, httponly=True, samesite="lax",
    )
    return resp


@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(auth.COOKIE_NAME)
    return resp

# 持久化的 gRPC channel
_channel = None
_stub = None
_stub_lock = threading.Lock()


def _new_stub():
    global _channel, _stub
    if _channel is not None:
        try:
            _channel.close()
        except Exception:
            pass
    _channel = grpc.insecure_channel(GRPC_TARGET)
    _stub = chatbot_pb2_grpc.ChatBotServiceStub(_channel)
    return _stub


def get_stub():
    global _stub
    with _stub_lock:
        if _stub is None:
            _new_stub()
    return _stub


def refresh_stub():
    """gRPC 连接出错时重建"""
    with _stub_lock:
        return _new_stub()


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_PATH.read_text(encoding="utf-8")


def _fetch_table(table: str, columns: str, status: Optional[str], q: str,
                 page: int, size: int, order_by: str = "created_at DESC"):
    where = []
    params = []
    if status and status != "all":
        where.append("status = %s")
        params.append(status)
    if q:
        where.append("(topic ILIKE %s OR content ILIKE %s)" if "topic" in columns
                     else "(question ILIKE %s OR answer ILIKE %s)")
        params.extend([f"%{q}%", f"%{q}%"])
    where_sql = "WHERE " + " AND ".join(where) if where else ""

    conn = get_pg_conn()
    with conn, conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table} {where_sql}", params)
        total = cur.fetchone()[0]
        offset = (page - 1) * size
        cur.execute(
            f"SELECT {columns} FROM {table} {where_sql} ORDER BY {order_by} "
            f"LIMIT %s OFFSET %s",
            params + [size, offset],
        )
        rows = cur.fetchall()
    conn.close()
    return total, rows


@app.get("/api/web_corpus")
def web_corpus(page: int = 1, size: int = 20, status: Optional[str] = None, q: str = ""):
    total, rows = _fetch_table(
        "web_corpus", "id, topic, content, status, credibility, created_at",
        status, q, page, size,
    )
    return {
        "total": total, "page": page, "size": size,
        "items": [
            {"id": r[0], "topic": r[1], "content": r[2],
             "status": r[3], "credibility": r[4],
             "created_at": str(r[5])} for r in rows
        ],
    }


@app.get("/api/sentence_corpus")
def sentence_corpus(page: int = 1, size: int = 20, status: Optional[str] = None, q: str = ""):
    total, rows = _fetch_table(
        "sentence_corpus", "id, topic, content, status, credibility, created_at",
        status, q, page, size,
    )
    return {
        "total": total, "page": page, "size": size,
        "items": [
            {"id": r[0], "topic": r[1], "content": r[2],
             "status": r[3], "credibility": r[4],
             "created_at": str(r[5])} for r in rows
        ],
    }


@app.get("/api/vision_corpus")
def vision_corpus(page: int = 1, size: int = 20, status: Optional[str] = None, q: str = ""):
    where = []
    params = []
    if status and status != "all":
        where.append("status = %s")
        params.append(status)
    if q:
        where.append("(question ILIKE %s OR answer ILIKE %s OR image_path ILIKE %s)")
        params.extend([f"%{q}%", f"%{q}%", f"%{q}%"])
    where_sql = "WHERE " + " AND ".join(where) if where else ""
    conn = get_pg_conn()
    with conn, conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM vision_corpus {where_sql}", params)
        total = cur.fetchone()[0]
        offset = (page - 1) * size
        cur.execute(
            f"SELECT id, image_path, question, answer, surprise_score, status, created_at "
            f"FROM vision_corpus {where_sql} ORDER BY created_at DESC LIMIT %s OFFSET %s",
            params + [size, offset],
        )
        rows = cur.fetchall()
    conn.close()
    return {
        "total": total, "page": page, "size": size,
        "items": [
            {"id": r[0], "image_path": r[1], "question": r[2], "answer": r[3],
             "surprise_score": r[4], "status": r[5], "created_at": str(r[6])} for r in rows
        ],
    }


@app.get("/api/training_log")
def training_log(page: int = 1, size: int = 20):
    total, rows = _fetch_table(
        "training_log", "id, ts, task, record_id, image_path, epoch, loss, note",
        None, "", page, size, order_by="ts DESC",
    )
    return {
        "total": total, "page": page, "size": size,
        "items": [
            {"id": r[0], "ts": str(r[1]), "task": r[2], "record_id": r[3],
             "image_path": r[4], "epoch": r[5], "loss": r[6], "note": r[7]}
            for r in rows
        ],
    }


@app.get("/api/stats")
def stats():
    """各表统计 + brain 状态"""
    out = {"tables": {}}
    conn = get_pg_conn()
    with conn, conn.cursor() as cur:
        for tbl in ["sentence_corpus", "web_corpus", "vision_corpus", "training_log"]:
            cur.execute(f"SELECT COUNT(*) FROM {tbl}")
            n = cur.fetchone()[0]
            cur.execute(
                f"SELECT status, COUNT(*) FROM {tbl} "
                f"WHERE status IS NOT NULL GROUP BY status"
            ) if tbl != "training_log" else cur.execute("SELECT 1 WHERE FALSE")
            by_status = dict(cur.fetchall()) if tbl != "training_log" else {}
            out["tables"][tbl] = {"count": n, "by_status": by_status}
    conn.close()

    # brain stats — 优先读 daemon 写的 stats.json(几 KB,毫秒级),
    # 避免每次 /api/stats 都 load 1+ GB 的 brain.pkl(会让 FastAPI OOM)
    stats_json_path = ROOT / "brain" / "models" / "brain_v2.pkl.stats.json"
    if stats_json_path.is_file():
        try:
            out["brain"] = json.loads(stats_json_path.read_text(encoding="utf-8"))
            return out
        except Exception:
            pass  # 落到下面兜底
    try:
        out["brain"] = _get_brain().stats()
    except Exception as e:
        out["brain"] = {"error": str(e)}
    return out


@app.get("/api/memory")
def list_memory(page: int = 1, size: int = 20, q: str = "", source: Optional[str] = None):
    """大脑记忆条目列表(只读)"""
    b = _get_brain()
    items = b.memory
    if source:
        items = [m for m in items if m["source"] == source]
    if q:
        items = [m for m in items if q in m["topic"] or q in m["content"]]
    total = len(items)
    items = items[(page - 1) * size:(page - 1) * size + size]
    return {
        "total": total, "page": page, "size": size,
        "items": [
            {"topic": m["topic"], "content": m["content"][:300],
             "source": m["source"], "credibility": m["credibility"],
             "hits": m["hits"]} for m in items
        ],
    }


_brain = None
_brain_mtime = 0.0
_brain_check_lock = threading.Lock()
_brain_last_check = 0.0


def _get_brain():
    """带 mtime 检查的 brain 单例 — daemon 写新 brain.pkl 后,
    下一次 API 调用最多 2 秒就能拿到最新数据。"""
    global _brain, _brain_mtime, _brain_last_check
    from brain.cognitive_brain import CognitiveBrain
    pkl_path = ROOT / "brain" / "models" / "brain_v2.pkl"
    now = time.time()
    # 节流:每 2 秒最多检查一次 mtime,避免高 QPS 下频繁 stat
    if _brain is not None and (now - _brain_last_check) < 2.0:
        return _brain
    with _brain_check_lock:
        try:
            mt = pkl_path.stat().st_mtime
        except OSError:
            mt = 0.0
        # 第一次或文件比上次记录的新 -> reload
        if _brain is None or mt > _brain_mtime + 0.5:
            try:
                _brain = CognitiveBrain()
                _brain_mtime = mt
                if _brain_last_check > 0:  # 不是第一次
                    print(f"[fastapi] brain hot-reloaded "
                          f"(memories={len(_brain.memory)}, steps={_brain.steps})", flush=True)
            except Exception as e:
                if _brain is None:
                    raise
                print(f"[fastapi] brain reload failed: {e}", flush=True)
        _brain_last_check = now
    return _brain


class ChatReq(BaseModel):
    message: str
    user_id: Optional[str] = "web"


@app.post("/api/chat")
def chat(req: ChatReq):
    if not req.message.strip():
        raise HTTPException(400, "empty message")
    grpc_req = chatbot_pb2.ReceivedMessage(
        user_id=req.user_id, user_message=req.message,
        create_time=int(time.time()),
    )
    last_err = None
    for attempt in range(2):
        try:
            stub = get_stub() if attempt == 0 else refresh_stub()
            resp = stub.Chatbot(grpc_req, timeout=20.0)
            return {"response": resp.user_response, "ts": resp.create_time}
        except grpc.RpcError as e:
            last_err = e
            continue
    raise HTTPException(503, f"gRPC unavailable: {last_err.details() if last_err else 'unknown'}")



# =====================================================================
# 学习流水线(持续学习看板)
# =====================================================================
# 两条学习线:
#   百科线   爬虫 → 巩固 → 大脑          (crawler/consolidator daemon)
#   思维链线 缺口队列 → 问 qwen → 思维语料 (thinking daemon)
# 存活判据用**日志 mtime**,不去枚举进程:思维链守护跑在另一套环境、
# 脚本也在本仓库之外,枚举进程既麻烦又容易被杀软打断。
_DAEMON_LOGS = {
    "crawler":      ROOT / "logs" / "crawler.log",
    "consolidator": ROOT / "logs" / "consolidator.log",
    "training":     ROOT / "logs" / "training.log",
    "thinking":     Path(os.environ.get("THINKING_LOG",
                                        str(ROOT / "logs" / "thinking.log"))),
}
# 各守护的轮询周期(秒),用来判"多久没动静算掉线"
_DAEMON_PERIOD = {"crawler": 3600, "consolidator": 600,
                  "training": 1800, "thinking": 1800}


def _daemon_health():
    out = {}
    now = time.time()
    for name, p in _DAEMON_LOGS.items():
        try:
            mt = p.stat().st_mtime
            idle = now - mt
        except OSError:
            out[name] = {"alive": False, "idle_s": None, "reason": "无日志"}
            continue
        # 宽限 2.5 个周期:一轮本身要跑一会儿,别把正常工作误判成掉线
        limit = _DAEMON_PERIOD.get(name, 1800) * 2.5
        out[name] = {"alive": idle < limit, "idle_s": int(idle),
                     "last": int(mt)}
    return out


@app.get("/api/learning/pipeline")
def learning_pipeline():
    """学习流水线全景:两条线各段的存活 / 速率 / 存量。"""
    res = {"daemons": _daemon_health(), "baike": {}, "thinking": {}}
    conn = get_pg_conn()
    try:
        with conn, conn.cursor() as cur:
            # ---- 百科线 ----
            cur.execute(
                "SELECT count(*) FILTER (WHERE created_at > now() - interval '1 hour'), "
                "       count(*) FILTER (WHERE created_at > now() - interval '24 hours'), "
                "       count(*) FILTER (WHERE status='pending'), count(*) "
                "FROM web_corpus")
            h1, h24, pending, total = cur.fetchone()
            res["baike"]["crawl"] = {"last_1h": h1, "last_24h": h24,
                                     "pending": pending, "total": total}
            cur.execute(
                "SELECT count(*) FILTER (WHERE created_at > now() - interval '24 hours'), "
                "       count(*) FILTER (WHERE status='learned') FROM web_corpus")
            c24, learned = cur.fetchone()
            cur.execute("SELECT count(*) FROM sentence_corpus WHERE status='pending'")
            res["baike"]["consolidate"] = {"learned": learned,
                                           "sent_pending": cur.fetchone()[0],
                                           "last_24h": c24}
            # ---- 思维链线 ----
            # ★ 只统计**学习集**:src='transcript_eval' 是我们自己的对话记录,
            #   按规矩只当评测集,不能算进"学到了多少"。
            cur.execute(
                "SELECT count(*), "
                "       coalesce(sum(jsonb_array_length(steps)),0), "
                "       coalesce(sum(jsonb_array_length(tool_plan)),0), "
                "       max(extract(epoch from created_at))::bigint "
                "FROM thinking_corpus WHERE src <> 'transcript_eval'")
            n, steps, tools, last_ts = cur.fetchone()
            cur.execute("SELECT count(*) FROM thinking_corpus "
                        "WHERE src = 'transcript_eval'")
            res["thinking"]["eval_holdout"] = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM thinking_seeds "
                        "WHERE status='pending'")
            try:
                res["thinking"]["seeds_pending"] = cur.fetchone()[0]
            except Exception:
                res["thinking"]["seeds_pending"] = 0
            res["thinking"]["corpus"] = {
                "count": n, "steps": int(steps), "tools": int(tools),
                "last_ts": last_ts,
                "avg_steps": round(steps / n, 1) if n else 0}
            cur.execute(
                "SELECT status, count(*) FROM thinking_gaps GROUP BY status")
            g = dict(cur.fetchall())
            res["thinking"]["gaps"] = {
                "pending": g.get("pending", 0), "learned": g.get("learned", 0),
                "failed": g.get("failed", 0)}
            cur.execute(
                "SELECT count(*) FROM thinking_corpus "
                "WHERE created_at > now() - interval '24 hours' "
                "  AND src <> 'transcript_eval'")
            res["thinking"]["corpus"]["last_24h"] = cur.fetchone()[0]
    except Exception as e:
        res["error"] = str(e)
    finally:
        conn.close()
    # 大脑存量走 stats.json(几 KB),别为这个 load 大脑
    try:
        sp = ROOT / "brain" / "models" / "brain_v2.pkl.stats.json"
        res["brain"] = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        res["brain"] = {}
    return res


@app.get("/api/learning/recent")
def learning_recent(limit: int = 12):
    """最近学到的思维链条目(时间倒序)。"""
    conn = get_pg_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, task_text, task_type, "
                "       jsonb_array_length(steps), jsonb_array_length(tool_plan), "
                "       extract(epoch from created_at)::bigint, elapsed "
                "FROM thinking_corpus WHERE src <> 'transcript_eval' "
                "ORDER BY created_at DESC, id DESC LIMIT %s",
                (max(1, min(limit, 50)),))
            rows = [{"id": r[0], "task": (r[1] or "")[:160], "type": r[2],
                     "steps": r[3], "tools": r[4], "ts": r[5],
                     "elapsed": r[6]} for r in cur.fetchall()]
    finally:
        conn.close()
    return {"items": rows}


@app.get("/api/learning/detail")
def learning_detail(id: int):
    """一条思维链的全文:拆解步骤 + 每步工具 + 坑 + 验收。"""
    conn = get_pg_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, task_text, task_type, steps, tool_plan, pitfalls, "
                "       acceptance, elapsed, extract(epoch from created_at)::bigint "
                "FROM thinking_corpus WHERE id=%s", (id,))
            r = cur.fetchone()
    finally:
        conn.close()
    if not r:
        raise HTTPException(404, "not found")
    return {"id": r[0], "task": r[1], "type": r[2], "steps": r[3],
            "tool_plan": r[4], "pitfalls": r[5], "acceptance": r[6],
            "elapsed": r[7], "ts": r[8]}


@app.get("/api/learning/gaps")
def learning_gaps(limit: int = 20):
    """待学队列:问过但答不上的任务类问题,按被问次数排。"""
    conn = get_pg_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, query, hits, best_rel, status, "
                "       extract(epoch from last_seen)::bigint "
                "FROM thinking_gaps ORDER BY "
                "  (status='pending') DESC, hits DESC, last_seen DESC LIMIT %s",
                (max(1, min(limit, 100)),))
            rows = [{"id": r[0], "query": r[1][:200], "hits": r[2],
                     "rel": round(r[3], 3) if r[3] is not None else None,
                     "status": r[4], "ts": r[5]} for r in cur.fetchall()]
    except Exception as e:
        return {"items": [], "error": str(e)}
    finally:
        conn.close()
    return {"items": rows}


@app.get("/api/thinking/list")
def thinking_list(page: int = 1, size: int = 20, q: str = "",
                  src: Optional[str] = None):
    """浏览面向 Qwen 学到的全部思维链语料。

    src=learn 只看学习集 / src=eval 只看评测集(我们自己的对话记录) / 空=全部。
    评测集条目会带 is_eval 标记,前端要明确区分 —— 它们不参与检索。
    """
    page = max(1, page)
    size = max(1, min(size, 100))
    where, params = [], []
    if src == "learn":
        where.append("src <> 'transcript_eval'")
    elif src == "eval":
        where.append("src = 'transcript_eval'")
    if q.strip():
        where.append("(task_text ILIKE %s OR answer_text ILIKE %s)")
        params += ["%%%s%%" % q.strip(), "%%%s%%" % q.strip()]
    wsql = ("WHERE " + " AND ".join(where)) if where else ""
    conn = get_pg_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM thinking_corpus " + wsql, params)
            total = cur.fetchone()[0]
            cur.execute(
                "SELECT id, src, task_text, task_type, "
                "       jsonb_array_length(steps), jsonb_array_length(tool_plan), "
                "       jsonb_array_length(pitfalls), "
                "       extract(epoch from created_at)::bigint, elapsed "
                "FROM thinking_corpus " + wsql +
                " ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s",
                params + [size, (page - 1) * size])
            items = [{"id": r[0], "src": r[1], "is_eval": r[1] == "transcript_eval",
                      "task": (r[2] or "")[:300], "type": r[3], "steps": r[4],
                      "tools": r[5], "pitfalls": r[6], "ts": r[7],
                      "elapsed": r[8]} for r in cur.fetchall()]
            cur.execute(
                "SELECT count(*) FILTER (WHERE src <> 'transcript_eval'), "
                "       count(*) FILTER (WHERE src = 'transcript_eval'), "
                "       coalesce(sum(jsonb_array_length(steps)) "
                "                FILTER (WHERE src <> 'transcript_eval'),0), "
                "       coalesce(sum(jsonb_array_length(tool_plan)) "
                "                FILTER (WHERE src <> 'transcript_eval'),0) "
                "FROM thinking_corpus")
            nl, ne, st, tl = cur.fetchone()
    finally:
        conn.close()
    return {"total": total, "page": page, "size": size, "items": items,
            "summary": {"learn": nl, "eval": ne, "steps": int(st),
                        "tools": int(tl)}}


@app.get("/api/thinking/tools")
def thinking_tools():
    """工具使用分布:Qwen 在这些任务里最常安排哪些工具(只统计学习集)。"""
    conn = get_pg_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT t->>'tool' AS tool, count(*) FROM thinking_corpus c, "
                "     jsonb_array_elements(c.tool_plan) t "
                "WHERE c.src <> 'transcript_eval' AND t->>'tool' IS NOT NULL "
                "GROUP BY 1 ORDER BY 2 DESC")
            rows = [{"tool": r[0], "n": r[1]} for r in cur.fetchall()]
            cur.execute(
                "SELECT c.task_type, count(*) FROM thinking_corpus c "
                "WHERE c.src <> 'transcript_eval' AND c.task_type IS NOT NULL "
                "GROUP BY 1 ORDER BY 2 DESC LIMIT 8")
            types = [{"type": (r[0] or "")[:24], "n": r[1]} for r in cur.fetchall()]
    except Exception as e:
        return {"tools": [], "types": [], "error": str(e)}
    finally:
        conn.close()
    return {"tools": rows, "types": types}


class CrawlReq(BaseModel):
    topics: list
    credibility: Optional[float] = 0.7


@app.post("/api/crawl")
def crawl(req: CrawlReq):
    """触发百度百科抓取一批主题。同步返回结果(限速,慢)。"""
    if not req.topics:
        raise HTTPException(400, "no topics")
    topics = [t.strip() for t in req.topics if t.strip()][:30]  # cap 30
    from scripts.web_crawler import fetch_baike, insert_pending
    import time as _t, random as _r
    results = []
    conn = get_pg_conn()
    try:
        for i, topic in enumerate(topics):
            if i > 0:
                _t.sleep(_r.uniform(1.0, 1.8))
            txt, status = fetch_baike(topic)
            if txt:
                with conn:
                    with conn.cursor() as cur:
                        rid, was_new = insert_pending(cur, topic, txt, req.credibility)
                results.append({
                    "topic": topic, "status": "ok" if was_new else "exists",
                    "id": rid, "length": len(txt), "preview": txt[:120],
                })
            else:
                results.append({"topic": topic, "status": status, "preview": ""})
    finally:
        conn.close()
    return {"results": results,
            "summary": {
                "ok": sum(1 for r in results if r["status"] == "ok"),
                "exists": sum(1 for r in results if r["status"] == "exists"),
                "fail": sum(1 for r in results if r["status"] not in ("ok", "exists")),
            }}


@app.post("/api/consolidate")
def consolidate(min_age_seconds: int = 0):
    """触发一次 daemon 巩固扫描(同步)。仅用于 UI 手动触发。"""
    import argparse as _ap
    from scripts.consolidator_daemon import run_pass
    from brain.cognitive_brain import CognitiveBrain
    args = _ap.Namespace(
        min_age_seconds=min_age_seconds,
        batch_size=500,
    )
    b = CognitiveBrain()
    r = run_pass(b, args)
    global _brain
    _brain = b
    return {"promoted": r["promoted"],
            "vision_promoted": r.get("vision_promoted", 0),
            "quarantined": r["quarantined"],
            "memories": len(b.memory)}


class RescoreReq(BaseModel):
    table: str  # 'sentence_corpus' | 'web_corpus'
    only_status: Optional[str] = None  # None = all
    dry_run: Optional[bool] = False


@app.post("/api/rescore")
def rescore(req: RescoreReq):
    """对已存在的行重新打分(回溯清洗)。返回各分数桶的统计。"""
    if req.table not in ("sentence_corpus", "web_corpus"):
        raise HTTPException(400, "table must be sentence_corpus or web_corpus")
    from brain.credibility import score as cred_score
    conn = get_pg_conn()
    where = "WHERE topic IS NOT NULL AND content IS NOT NULL"
    if req.only_status:
        where += " AND status = '%s'" % req.only_status.replace("'", "")
    buckets = {"high": 0, "mid": 0, "low": 0, "zero": 0}
    updated = 0
    quarantined = 0
    with conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, topic, content, status FROM {req.table} {where}")
            rows = cur.fetchall()
            for row_id, topic, content, status in rows:
                q = cred_score(topic, content, source=req.table)
                if q >= 0.7: buckets["high"] += 1
                elif q >= 0.4: buckets["mid"] += 1
                elif q > 0:    buckets["low"] += 1
                else:          buckets["zero"] += 1
                if not req.dry_run:
                    new_status = status
                    if q < 0.30 and status != "quarantined":
                        new_status = "quarantined"
                        quarantined += 1
                    cur.execute(
                        f"UPDATE {req.table} SET credibility=%s, status=%s WHERE id=%s",
                        (q, new_status, row_id),
                    )
                    updated += 1
    conn.close()
    return {"scanned": sum(buckets.values()), "buckets": buckets,
            "updated": updated, "newly_quarantined": quarantined,
            "dry_run": req.dry_run}


_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper(model_size="base"):
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            import whisper as _w
            _whisper_model = _w.load_model(model_size)
    return _whisper_model


@app.post("/api/voice")
async def voice(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    """语音文件 -> Whisper 转录 -> 走 gRPC 问答流水线。返回 {transcription, response}"""
    if not file.filename:
        raise HTTPException(400, "no file")
    suffix = Path(file.filename).suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with tmp:
            shutil.copyfileobj(file.file, tmp)
        path = tmp.name
        model = _get_whisper()
        # fp16=False 在 CPU 上要的;不传 language 让模型自动检测
        kwargs = {"fp16": False}
        if language:
            kwargs["language"] = language
        result = model.transcribe(path, **kwargs)
        transcription = (result.get("text") or "").strip()
        detected_lang = result.get("language")
        if not transcription:
            return {"transcription": "", "response": None,
                    "language": detected_lang, "error": "empty"}
        # 转给 gRPC
        try:
            stub = get_stub()
            grpc_req = chatbot_pb2.ReceivedMessage(
                user_id="voice", user_message=transcription,
                create_time=int(time.time()),
            )
            resp = stub.Chatbot(grpc_req, timeout=20.0)
            return {"transcription": transcription,
                    "response": resp.user_response,
                    "language": detected_lang}
        except grpc.RpcError as e:
            return {"transcription": transcription, "response": None,
                    "language": detected_lang, "error": str(e.details())}
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


@app.post("/api/vision")
async def vision(file: UploadFile = File(...)):
    """图片上传 -> pHash -> brain 多模态记忆里找最相似 -> 返回匹配 + 答案文本"""
    suffix = (Path(file.filename or "image").suffix or ".jpg").lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"):
        raise HTTPException(400, f"unsupported format: {suffix}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = tmp.name
    try:
        with tmp:
            shutil.copyfileobj(file.file, tmp)
        import imagehash
        from PIL import Image
        with Image.open(path) as im:
            query_hash = imagehash.phash(im)
        query_hex = str(query_hash)

        b = _get_brain()
        candidates = []
        for m in b.memory:
            meta = m.get("meta") or {}
            if meta.get("modality") != "image" or "phash" not in meta:
                continue
            try:
                mh = imagehash.hex_to_hash(meta["phash"])
            except Exception:
                continue
            dist = query_hash - mh
            sim = 1.0 - dist / 64.0
            candidates.append({
                "topic": m["topic"],
                "content": m["content"],
                "image_path": meta.get("image_path"),
                "phash": meta["phash"],
                "hamming_distance": int(dist),
                "similarity": round(float(sim), 3),
                "credibility": float(m["credibility"]),
            })
        candidates.sort(key=lambda x: x["hamming_distance"])
        return {
            "phash": query_hex,
            "candidates": candidates[:5],
            "matched": bool(candidates and candidates[0]["hamming_distance"] <= 16),
        }
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


class FeedbackReq(BaseModel):
    topic: str
    delta: float  # +0.05 (👍) / -0.05 (👎)


@app.post("/api/feedback")
def feedback(req: FeedbackReq):
    """显式 RL 反馈:微调指定 topic 的可信度"""
    b = _get_brain()
    new_cred = b.feedback(req.topic, req.delta)
    if new_cred is None:
        raise HTTPException(404, f"topic '{req.topic}' not in brain memory")
    # 持久化(下次 brain 加载会保留)
    try:
        b.save()
    except Exception as e:
        return {"topic": req.topic, "credibility": new_cred, "saved": False, "err": str(e)}
    return {"topic": req.topic, "credibility": new_cred, "saved": True}


@app.get("/api/deep_think")
def deep_think(q: str = Query(..., min_length=1), depth: int = 3):
    """链式联想思考:返回思考路径 + 最终答复"""
    b = _get_brain()
    r = b.deep_think(q, depth=depth)
    ans, dbg = r["final"]
    return {
        "query": q,
        "chain": r["chain"],
        "answer": ans,
        "answer_topic": dbg.get("topic"),
        "hemisphere": dbg.get("hemisphere"),
    }


@app.get("/api/hemisphere_route")
def hemisphere_route(q: str):
    """仅返回该 query 的半球路由决定(轻量,只用于 UI 标识)"""
    b = _get_brain()
    return {"query": q, "hemisphere": b.hemisphere_route(q)}


class GrowBrainReq(BaseModel):
    delta: int = 256
    reason: Optional[str] = "manual_ui"


@app.post("/api/grow_brain")
def grow_brain(req: GrowBrainReq):
    """新增联想层神经元 + 重编码所有记忆 + 持久化"""
    if req.delta <= 0 or req.delta > 4096:
        raise HTTPException(400, "delta must be in [1, 4096]")
    b = _get_brain()
    ev = b.grow_neurons(delta=req.delta, reason=req.reason or "manual")
    b.save()
    # 写 training_log 让概览页能看到
    try:
        import psycopg2
        conn = get_pg_conn()
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO training_log (task, note) VALUES (%s, %s)",
                ("brain_growth",
                 f"+{ev['delta']} neurons, assoc_dim {ev['old_assoc_dim']}->{ev['new_assoc_dim']}, reason={ev['reason']}"),
            )
        conn.close()
    except Exception:
        pass
    return ev


@app.get("/api/growth_log")
def growth_log():
    """大脑成长历史"""
    b = _get_brain()
    return {"events": b.growth_log[-50:]}


@app.get("/api/features/top")
def features_top(n: int = 30, klass: Optional[str] = None):
    """返回出现频次最高的特征。
    klass: 'entities'|'nouns'|'verbs'|'adjs'|'numbers',不传则混合
    """
    b = _get_brain()
    # 拿足够多的候选,过滤后取 top-n
    items = b.feature_stats(top_n=n * 6)
    if klass:
        items = [(k, c) for k, c in items if k.startswith(klass + ":")]
    items = items[:n]
    fi = getattr(b, "feature_index", None) or {}
    # 按 klass 统计
    by_klass = {}
    for k in fi.keys():
        cls = k.split(":", 1)[0] if ":" in k else "unknown"
        by_klass[cls] = by_klass.get(cls, 0) + 1
    return {
        "total_unique_features": len(fi),
        "by_klass": by_klass,
        "items": [
            {"feature": k, "klass": k.split(":", 1)[0],
             "value": k.split(":", 1)[1], "count": c}
            for k, c in items
        ],
    }


@app.get("/api/features/search")
def features_search(feature: str, limit: int = 20):
    """通过特征反查 memory。例: feature=entities:光速"""
    b = _get_brain()
    mems = b.search_by_feature(feature, top_k=limit)
    return {
        "feature": feature, "count": len(mems),
        "items": [
            {"topic": m["topic"], "content": (m["content"] or "")[:200],
             "source": m["source"], "hemisphere": m.get("hemisphere"),
             "credibility": m["credibility"]}
            for m in mems
        ],
    }


@app.get("/api/image/{filename}")
def serve_image(filename: str):
    """供视觉语料网格视图加载图片缩略图。"""
    from fastapi.responses import FileResponse
    safe = os.path.basename(filename)  # 防止路径穿越
    p = ROOT / "corpus" / "images" / safe
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(str(p))


@app.get("/api/brain_test")
def brain_test(q: str = Query(..., min_length=1)):
    """直接查 brain 检索结果(不走 fallback,纯检索 top-K 调试)
    + 半球路由结果"""
    b = _get_brain()
    hem = b.hemisphere_route(q)
    results = b.retrieve(q, top_k=5, hemisphere=hem if hem != "both" else None)
    return {
        "query": q,
        "hemisphere": hem,
        "results": [
            {"topic": m["topic"], "content": m["content"][:200],
             "source": m["source"], "similarity": float(sim),
             "credibility": float(cred)}
            for m, sim, cred in results
        ],
    }
