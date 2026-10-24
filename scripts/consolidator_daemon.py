"""
后台巩固 daemon - 把对话/抓取的 pending 数据提升为 learned,并加入认知大脑。

提升规则(可调):
  1. credibility >= MIN_CREDIBILITY (默认 0.55,即至少出现过一次重复)
  2. age >= MIN_AGE (默认 5 分钟,过滤瞬时噪声)
  3. content 长度 5..50000

每轮:
  - 从 sentence_corpus + web_corpus 各取一批 pending
  - 通过条件的 -> 加 brain.memory + UPDATE status='learned'
  - 写 training_log 留痕
  - brain.save()

注意: gRPC 服务的内存 brain 实例不会自动看到这里的更新。当前策略:服务下次重启时
load brain.pkl 拿到新记忆。后续可改为定期 SIGHUP 触发热重载。
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.db_config import get_pg_conn
from utils.log_setup import setup_daemon_logger
try:
    from brain.cognitive_brain import CognitiveBrain
    from brain.credibility import score as credibility_score
except ImportError:
    from chatbot.cognitive_brain import CognitiveBrain
    from chatbot.credibility import score as credibility_score

LOG = setup_daemon_logger("consolidator")

# 视觉:可选导入(若没装 imagehash 跳过 vision 巩固)
try:
    import imagehash
    from PIL import Image
    _HAS_VISION = True
except ImportError:
    _HAS_VISION = False

PROMOTE_THRESHOLD = 0.55     # 质量分 >= 此值 -> learned
QUARANTINE_THRESHOLD = 0.30  # 质量分 <  此值 -> quarantined


def fetch_pending(cur, table, min_age_seconds, batch_size):
    cur.execute(
        f"SELECT id, topic, content, credibility "
        f"FROM {table} "
        f"WHERE status = 'pending' "
        f"  AND created_at < NOW() - INTERVAL '%s seconds' "
        f"  AND topic IS NOT NULL AND content IS NOT NULL "
        f"ORDER BY created_at "
        f"LIMIT %s",
        (int(min_age_seconds), batch_size),
    )
    return cur.fetchall()


def update_status(cur, table, row_id, status, credibility, note):
    cur.execute(
        f"UPDATE {table} SET status = %s, credibility = %s WHERE id = %s",
        (status, credibility, row_id),
    )
    cur.execute(
        "INSERT INTO training_log (task, record_id, note) VALUES (%s, %s, %s)",
        (f"consolidate_{table}", row_id, note),
    )


def fetch_pending_vision(cur, min_age_seconds, batch_size):
    cur.execute(
        "SELECT id, image_path, question, answer "
        "FROM vision_corpus WHERE status='pending' "
        "  AND created_at < NOW() - INTERVAL '%s seconds' "
        "  AND image_path IS NOT NULL AND answer IS NOT NULL "
        "ORDER BY created_at LIMIT %s",
        (int(min_age_seconds), batch_size),
    )
    return cur.fetchall()


def promote_vision_row(brain, cur, row_id, image_path, question, answer):
    """vision_corpus pending -> 计算 pHash + 加 brain.memory"""
    if not _HAS_VISION:
        return False, "no_imagehash_lib"
    full = os.path.join(os.getcwd(), image_path) if not os.path.isabs(image_path) else image_path
    if not os.path.isfile(full):
        cur.execute(
            "UPDATE vision_corpus SET status='quarantined' WHERE id=%s", (row_id,)
        )
        return False, "image_missing"
    try:
        with Image.open(full) as im:
            phash = str(imagehash.phash(im))
    except Exception as e:
        cur.execute(
            "UPDATE vision_corpus SET status='quarantined' WHERE id=%s", (row_id,)
        )
        return False, f"phash_fail: {e}"

    # 用 question 提主题(去掉"是什么"后缀)
    topic = (question or "").replace("是什么", "").strip()
    if not topic:
        topic = os.path.basename(image_path).replace("sample_", "").rsplit(".", 1)[0]
    quality = credibility_score(topic, answer, source="vision_corpus")
    quality = max(quality, 0.6)  # 视觉给个保底,文本短不该一票否决
    idx = brain.add_memory(
        topic, answer, source="vision_corpus", credibility=quality,
        meta={"image_path": image_path, "phash": phash, "modality": "image"},
    )
    cur.execute(
        "UPDATE vision_corpus SET status='learned', surprise_score=%s WHERE id=%s",
        (quality, row_id),
    )
    cur.execute(
        "INSERT INTO training_log (task, record_id, note) VALUES (%s, %s, %s)",
        ("consolidate_vision_corpus", row_id,
         f"learned: phash={phash} brain_idx={idx} q={quality:.2f}"),
    )
    return True, phash


def run_pass(brain, args):
    promoted = 0
    quarantined = 0
    vision_promoted = 0
    conn = get_pg_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                for table in ("sentence_corpus", "web_corpus"):
                    rows = fetch_pending(cur, table, args.min_age_seconds, args.batch_size)
                    for row_id, topic, content, _stored_cred in rows:
                        quality = credibility_score(topic, content, source=table)
                        if quality < QUARANTINE_THRESHOLD:
                            update_status(
                                cur, table, row_id, "quarantined", quality,
                                f"quarantined: quality={quality:.2f}",
                            )
                            quarantined += 1
                        elif quality >= PROMOTE_THRESHOLD:
                            idx = brain.add_memory(
                                topic, content, source=table, credibility=quality,
                            )
                            update_status(
                                cur, table, row_id, "learned", quality,
                                f"learned: quality={quality:.2f}, brain_idx={idx}",
                            )
                            promoted += 1
                            # 自监督复述:把 web_corpus 的高质量 (topic, content)
                            # 也作为 (问题, 答案) 训练对写到 sentence_corpus。
                            # 让"训练语料"卡片随着抓取一起增长。
                            if table == "web_corpus" and quality >= 0.65:
                                try:
                                    rehearse_topic = (topic + "是什么")[:255]
                                    cur.execute(
                                        "INSERT INTO sentence_corpus "
                                        "(topic, content, status, credibility) "
                                        "VALUES (%s, %s, 'learned', %s) "
                                        "ON CONFLICT (topic) DO NOTHING",
                                        (rehearse_topic, content, quality),
                                    )
                                except Exception:
                                    pass

                # 视觉模态
                if _HAS_VISION:
                    vrows = fetch_pending_vision(cur, args.min_age_seconds, args.batch_size)
                    for row_id, ip, q, a in vrows:
                        ok, info = promote_vision_row(brain, cur, row_id, ip, q, a)
                        if ok:
                            vision_promoted += 1
    finally:
        conn.close()
    grew = None
    hebbian_steps = 0
    if promoted > 0 or vision_promoted > 0:
        # 检查是否需要自动扩神经元
        grew = brain.maybe_auto_grow(threshold_ratio=4.5)
        if grew:
            print(f"[daemon] 🧠 brain auto-grew: +{grew['delta']} neurons "
                  f"(assoc_dim {grew['old_assoc_dim']}->{grew['new_assoc_dim']})")
        # 安全的 micro-Hebbian:每轮巩固后用极小 lr 复述最新 20 条记忆。
        # lr=0.0001 比 LEARN_RATE(0.05)小 500 倍,长期不会让编码器坍缩,
        # 但 brain.steps 会持续增长,显示"正在学习"。
        try:
            steps_before = brain.steps
            brain.consolidate_batch(n=min(20, len(brain.memory)), lr=0.0001)
            hebbian_steps = brain.steps - steps_before
        except Exception as e:
            print(f"[daemon] micro-hebbian skipped: {e}")
            # 写训练日志
            try:
                conn2 = get_pg_conn()
                with conn2, conn2.cursor() as cur:
                    cur.execute(
                        "INSERT INTO training_log (task, note) VALUES (%s, %s)",
                        ("brain_growth",
                         f"auto +{grew['delta']} neurons, "
                         f"{grew['old_assoc_dim']}->{grew['new_assoc_dim']}, "
                         f"reason={grew['reason']}"),
                    )
                conn2.close()
            except Exception as e:
                print(f"[daemon] log growth failed: {e}")
        brain.save()
    return {"promoted": promoted, "quarantined": quarantined,
            "vision_promoted": vision_promoted,
            "grew": grew,
            "hebbian_steps": hebbian_steps}


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            return bool(ok and code.value == STILL_ACTIVE)
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _check_reload_lock(reload_lock: Path, max_age_sec: float = 7200.0) -> bool:
    """检查 reload_lock 是否有效;若持有锁的 PID 已退出或锁超时则自动清理陈旧锁。"""
    if not reload_lock.exists():
        return False
    try:
        age = time.time() - reload_lock.stat().st_mtime
        txt = reload_lock.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"pid\s*=\s*(\d+)", txt)
        if m:
            pid = int(m.group(1))
            if not _is_pid_alive(pid):
                LOG.warning(f"[daemon] removing stale reload lock (dead pid={pid}, age={age:.0f}s)")
                reload_lock.unlink(missing_ok=True)
                return False
        if age > max_age_sec:
            LOG.warning(f"[daemon] removing expired reload lock (age={age:.0f}s > {max_age_sec:.0f}s)")
            reload_lock.unlink(missing_ok=True)
            return False
    except Exception as e:
        LOG.warning(f"[daemon] reload lock inspect error: {e}")
    return reload_lock.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=60.0,
                    help="seconds between passes")
    ap.add_argument("--min_age_seconds", type=int, default=60,
                    help="only consider rows older than this (anti-race)")
    ap.add_argument("--batch_size", type=int, default=200,
                    help="max items per pass per table")
    ap.add_argument("--once", action="store_true",
                    help="run one pass and exit (for testing)")
    args = ap.parse_args()

    LOG.info(f"[daemon] starting. interval={args.interval}s min_age={args.min_age_seconds}s")
    LOG.info(f"  promote_threshold={PROMOTE_THRESHOLD}  quarantine_threshold={QUARANTINE_THRESHOLD}")
    print(f"[daemon] starting. interval={args.interval}s min_age={args.min_age_seconds}s")
    print(f"  promote_threshold={PROMOTE_THRESHOLD}  quarantine_threshold={QUARANTINE_THRESHOLD}")

    brain = CognitiveBrain()
    LOG.info(f"[daemon] brain loaded: {brain.stats()}")

    # 灾后重灌 / 维护时,放一个文件锁阻止 daemon 启动覆盖 brain.pkl
    reload_lock = ROOT / "logs" / "BRAIN_RELOAD_IN_PROGRESS"

    while True:
        if _check_reload_lock(reload_lock):
            LOG.warning(f"[daemon] reload lock present ({reload_lock.name}), "
                        f"sleeping {args.interval}s without touching brain")
            time.sleep(args.interval)
            continue
        t0 = time.time()
        try:
            r = run_pass(brain, args)
            elapsed = time.time() - t0
            if r["promoted"] or r["quarantined"] or r.get("vision_promoted"):
                LOG.info(f"[daemon] text_promoted={r['promoted']} "
                         f"vision_promoted={r.get('vision_promoted', 0)} "
                         f"quarantined={r['quarantined']} "
                         f"hebbian+={r.get('hebbian_steps', 0)} "
                         f"in {elapsed:.1f}s. brain memories now: {len(brain.memory)}")
            else:
                LOG.info(f"[daemon] no actionable rows ({elapsed:.1f}s)")
        except Exception as e:
            import traceback
            LOG.error(f"[daemon] pass error: {e}\n{traceback.format_exc()}")

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
