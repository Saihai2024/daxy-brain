"""
把 vision_corpus pending 行计算 pHash 后,作为多模态记忆灌入认知大脑。

每条记忆的 meta 字段保存:
  {"image_path": ..., "phash": "0xhexhash", "modality": "image"}

retrieval 时:
  - 文本 query 走原 brain.retrieve(基于 topic 字符)
  - 图片 query 走新加的 brain.retrieve_by_image(基于 hamming(phash))
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imagehash
from PIL import Image
from utils.db_config import get_pg_conn
from brain.cognitive_brain import CognitiveBrain


def compute_phash(image_path):
    full = ROOT / image_path
    if not full.is_file():
        return None
    try:
        with Image.open(full) as im:
            return str(imagehash.phash(im))  # 16 hex chars
    except Exception as e:
        print(f"  phash failed for {image_path}: {e}")
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default="pending",
                    help="处理哪些状态的行,默认 pending;灾后重灌时用 learned")
    ap.add_argument("--update_status", action="store_true",
                    help="处理完把 pending 行标 learned(--status=learned 模式下不动状态)")
    args = ap.parse_args()

    # consolidator 看到 reload 锁会跳过本轮 brain 写
    lock = ROOT / "logs" / "BRAIN_RELOAD_IN_PROGRESS"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(f"pid={os.getpid()} script=index_vision_to_brain\n", encoding="utf-8")

    try:
        _run_main(args)
    finally:
        try: lock.unlink()
        except OSError: pass


def _run_main(args):
    conn = get_pg_conn()
    brain = CognitiveBrain()
    print(f"brain before: {brain.stats()}")

    promoted = 0
    skipped = 0
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, image_path, question, answer, surprise_score "
                "FROM vision_corpus WHERE status=%s",
                (args.status,),
            )
            for row_id, image_path, question, answer, score in cur.fetchall():
                phash = compute_phash(image_path)
                if not phash:
                    if args.update_status and args.status == "pending":
                        cur.execute(
                            "UPDATE vision_corpus SET status='quarantined' WHERE id=%s",
                            (row_id,),
                        )
                    skipped += 1
                    continue
                # 主题用 question 中的实体名(由 cleanup 脚本写入,如"火星是什么"取首段)
                topic = question.replace("是什么", "").strip()
                idx = brain.add_memory(
                    topic, answer or "",
                    source="vision_corpus",
                    credibility=float(score) if score else 0.7,
                    meta={"image_path": image_path,
                          "phash": phash,
                          "modality": "image"},
                    dedup_threshold=None,  # 批量重灌跳过 O(n²) 去重
                )
                if args.update_status and args.status == "pending":
                    cur.execute(
                        "UPDATE vision_corpus SET status='learned' WHERE id=%s",
                        (row_id,),
                    )
                if args.update_status:
                    cur.execute(
                        "INSERT INTO training_log (task, record_id, note) VALUES (%s, %s, %s)",
                        ("vision_index", row_id,
                         f"indexed phash={phash} brain_idx={idx}"),
                    )
                if promoted < 10 or promoted % 500 == 0:
                    print(f"  + idx={idx} topic={topic} phash={phash}  (n={promoted})")
                promoted += 1
                # 每 2000 条 save 一次,防止崩了全丢
                if promoted % 2000 == 0:
                    brain.save()
                    print(f"  [checkpoint] saved at n={promoted}")
    brain.save()
    print(f"\n[done] indexed {promoted}, skipped {skipped}")
    print(f"brain after: {brain.stats()}")
    print(f"brain after: {brain.stats()}")
    conn.close()


if __name__ == "__main__":
    main()
