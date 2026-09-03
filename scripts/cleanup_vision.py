"""
清理 vision_corpus:
  1. 把英文别名(sample_mars.jpg)映射到中文文件(sample_火星.jpg)
  2. 删除引用了不存在图片的行(标记为 quarantined)
  3. 重建 question 字段(用图片对应实体,如"sample_火星.jpg" -> question="火星图片是什么")
  4. 同一 image_path 多条记录:保留 answer 最长的,删重复
  5. 重新评分 + 把存活的 status 改 'pending' 让 daemon 处理
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.db_config import get_pg_conn, get_mysql_conn
from brain.credibility import score as credibility_score, _topic_relevance


# 英文文件名 -> 中文等价名(根据现有磁盘文件)
ALIAS_MAP = {
    "sample_mars.jpg": "sample_火星.jpg",
}

# 哪些图片文件实际存在(扫盘获取)
IMAGES_DIR = ROOT / "corpus" / "images"
EXISTING = {p.name for p in IMAGES_DIR.glob("*.jpg")} if IMAGES_DIR.is_dir() else set()


def normalize_path(image_path):
    """统一路径分隔符 + 应用别名映射 -> (规范路径, 文件名)"""
    p = (image_path or "").replace("\\", "/")
    name = os.path.basename(p)
    name = ALIAS_MAP.get(name, name)
    return f"corpus/images/{name}", name


def topic_from_filename(name):
    """sample_火星.jpg -> 火星"""
    base = os.path.splitext(name)[0]
    return base.replace("sample_", "").strip()


def main():
    summary = {"restored_from_mysql": 0, "quarantined_missing": 0,
               "deduped": 0, "kept_with_relevance": 0}

    # 0. 从 MySQL 恢复原始 vision_corpus 全量(防止之前破坏的 cleanup 留下错配)
    my = get_mysql_conn()
    pg = get_pg_conn()
    try:
        with my.cursor() as mc:
            mc.execute("SELECT image_path, question, answer, surprise_score FROM vision_corpus")
            mysql_rows = mc.fetchall()
        with pg, pg.cursor() as cur:
            cur.execute("DELETE FROM vision_corpus")
            for img, q, a, sc in mysql_rows:
                cur.execute(
                    "INSERT INTO vision_corpus (image_path, question, answer, surprise_score, status) "
                    "VALUES (%s, %s, %s, %s, 'pending')",
                    (img, q, a, sc),
                )
        summary["restored_from_mysql"] = len(mysql_rows)
    finally:
        my.close()

    with pg:
        with pg.cursor() as cur:
            cur.execute("SELECT id, image_path, question, answer FROM vision_corpus")
            rows = cur.fetchall()

            # 路径规范 + 缺失检查 + 主题相关性挑选
            best_per_path = {}  # path -> (id, relevance, len)
            to_quarantine = []
            for row_id, image_path, q, a in rows:
                norm, name = normalize_path(image_path)
                if name not in EXISTING:
                    to_quarantine.append(row_id)
                    continue
                topic = topic_from_filename(name)
                rel = _topic_relevance(topic, a or "")
                a_len = len(a or "")
                # 优先 relevance 高,其次 answer 长(确保答案谈的是该主题)
                key = (rel, a_len)
                if norm not in best_per_path or key > best_per_path[norm][1]:
                    best_per_path[norm] = (row_id, key, a_len)
            keepers = {info[0] for info in best_per_path.values()}

            if to_quarantine:
                cur.execute(
                    "UPDATE vision_corpus SET status='quarantined' WHERE id = ANY(%s)",
                    (to_quarantine,),
                )
                summary["quarantined_missing"] = len(to_quarantine)

            cur.execute(
                "DELETE FROM vision_corpus WHERE id != ALL(%s) AND id != ALL(%s)",
                (list(keepers), to_quarantine),
            )
            summary["deduped"] = cur.rowcount

            cur.execute(
                "SELECT id, image_path, answer FROM vision_corpus WHERE id = ANY(%s)",
                (list(keepers),),
            )
            for row_id, image_path, answer in cur.fetchall():
                norm, name = normalize_path(image_path)
                topic = topic_from_filename(name)
                new_q = f"{topic}是什么"
                quality = credibility_score(topic, answer or "", source="vision_corpus")
                cur.execute(
                    "UPDATE vision_corpus SET image_path=%s, question=%s, "
                    "       status='pending', surprise_score=%s "
                    "WHERE id = %s",
                    (norm, new_q, quality, row_id),
                )
                summary["kept_with_relevance"] += 1

    with pg, pg.cursor() as cur:
        cur.execute("SELECT status, COUNT(*) FROM vision_corpus GROUP BY status")
        summary["final_status"] = dict(cur.fetchall())
        # 顺便预览每个 keeper 的 topic + 答案前 60 字
        cur.execute(
            "SELECT image_path, question, LEFT(answer, 60) FROM vision_corpus WHERE status='pending'"
        )
        summary["preview"] = [{"img": r[0], "q": r[1], "a_head": r[2]} for r in cur.fetchall()]
    print("[cleanup]", summary)
    pg.close()


if __name__ == "__main__":
    main()
