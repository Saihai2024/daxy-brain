"""
把 daxy_cognitive_db 从 MySQL 迁移到 PostgreSQL。

幂等:重跑会 DROP & 重建表然后整体灌一次。
"""
import os

import pymysql
import psycopg2
from psycopg2.extras import execute_batch

MYSQL_KW = dict(
    host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
    port=int(os.environ.get("MYSQL_PORT", "3306")),
    user=os.environ.get("MYSQL_USER", "root"),
    password=os.environ.get("MYSQL_PASSWORD", ""),
    database=os.environ.get("MYSQL_DB", "daxy_cognitive_db"),
    charset="utf8mb4",
)
PG_DSN = os.environ.get(
    "PG_DSN", "host=127.0.0.1 port=5432 user=postgres dbname=daxy_cognitive_db"
)

SCHEMA_SQL = """
DROP TABLE IF EXISTS sentence_corpus CASCADE;
DROP TABLE IF EXISTS vision_corpus CASCADE;
DROP TABLE IF EXISTS web_corpus CASCADE;
DROP TABLE IF EXISTS training_log CASCADE;

CREATE TABLE sentence_corpus (
    id          SERIAL PRIMARY KEY,
    topic       VARCHAR(255) UNIQUE,
    content     TEXT,
    status      VARCHAR(50) DEFAULT 'pending',
    credibility REAL DEFAULT 1.0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE vision_corpus (
    id              SERIAL PRIMARY KEY,
    image_path      VARCHAR(255) NOT NULL,
    question        TEXT,
    answer          TEXT,
    surprise_score  REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status          VARCHAR(20) DEFAULT 'pending'
);

CREATE TABLE web_corpus (
    id          SERIAL PRIMARY KEY,
    topic       VARCHAR(255) UNIQUE,
    content     TEXT,
    status      VARCHAR(50) DEFAULT 'pending',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    credibility REAL DEFAULT 1.0
);

CREATE TABLE training_log (
    id          SERIAL PRIMARY KEY,
    ts          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    task        VARCHAR(32) NOT NULL DEFAULT 'multimodal',
    record_id   INT,
    image_path  VARCHAR(255),
    epoch       INT,
    loss        DOUBLE PRECISION,
    note        VARCHAR(255)
);

CREATE INDEX ON training_log (ts);
CREATE INDEX ON training_log (task);
CREATE INDEX ON sentence_corpus (status);
CREATE INDEX ON web_corpus (status);
CREATE INDEX ON vision_corpus (status);
"""

def migrate_table(my_cur, pg_cur, table, columns, pg_table=None):
    pg_table = pg_table or table
    cols = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    my_cur.execute(f"SELECT {cols} FROM `{table}`")
    rows = my_cur.fetchall()
    if not rows:
        print(f"  {table}: 0 rows, skip")
        return 0
    insert_sql = f"INSERT INTO {pg_table} ({cols}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    execute_batch(pg_cur, insert_sql, rows, page_size=200)
    return len(rows)


def main():
    my = pymysql.connect(**MYSQL_KW)
    pg = psycopg2.connect(PG_DSN)
    pg.autocommit = False

    print("creating PG schema ...")
    with pg.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    pg.commit()

    with my.cursor() as my_cur, pg.cursor() as pg_cur:
        n = migrate_table(my_cur, pg_cur, "sentence_corpus",
                          ["id","topic","content","status","credibility","created_at"])
        print(f"  sentence_corpus: {n} rows")
        n = migrate_table(my_cur, pg_cur, "vision_corpus",
                          ["id","image_path","question","answer","surprise_score","created_at","status"])
        print(f"  vision_corpus: {n} rows")
        n = migrate_table(my_cur, pg_cur, "web_corpus",
                          ["id","topic","content","status","created_at","credibility"])
        print(f"  web_corpus: {n} rows")
        n = migrate_table(my_cur, pg_cur, "training_log",
                          ["id","ts","task","record_id","image_path","epoch","loss","note"])
        print(f"  training_log: {n} rows")

        # Reset sequences to max(id)+1 to avoid PK conflicts on future inserts
        for tbl in ["sentence_corpus", "vision_corpus", "web_corpus", "training_log"]:
            pg_cur.execute(
                f"SELECT setval(pg_get_serial_sequence('{tbl}','id'), "
                f"COALESCE((SELECT MAX(id) FROM {tbl}), 0) + 1, false)"
            )
    pg.commit()

    # verify
    print("\nverification:")
    with pg.cursor() as cur:
        for tbl in ["sentence_corpus","vision_corpus","web_corpus","training_log"]:
            cur.execute(f"SELECT COUNT(*) FROM {tbl}")
            print(f"  pg.{tbl}: {cur.fetchone()[0]} rows")

    my.close(); pg.close()
    print("done.")


if __name__ == "__main__":
    main()
