"""数据库连接(MySQL legacy + PostgreSQL primary,可用 env vars 或项目根 .env 覆盖)"""
import os
from pathlib import Path


def load_dotenv_if_present():
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.is_file():
        return
    try:
        for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                v = v[1:-1]
            if k:
                os.environ.setdefault(k, v)
    except Exception:
        pass


load_dotenv_if_present()

DB_CONFIG = {
    'host': os.environ.get('MYSQL_HOST', 'localhost'),
    'port': int(os.environ.get('MYSQL_PORT', '3306')),
    'user': os.environ.get('MYSQL_USER', 'root'),
    'password': os.environ.get('MYSQL_PASSWORD', ''),
    'database': os.environ.get('MYSQL_DB', 'daxy_cognitive_db'),
    'charset': 'utf8mb4',
}

PG_DSN = os.environ.get(
    'PG_DSN',
    'host=127.0.0.1 port=5432 user=postgres dbname=daxy_cognitive_db',
)


def get_mysql_conn(with_db=True):
    import pymysql
    config = DB_CONFIG.copy()
    if not with_db:
        config.pop('database', None)
    return pymysql.connect(**config)


def get_pg_conn():
    import psycopg2
    return psycopg2.connect(PG_DSN)


def init_mysql_db():
    conn = get_mysql_conn(with_db=False)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.commit()
    conn.close()

    conn = get_mysql_conn()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS web_corpus (
            id INT AUTO_INCREMENT PRIMARY KEY,
            topic VARCHAR(255) UNIQUE,
            content TEXT,
            status VARCHAR(50) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
