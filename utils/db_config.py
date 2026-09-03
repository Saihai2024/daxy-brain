"""数据库连接(MySQL legacy + PostgreSQL primary,可用 env vars 覆盖)"""
import os
import pymysql

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
