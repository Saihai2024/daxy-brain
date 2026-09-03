import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.db_config import get_mysql_conn

conn = get_mysql_conn()
cursor = conn.cursor()
try:
    cursor.execute("ALTER TABLE web_corpus ADD COLUMN credibility FLOAT DEFAULT 1.0")
    conn.commit()
    print("Successfully added credibility column to web_corpus.")
except Exception as e:
    print(f"Error or column already exists: {e}")
conn.close()
