import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.db_config import get_mysql_conn

conn = get_mysql_conn()
cursor = conn.cursor()
cursor.execute("UPDATE web_corpus SET status='pending'")
conn.commit()
print(f"Reset {cursor.rowcount} records to pending.")
conn.close()
