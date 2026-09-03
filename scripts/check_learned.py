import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.db_config import get_mysql_conn

conn = get_mysql_conn()
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM web_corpus WHERE status="learned"')
w_learned = c.fetchone()[0]
c.execute('SELECT COUNT(*) FROM sentence_corpus WHERE status="learned"')
s_learned = c.fetchone()[0]
print(f'Web learned: {w_learned}, Sentences learned: {s_learned}')
conn.close()
