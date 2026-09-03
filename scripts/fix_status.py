import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.db_config import get_mysql_conn

conn = get_mysql_conn()
cursor = conn.cursor()
cursor.execute("SELECT topic, content FROM web_corpus WHERE status='learned'")
rows = cursor.fetchall()
conn.close()

data = {}
CONCEPT_PATH = "corpus/knowledge/concept_dict.json"
if os.path.exists(CONCEPT_PATH):
    try:
        with open(CONCEPT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        pass

for topic, content in rows:
    summary = content[:100] + "..." if len(content) > 100 else content
    data[topic] = summary

with open(CONCEPT_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Fixed {len(rows)} concepts into concept_dict.json")
