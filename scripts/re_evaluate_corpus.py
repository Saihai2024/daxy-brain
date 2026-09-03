import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.db_config import get_mysql_conn
from skills.immune_system import CognitiveImmuneSystem

conn = get_mysql_conn()
cursor = conn.cursor()
cursor.execute('SELECT id, topic, content FROM web_corpus')
rows = cursor.fetchall()

immune_system = CognitiveImmuneSystem()
updated_count = 0
quarantined_count = 0

for row in rows:
    cid, topic, content = row
    if not content:
        continue
    cred = immune_system.evaluate_credibility(content)
    status = 'pending'
    if cred < 0.6:
        status = 'quarantined'
        quarantined_count += 1
        
    cursor.execute('UPDATE web_corpus SET credibility=%s, status=%s WHERE id=%s', (cred, status, cid))
    updated_count += 1

conn.commit()
conn.close()

print(f"Re-evaluated {updated_count} records.")
print(f"Quarantined {quarantined_count} low-credibility records.")

# 顺便导出一下置信度排名表供查看
import csv
conn = get_mysql_conn()
cursor = conn.cursor()
cursor.execute('SELECT topic, credibility, status FROM web_corpus ORDER BY credibility ASC LIMIT 50')
rank_rows = cursor.fetchall()
conn.close()

with open('corpus/knowledge/credibility_rank.csv', 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Topic', 'Credibility Score', 'Status'])
    for r in rank_rows:
        writer.writerow(r)

print("Generated credibility ranking table at corpus/knowledge/credibility_rank.csv")
