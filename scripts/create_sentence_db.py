import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.db_config import get_mysql_conn

conn = get_mysql_conn()
cursor = conn.cursor()
try:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentence_corpus (
            id INT AUTO_INCREMENT PRIMARY KEY,
            topic VARCHAR(255) UNIQUE,
            content TEXT,
            status VARCHAR(50) DEFAULT 'pending',
            credibility FLOAT DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    print("Successfully created sentence_corpus table.")
except Exception as e:
    print(f"Error: {e}")
conn.close()
