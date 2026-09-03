import sqlite3
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.db_config import init_mysql_db, get_mysql_conn

def migrate():
    print("Initializing MySQL Database...")
    init_mysql_db()
    
    sqlite_path = "corpus/knowledge/corpus.db"
    if not os.path.exists(sqlite_path):
        print("No SQLite database found. Skip migration.")
        return
        
    print("Reading from SQLite...")
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_cursor = sqlite_conn.cursor()
    sqlite_cursor.execute("SELECT topic, content, status FROM web_corpus")
    rows = sqlite_cursor.fetchall()
    sqlite_conn.close()
    
    if not rows:
        print("SQLite database is empty. Skip migration.")
        return
        
    print("Writing to MySQL...")
    mysql_conn = get_mysql_conn()
    mysql_cursor = mysql_conn.cursor()
    
    count = 0
    for row in rows:
        topic, content, status = row
        try:
            mysql_cursor.execute(
                "INSERT IGNORE INTO web_corpus (topic, content, status) VALUES (%s, %s, %s)",
                (topic, content, status)
            )
            count += mysql_cursor.rowcount
        except Exception as e:
            print(f"Error migrating {topic}: {e}")
            
    mysql_conn.commit()
    mysql_conn.close()
    print(f"Migration completed. {count} new records moved to MySQL.")

if __name__ == "__main__":
    migrate()
