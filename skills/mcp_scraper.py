import sqlite3
import urllib.request
import urllib.parse
import re
import os
from datetime import datetime

DB_PATH = "corpus/knowledge/corpus.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS web_corpus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT UNIQUE,
            content TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def scrape_concept(topic):
    try:
        print(f"[*] 正在从公共网络探测概念: {topic}...")
        url = 'https://baike.baidu.com/item/' + urllib.parse.quote(topic)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        response = urllib.request.urlopen(req, timeout=10)
        html = response.read().decode('utf-8')
        
        match = re.search(r'<meta name="description" content="(.*?)">', html)
        if match:
            desc = match.group(1)
            # 清洗特殊符号和多余文字
            desc = re.sub(r'\[.*?\]', '', desc)
            desc = desc.replace('秒懂百科', '').replace('百度百科', '').strip()
            if len(desc) > 10:
                print(f"  [成功] 获取到知识: {desc[:40]}...")
                return desc
    except Exception as e:
        print(f"  [失败] 网络拦截或超时: {e}")
    return None

def mcp_run(topics):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    success_count = 0
    for topic in topics:
        cursor.execute('SELECT id FROM web_corpus WHERE topic=?', (topic,))
        if cursor.fetchone():
            print(f"[-] 概念 '{topic}' 已存在于数据库中，跳过。")
            continue
            
        content = scrape_concept(topic)
        if content:
            try:
                cursor.execute('INSERT INTO web_corpus (topic, content, status) VALUES (?, ?, ?)', (topic, content, 'pending'))
                conn.commit()
                success_count += 1
            except sqlite3.IntegrityError:
                pass
                
    conn.close()
    print(f"\n[OK] MCP 探测完毕。共新增 {success_count} 条知识待训练。")

if __name__ == "__main__":
    target_topics = [
        "量子纠缠", "降维打击", "光速", "引力波", "赛博朋克",
        "戴森球", "暗物质", "费米悖论", "平行宇宙", "薛定谔的猫"
    ]
    mcp_run(target_topics)
