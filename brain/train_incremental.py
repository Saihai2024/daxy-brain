import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import json
import pickle
import time
import config
from brain.cognitive_engine import CognitiveEngine

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.db_config import get_mysql_conn

CONCEPT_PATH = "corpus/knowledge/concept_dict.json"

def get_pending_corpus(limit=3):
    conn = get_mysql_conn()
    cursor = conn.cursor()
    cursor.execute('SELECT id, topic, content FROM web_corpus WHERE status="pending" LIMIT %s', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_pending_sentences(limit=3):
    conn = get_mysql_conn()
    cursor = conn.cursor()
    cursor.execute('SELECT id, topic, content FROM sentence_corpus WHERE status="pending" LIMIT %s', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def mark_as_learned(ids, table="web_corpus"):
    if not ids: return
    conn = get_mysql_conn()
    cursor = conn.cursor()
    format_strings = ','.join(['%s'] * len(ids))
    cursor.execute(f'UPDATE {table} SET status="learned" WHERE id IN ({format_strings})', tuple(ids))
    conn.commit()
    conn.close()
    
def update_concept_dict(topic, content):
    data = {}
    if os.path.exists(CONCEPT_PATH):
        try:
            with open(CONCEPT_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            pass
            
    summary = content[:100] + "..." if len(content) > 100 else content
    data[topic] = summary
    
    with open(CONCEPT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def incremental_train():
    pending_phrases = get_pending_corpus(limit=3)
    pending_sentences = get_pending_sentences(limit=3)
    
    pending_rows = []
    if pending_phrases:
        pending_rows.extend([('web_corpus',) + row for row in pending_phrases])
    if pending_sentences:
        pending_rows.extend([('sentence_corpus',) + row for row in pending_sentences])
        
    if not pending_rows:
        print("[-] 数据库中没有新的待学习语料。休息中...")
        return False
        
    print(f"[*] 发现 {len(pending_phrases)} 条新短语，{len(pending_sentences)} 条新长句！启动双轨极低功耗增量训练...")
    
    ws = pickle.load(open("brain/models/ws_unified.pkl", "rb"))
    config.input_ws = ws
    
    model = CognitiveEngine(len(ws)).to(config.device)
    model_path = "brain/models/gpt_model.pkl"
    try:
        model.load_state_dict(torch.load(model_path, map_location=config.device, weights_only=True), strict=False)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False
        
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=5e-5) # 极低学习率，防止遗忘旧知识
    criterion = nn.CrossEntropyLoss(ignore_index=ws.PAD)
    
    epochs = 2
    accumulation_steps = 2
    
    from utils.cut_sentence import cut
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        
        for i, (table_name, cid, topic, content) in enumerate(pending_rows):
            if table_name == "sentence_corpus":
                text = f"{topic} {content}"
            else:
                text = f"{topic}是什么？ {content}"
            words = cut(text, by_word=True)
            indices = ws.transform(words, max_len=len(words))
            indices.insert(0, ws.SOS)
            indices.append(ws.EOS)
            
            input_tensor = torch.LongTensor([indices[:-1]]).to(config.device)
            target_tensor = torch.LongTensor([indices[1:]]).to(config.device)
            
            logits = model(input_tensor)
            
            loss = criterion(logits.view(-1, logits.size(-1)), target_tensor.view(-1))
            loss = loss / accumulation_steps
            loss.backward()
            
            total_loss += loss.item()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(pending_rows):
                optimizer.step()
                optimizer.zero_grad()
                
        print(f"  [Epoch {epoch+1}/{epochs}] Loss: {total_loss:.4f}")
        time.sleep(2) # 极低功耗：每跑完一轮主动休眠降温
        
    torch.save(model.state_dict(), model_path)
    print("[OK] 突触连接重塑完成，模型已保存。")
    
    web_ids = [row[1] for row in pending_rows if row[0] == "web_corpus"]
    sent_ids = [row[1] for row in pending_rows if row[0] == "sentence_corpus"]
    
    mark_as_learned(web_ids, "web_corpus")
    mark_as_learned(sent_ids, "sentence_corpus")
    
    for table_name, cid, topic, content in pending_rows:
        update_concept_dict(topic, content)
        
    print(f"[OK] 常识图谱已更新。本次消化短语 ID: {web_ids}, 长句 ID: {sent_ids}\n")
    return True

if __name__ == "__main__":
    print("=== Cognitive Incremental Learner Daemon ===")
    while True:
        has_data = incremental_train()
        if not has_data:
            time.sleep(10) # 没数据就等10秒
        else:
            time.sleep(3) # 有数据也休眠3秒，防止连轴转导致CPU发热
