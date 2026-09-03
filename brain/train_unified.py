import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import json
import pickle
import time
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from brain.cognitive_engine import CognitiveEngine
from utils.db_config import get_mysql_conn
from utils.cut_sentence import cut

CONCEPT_PATH = "corpus/knowledge/concept_dict.json"

class UnifiedCognitiveTrainer:
    def __init__(self):
        self.ws = pickle.load(open("brain/models/ws_unified.pkl", "rb"))
        config.input_ws = self.ws
        self.model = CognitiveEngine(len(self.ws)).to(config.device)
        self.model_path = "brain/models/gpt_model.pkl"
        
        self.load_model()
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ws.PAD)

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=config.device, weights_only=True), strict=False)
            print("[统一训练中枢] 成功加载多模态权重。")
        except Exception as e:
            print(f"[统一训练中枢] 加载失败: {e}")

    def save_model(self):
        # We save to a temp file first, then replace to prevent corruption
        temp_path = self.model_path + ".tmp"
        torch.save(self.model.state_dict(), temp_path)
        os.replace(temp_path, self.model_path)
        
    def _fetch_rows(self, query, limit):
        conn = get_mysql_conn()
        cursor = conn.cursor()
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return rows
        
    def get_pending_text(self, limit=3):
        phrases = self._fetch_rows('SELECT id, topic, content FROM web_corpus WHERE status="pending" LIMIT %s', limit)
        sentences = self._fetch_rows('SELECT id, topic, content FROM sentence_corpus WHERE status="pending" LIMIT %s', limit)
        return phrases, sentences

    def get_pending_vision(self, limit=1):
        return self._fetch_rows('SELECT id, image_path, question, answer FROM vision_corpus WHERE status="pending" LIMIT %s', limit)

    def mark_as_learned(self, ids, table):
        if not ids: return
        conn = get_mysql_conn()
        cursor = conn.cursor()
        format_strings = ','.join(['%s'] * len(ids))
        cursor.execute(f'UPDATE {table} SET status="learned" WHERE id IN ({format_strings})', tuple(ids))
        conn.commit()
        conn.close()
        
    def update_concept_dict(self, topic, content):
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

    def train_text_batch(self, phrases, sentences):
        pending_rows = []
        if phrases: pending_rows.extend([('web_corpus',) + row for row in phrases])
        if sentences: pending_rows.extend([('sentence_corpus',) + row for row in sentences])
        
        if not pending_rows: return False
        
        print(f"[*] [文本左脑] 发现 {len(phrases)} 条新短语，{len(sentences)} 条新长句，开始吸收...")
        self.model.train()
        
        accumulation_steps = 2
        self.optimizer.zero_grad()
        total_loss = 0
        
        for i, (table_name, cid, topic, content) in enumerate(pending_rows):
            text = f"{topic} {content}" if table_name == "sentence_corpus" else f"{topic}是什么？ {content}"
            words = cut(text, by_word=True)
            indices = self.ws.transform(words, max_len=len(words))
            indices.insert(0, self.ws.SOS)
            indices.append(self.ws.EOS)
            
            input_tensor = torch.LongTensor([indices[:-1]]).to(config.device)
            target_tensor = torch.LongTensor([indices[1:]]).to(config.device)
            
            logits = self.model(input_tensor)
            loss = self.criterion(logits.view(-1, logits.size(-1)), target_tensor.view(-1))
            loss = loss / accumulation_steps
            loss.backward()
            total_loss += loss.item()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(pending_rows):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
        print(f"  [Text Loss: {total_loss:.4f}]")
        self.save_model()
        
        self.mark_as_learned([r[1] for r in pending_rows if r[0] == "web_corpus"], "web_corpus")
        self.mark_as_learned([r[1] for r in pending_rows if r[0] == "sentence_corpus"], "sentence_corpus")
        for r in pending_rows:
            self.update_concept_dict(r[2], r[3])
            
        return True

    def train_vision_batch(self, vision_rows):
        if not vision_rows: return False
        
        print(f"[*] [视觉右脑] 发现 {len(vision_rows)} 张新视觉图像，开始重塑视觉神经突触...")
        self.model.train()
        
        from brain.vision_encoder import CognitiveVisionEncoder
        dummy_encoder = CognitiveVisionEncoder()
        
        for cid, img_path, question, answer in vision_rows:
            try:
                text_target = f"问题：{question} 答案：{answer}" if answer else question
                img = Image.open(img_path).convert('RGB')
                img_tensor = dummy_encoder.transform(img).unsqueeze(0).to(config.device)
                
                words = cut(text_target, by_word=True)
                indices = self.ws.transform(words, max_len=len(words))
                indices.append(self.ws.EOS)
                
                input_tensor = torch.LongTensor([indices[:-1]]).to(config.device)
                target_tensor = torch.LongTensor([indices[1:]]).to(config.device)
                
                self.optimizer.zero_grad()
                logits = self.model(input_tensor, image_tensor=img_tensor)
                loss = self.criterion(logits.view(-1, logits.size(-1)), target_tensor.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                print(f"  [{img_path}] Vision Loss: {loss.item():.4f}")
                self.mark_as_learned([cid], "vision_corpus")
            except Exception as e:
                print(f"  [Error] 处理图像 {img_path} 失败: {e}")
                # mark as error
                conn = get_mysql_conn()
                conn.cursor().execute('UPDATE vision_corpus SET status="error" WHERE id=%s', (cid,))
                conn.commit()
                conn.close()
                
        self.save_model()
        return True

    def run_daemon(self):
        print("=== Unified Cognitive & Vision Training Daemon ===")
        while True:
            phrases, sentences = self.get_pending_text(limit=3)
            vision_rows = self.get_pending_vision(limit=1)
            
            trained_text = self.train_text_batch(phrases, sentences)
            trained_vision = self.train_vision_batch(vision_rows)
            
            if not trained_text and not trained_vision:
                print("[-] 左右脑训练队列为空，休眠 10 秒...")
                time.sleep(10)
            else:
                time.sleep(3) # 防止 CPU 持续过热

if __name__ == "__main__":
    trainer = UnifiedCognitiveTrainer()
    trainer.run_daemon()
