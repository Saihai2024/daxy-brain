import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import time
from PIL import Image
import config
from brain.cognitive_engine import CognitiveEngine
from utils.cut_sentence import cut
from skills.autonomous_crawler import get_mysql_conn

class MultimodalTrainer:
    def __init__(self):
        self.ws = pickle.load(open("brain/models/ws_unified.pkl", "rb"))
        config.input_ws = self.ws
        self.model = CognitiveEngine(len(self.ws)).to(config.device)
        self.load_model()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ws.PAD)
        
    def load_model(self):
        try:
            self.model.load_state_dict(torch.load("brain/models/gpt_model.pkl", map_location=config.device, weights_only=True))
            print("[多模态训练营] 成功加载模型权重！")
        except Exception as e:
            print(f"[多模态训练营] 加载失败: {e}")

    def save_model(self):
        torch.save(self.model.state_dict(), "brain/models/gpt_model.pkl")

    def run_training_loop(self):
        print("\n=============================================")
        print("[START] 多模态视觉增量学习回路 (Continuous Multimodal Training)")
        print("=============================================\n")
        
        while True:
            conn = get_mysql_conn()
            cursor = conn.cursor()
            # Fetch a pending image-text pair
            cursor.execute('SELECT id, image_path, question, answer FROM vision_corpus WHERE status="pending" LIMIT 1')
            row = cursor.fetchone()
            
            if row is None:
                conn.close()
                print("[休眠] 没有新的视觉记忆需要固化，休眠 30 秒...")
                time.sleep(30)
                continue
                
            record_id, img_path, question, answer = row
            
            # Use answer if it exists and is meaningful, otherwise train on the question it generated
            text_target = f"问题：{question} 答案：{answer}" if answer else question
            
            print(f"\n[视觉特训] 开始固化记忆: {img_path}")
            
            try:
                img = Image.open(img_path).convert('RGB')
                from brain.vision_encoder import CognitiveVisionEncoder
                dummy_encoder = CognitiveVisionEncoder()
                img_tensor = dummy_encoder.transform(img).unsqueeze(0).to(config.device)
                
                words = cut(text_target, by_word=True)
                indices = self.ws.transform(words, max_len=len(words))
                indices.append(self.ws.EOS)
                
                input_tensor = torch.LongTensor([indices[:-1]]).to(config.device)
                target_tensor = torch.LongTensor([indices[1:]]).to(config.device)
                
                self.model.train()
                for epoch in range(3): # Train on this specific multimodal memory a few times
                    self.optimizer.zero_grad()

                    logits = self.model(input_tensor, image_tensor=img_tensor)

                    # shape: [1, L, V] -> [1*L, V]
                    loss = self.criterion(logits.view(-1, logits.size(-1)), target_tensor.view(-1))

                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    loss_val = float(loss.item())
                    print(f"  [Epoch {epoch+1}/3] 视觉神经突触重塑 Loss: {loss_val:.4f}")
                    try:
                        cursor.execute(
                            'INSERT INTO training_log (task, record_id, image_path, epoch, loss) VALUES (%s,%s,%s,%s,%s)',
                            ('multimodal', record_id, img_path, epoch + 1, loss_val)
                        )
                        conn.commit()
                    except Exception as log_err:
                        print(f"  [训练日志写入失败] {log_err}")
                
                self.save_model()
                print("  [OK] 多模态记忆固化完成，权重已保存！")
                
                # Update status
                cursor.execute('UPDATE vision_corpus SET status="learned" WHERE id=%s', (record_id,))
                conn.commit()
                
            except Exception as e:
                print(f"  [Error] 训练失败: {e}")
                cursor.execute('UPDATE vision_corpus SET status="error" WHERE id=%s', (record_id,))
                conn.commit()
                
            conn.close()
            time.sleep(5)

if __name__ == "__main__":
    trainer = MultimodalTrainer()
    trainer.run_training_loop()
