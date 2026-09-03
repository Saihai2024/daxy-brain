import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
import config
import pickle
import time
import urllib.request
import re
from PIL import Image
from brain.cognitive_engine import CognitiveEngine
from skills.autonomous_crawler import scrape_concept, get_mysql_conn

# Make sure we have an image directory
IMG_DIR = "corpus/images"
os.makedirs(IMG_DIR, exist_ok=True)

class VisualCuriosityCrawler:
    def __init__(self):
        self.ws = pickle.load(open("brain/models/ws_unified.pkl", "rb"))
        config.input_ws = self.ws
        self.model = CognitiveEngine(len(self.ws)).to(config.device)
        self.load_model()
        
    def load_model(self):
        print("\n[视觉皮层] 正在加载多模态权重...")
        try:
            self.model.load_state_dict(torch.load("brain/models/gpt_model.pkl", map_location=config.device, weights_only=True), strict=False)
            self.model.eval()
            print("[视觉皮层] 视觉神经元加载完毕！")
        except Exception as e:
            print(f"[视觉皮层] 权重加载失败: {e}")

    def download_sample_image(self):
        import random
        import io
        # 多样化的视觉认知图库池
        image_pool = [
            ("火星", "https://upload.wikimedia.org/wikipedia/commons/0/02/OSIRIS_Mars_true_color.jpg"),
            ("黑洞", "https://upload.wikimedia.org/wikipedia/commons/4/4f/Black_hole_-_Messier_87_crop_max_res.jpg"),
            ("DNA", "https://upload.wikimedia.org/wikipedia/commons/3/3c/DNA_animation.gif"),
            ("大脑", "https://upload.wikimedia.org/wikipedia/commons/b/b2/Human_brain_NIH.png"),
            ("量子计算机", "https://upload.wikimedia.org/wikipedia/commons/1/1b/IBM_Q_System_One_Cryostat.jpg"),
            ("猫", "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"),
            ("银河系", "https://upload.wikimedia.org/wikipedia/commons/c/c3/NGC_4414_%28NASA-med%29.jpg")
        ]

        random.shuffle(image_pool)
        for subject, url in image_pool:
            filename = f"sample_{subject}.jpg"
            filepath = os.path.join(IMG_DIR, filename)

            # 已有真实图片则直接使用；遇到旧的灰色占位图(<5KB)则强制重下
            if os.path.exists(filepath) and os.path.getsize(filepath) > 5 * 1024:
                try:
                    with Image.open(filepath) as im:
                        im.verify()
                    return filepath, subject
                except Exception:
                    pass  # 文件损坏，落到下载分支重下

            print(f"[视觉爬虫] 正在捕获新的视觉目标 ({subject}) -> {url}")
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'DaxyCognitiveCrawler/1.0 (research; contact: localhost)',
                    'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Referer': 'https://commons.wikimedia.org/'
                })
                with urllib.request.urlopen(req, timeout=20) as response:
                    ctype = response.headers.get('Content-Type', '')
                    data = response.read()

                if not ctype.startswith('image/'):
                    raise ValueError(f"返回的不是图片(Content-Type={ctype}, len={len(data)})")
                if len(data) < 5 * 1024:
                    raise ValueError(f"图片体积过小，疑似占位图({len(data)} bytes)")

                # 用 PIL 解码校验，统一转 RGB JPEG，规避 GIF/PNG 训练侧异常
                img = Image.open(io.BytesIO(data)).convert('RGB')
                img.save(filepath, format='JPEG', quality=92)
                print(f"[视觉爬虫] OK 已保存 {filepath} ({os.path.getsize(filepath)} bytes, {img.size})")
                return filepath, subject
            except Exception as e:
                print(f"[视觉爬虫] 下载失败({subject}): {e}，跳过此目标")
                # 关键修复：失败时绝不写灰色占位图，直接尝试下一个 url
                continue

        # 整轮全部失败：抛出，让 run_loop 走异常分支休眠重试，而不是污染训练数据
        raise RuntimeError("本轮 image_pool 全部下载失败，等待网络恢复")

    def generate_question_from_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        # Prepare image tensor using vision_encoder's transform
        from brain.vision_encoder import CognitiveVisionEncoder
        dummy_encoder = CognitiveVisionEncoder()
        img_tensor = dummy_encoder.transform(img).unsqueeze(0).to(config.device)
        
        # We start the prompt with "这是"
        prompt = "这是"
        from utils.cut_sentence import cut
        words = cut(prompt, by_word=True)
        indices = self.ws.transform(words, max_len=len(words))
        input_tensor = torch.LongTensor([indices]).to(config.device)
        
        generated_indices = []
        entropy_sum = 0.0
        
        with torch.no_grad():
            for _ in range(10): # Generate a short question
                outputs = self.model(input_tensor, image_tensor=img_tensor)
                next_token_logits = outputs[0, -1, :]
                
                # Calculate Visual Cognitive Surprise (Entropy)
                probs = F.softmax(next_token_logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
                entropy_sum += entropy
                
                # Sample
                probs = F.softmax(next_token_logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                if next_token == self.ws.EOS:
                    break
                generated_indices.append(next_token)
                next_tensor = torch.LongTensor([[next_token]]).to(config.device)
                input_tensor = torch.cat([input_tensor, next_tensor], dim=1)
                
        text = "".join(self.ws.inverse_transform(generated_indices))
        avg_surprise = entropy_sum / max(1, len(generated_indices))
        return prompt + text + "？", avg_surprise

    def run_loop(self):
        print("\n=============================================")
        print("[START] 多模态视觉好奇心引擎 (Continuous Vision-Language Crawler)")
        print("=============================================\n")
        
        while True:
            try:
                img_path, subject = self.download_sample_image()
                print(f"\n[*] 发现视觉目标: {img_path}")
                
                question, surprise = self.generate_question_from_image(img_path)
                print(f"  [视觉认知惊奇度] Visual CSI: {surprise:.2f}")
                
                if surprise > 2.0:
                    print(f"  [产生视觉疑惑] 模型看图后提出问题: “{question}”")
                    print(f"  [全网求知] 正在查阅关于 '{subject}' 的知识...")
                    
                    answer = scrape_concept(subject)
                    if answer:
                        print(f"  [OK] 获得视觉对象解析: {answer[:30]}...")
                        # Insert into vision_corpus
                        conn = get_mysql_conn()
                        cursor = conn.cursor()
                        cursor.execute(
                            'INSERT INTO vision_corpus (image_path, question, answer, surprise_score) VALUES (%s, %s, %s, %s)',
                            (img_path, question, answer, surprise)
                        )
                        conn.commit()
                        conn.close()
                        print("  [记忆固化] 视觉-文本知识对已写入 MySQL `vision_corpus`！")
                    else:
                        print("  [求知失败] 未能从全网找到答案。")
                else:
                    print(f"  [视觉认知稳定] 模型认为它认识这个东西: {question}")
                    
                print("[休眠] 视觉皮层冷却中，30秒后扫描下一张图像...")
                time.sleep(30)
            except Exception as e:
                print(f"[视觉爬虫异常] {e}")
                time.sleep(60)

if __name__ == "__main__":
    crawler = VisualCuriosityCrawler()
    crawler.run_loop()
