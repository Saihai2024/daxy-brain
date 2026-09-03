import torch
import torch.nn.functional as F
import sqlite3
import urllib.request
import urllib.parse
import re
import os
import pickle
import time
import random
import sys
import config
from brain.cognitive_engine import CognitiveEngine

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.db_config import get_mysql_conn
from skills.immune_system import CognitiveImmuneSystem

MODEL_PATH = "brain/models/gpt_model.pkl"
WS_PATH = "brain/models/ws_unified.pkl"
RELOAD_INTERVAL = 3600 # 每 3600 秒（1小时）热更新一次模型

# init_db is now handled externally via migration script or db_config

def is_topic_scraped(topic, table="web_corpus"):
    conn = get_mysql_conn()
    cursor = conn.cursor()
    cursor.execute(f'SELECT id FROM {table} WHERE topic=%s', (topic,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def scrape_concept(topic):
    """名词 -> 摘要。先试百度百科,失败自动切 wiki / 360 / sogou / moegirl / bing。
    任意一个返回 >10 字符即算成功。"""
    # 旧通道(百度百科,无需任何外部包)
    try:
        url = 'https://baike.baidu.com/item/' + urllib.parse.quote(topic)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        response = urllib.request.urlopen(req, timeout=5)
        html = response.read().decode('utf-8')

        match = re.search(r'<meta name="description" content="(.*?)">', html)
        if match:
            desc = match.group(1)
            desc = re.sub(r'\[.*?\]', '', desc).replace('秒懂百科', '').replace('百度百科', '').strip()
            if len(desc) > 10:
                return desc
    except Exception:
        pass

    # 多源兜底
    try:
        from skills.multi_source_crawler import fetch_topic
        res = fetch_topic(topic, strategy="first_ok", timeout=8)
        if res and len(res.get("content") or "") > 10:
            return res["content"]
    except Exception:
        pass
    return None

def extract_topics_and_questions(text):
    topics = []
    
    # 1. 尝试提取包含疑问词的完整句子/子句
    import re
    clauses = re.split(r'[，。！？、\n]', text)
    question_words = ['为什么', '怎么', '什么', '如何', '多少', '哪里', '谁', '吗']
    
    for clause in clauses:
        clause = clause.strip()
        clean_clause = clause.replace('<UNK>', '').strip()
        clean_clause = re.sub(r'[a-zA-Z0-9]+', '', clean_clause)
        if len(clean_clause) >= 4 and any(qw in clean_clause for qw in question_words):
            topics.append(clean_clause + "？")
            
    # 2. 回退到提取名词
    try:
        import jieba.posseg as pseg
        words = pseg.cut(text.replace('<UNK>', ''))
        for w, flag in words:
            if flag.startswith('n') and len(w) > 1:
                topics.append(w)
    except Exception:
        pass
        
    return list(set(topics))

def scrape_zhidao(question):
    """长句问答检索。先 Wikipedia search,失败用多源(arxiv / 百科 / bing 等)。"""
    import json
    # 旧通道:Wikipedia Search API
    try:
        url = 'https://zh.wikipedia.org/w/api.php?action=query&list=search&srsearch=' + urllib.parse.quote(question) + '&utf8=&format=json'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=5)
        data = json.loads(response.read().decode('utf-8'))

        search_results = data.get('query', {}).get('search', [])
        if search_results:
            snippet = search_results[0].get('snippet', '')
            ans = re.sub(r'<.*?>', '', snippet).strip()
            ans = re.sub(r'\s+', ' ', ans)
            if len(ans) > 10:
                return ans
    except Exception:
        pass

    # 多源兜底(对长句最有用的是 Bing / DDG 搜索摘要)
    try:
        from skills.multi_source_crawler import fetch_topic
        res = fetch_topic(question, strategy="first_ok", timeout=8)
        if res and len(res.get("content") or "") > 10:
            return res["content"]
    except Exception:
        pass
    return None

class AutonomousAgent:
    def __init__(self):
        self.ws = pickle.load(open(WS_PATH, "rb"))
        config.input_ws = self.ws
        self.model = CognitiveEngine(len(self.ws)).to(config.device)
        self.immune_system = CognitiveImmuneSystem()
        self.load_model()
        
    def load_model(self):
        print("\n[系统] 正在从硬盘热更新认知模型 (Hot-Reload)...")
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=config.device, weights_only=True), strict=False)
            self.model.eval()
            print("[系统] 模型重载成功，已装备最新智力！")
        except Exception as e:
            print(f"[系统] 模型加载失败: {e}")
        self.last_load_time = time.time()
        
    def generate_dream(self, seed="什么是"):
        from utils.cut_sentence import cut
        words = cut(seed, by_word=True)
        indices = self.ws.transform(words, max_len=len(words))
        indices.append(self.ws.EOS)
        input_tensor = torch.LongTensor([indices]).to(config.device)
        
        generated_indices = []
        with torch.no_grad():
            for _ in range(30): # 发散生成30个Token
                outputs = self.model(input_tensor)
                next_token_logits = outputs[0, -1, :] / 0.8 # Temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                if next_token == self.ws.EOS:
                    break
                generated_indices.append(next_token)
                next_tensor = torch.LongTensor([[next_token]]).to(config.device)
                input_tensor = torch.cat([input_tensor, next_tensor], dim=1)
                
        text = "".join(self.ws.inverse_transform(generated_indices))
        return seed + text

    def run_loop(self):
        seeds = ["什么是", "为什么", "关于", "我想了解", "解释一下", "未来的", "科学上的"]
        
        print("\n=============================================")
        print("[START] 启动全自动学习永动机 (Autonomous Learning Loop)")
        print("=============================================\n")
        
        while True:
            try:
                # 1. 检查是否需要热更新模型
                if time.time() - self.last_load_time > RELOAD_INTERVAL:
                    self.load_model()
                    
                # 2. 发散思考：让模型做梦
                seed = random.choice(seeds)
                dream_text = self.generate_dream(seed)
                print(f"\n[脑力发散] 引擎发散思考中: 「{dream_text}」")
                
                # 3. 提取潜意识中产生的新名词或长句问题
                topics = extract_topics_and_questions(dream_text)
                if not topics:
                    print("  [-] 思考结果中未包含有效主题，休息中...")
                    time.sleep(5)
                    continue
                    
                # 4. 探索未知
                found_new = False
                for noun in topics:
                    is_question = "？" in noun or "什么" in noun or "怎么" in noun or len(noun) > 5
                    table_name = "sentence_corpus" if is_question else "web_corpus"
                    
                    if is_topic_scraped(noun, table_name):
                        continue
                    mode = "[问答模式]" if is_question else "[百科模式]"
                    
                    print(f"  [*] 发现全新认知盲区 {mode}，启动触手抓取: 【{noun}】")
                    
                    if is_question:
                        content = scrape_zhidao(noun)
                    else:
                        content = scrape_concept(noun)
                        
                    if content:
                        # [Phase 10] Immune System Check
                        cred = self.immune_system.evaluate_credibility(content)
                        status = 'pending'
                        
                        if cred < 0.6:
                            print(f"  [免疫系统警告] 发现可疑内容！可信度评分: {cred:.2f}。触发错误认知增强...")
                            # 二次抓取增强 (Error Enhancement)
                            enhanced_content = scrape_concept(noun + " 科学定义")
                            if enhanced_content:
                                cred_enhanced = self.immune_system.evaluate_credibility(enhanced_content)
                                if cred_enhanced >= 0.6:
                                    print(f"  [纠偏成功] 成功获取高质量交叉验证数据 (可信度 {cred_enhanced:.2f})")
                                    content = enhanced_content
                                    cred = cred_enhanced
                                else:
                                    print(f"  [纠偏失败] 二次查证内容依然低质，数据将被隔离。")
                                    status = 'quarantined'
                            else:
                                print(f"  [纠偏失败] 无法获取补充信息，数据将被隔离。")
                                status = 'quarantined'
                                
                        try:
                            conn = get_mysql_conn()
                            cursor = conn.cursor()
                            cursor.execute(f'INSERT IGNORE INTO {table_name} (topic, content, status, credibility) VALUES (%s, %s, %s, %s)', (noun, content, status, cred))
                            conn.commit()
                            conn.close()
                            if status == 'pending':
                                print(f"  [OK] 捕获成功 (可信度 {cred:.2f})！知识已同步至 MySQL {table_name}。")
                            else:
                                print(f"  [隔离] 伪科学/低质内容已屏蔽 (可信度 {cred:.2f})。")
                                
                            # 触发衍生搜索: 如果学到了一个短语，自动顺带学习与之相关的长句问答
                            if not is_question and status == 'pending':
                                derived_q = f"为什么会有{noun}？"
                                if not is_topic_scraped(derived_q, "sentence_corpus"):
                                    print(f"  [衍生] 自动衍生好奇心问题：【{derived_q}】")
                                    ans = scrape_zhidao(derived_q)
                                    if ans and self.immune_system.evaluate_credibility(ans) >= 0.6:
                                        conn = get_mysql_conn()
                                        c = conn.cursor()
                                        c.execute('INSERT IGNORE INTO sentence_corpus (topic, content, status, credibility) VALUES (%s, %s, %s, %s)', (derived_q, ans, 'pending', 1.0))
                                        conn.commit()
                                        conn.close()
                                        print(f"  [衍生成功] 句子已入库！")
                        except Exception as e:
                            print(f"  [Error] 数据库写入失败: {e}")
                    else:
                        print(f"  [失败] 无法在全网检索到该词条的有效定义。")
                    
                    time.sleep(3) # 降低网络请求频率
                    
                if not found_new:
                    print("  [-] 思考出的名词均已在知识库中。")
                    
                print("\n[休眠] 本轮认知探索完毕，CPU进入极低功耗模式，60秒后开启下一轮思考...")
                time.sleep(60)
                
            except Exception as e:
                print(f"[异常拦截] 主循环异常: {e}，5秒后重试。")
                time.sleep(5)

if __name__ == "__main__":
    agent = AutonomousAgent()
    agent.run_loop()
