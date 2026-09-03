import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import torch
import torch.nn as nn
from utils.db_config import get_mysql_conn
from utils.cut_sentence import cut
from brain.cognitive_engine import CognitiveEngine
import config

print("=== 神经突触扩容手术准备 ===")

# 1. 载入原有词表
ws_path = "brain/models/ws_unified.pkl"
with open(ws_path, "rb") as f:
    ws = pickle.load(f)
old_vocab_size = len(ws)
print(f"原有词表大小: {old_vocab_size}")

# 2. 从数据库提取所有语料
conn = get_mysql_conn()
c = conn.cursor()
c.execute('SELECT topic, content FROM web_corpus')
web_data = c.fetchall()
c.execute('SELECT topic, content FROM sentence_corpus')
sent_data = c.fetchall()
conn.close()

all_texts = []
for topic, content in web_data:
    all_texts.append(f"{topic}是什么？ {content}")
for topic, content in sent_data:
    all_texts.append(f"{topic} {content}")

print(f"读取到 {len(all_texts)} 条语料，开始重新编译词频...")

# 3. 提取新字并扩展词表
new_words_count = 0
for text in all_texts:
    words = cut(text, by_word=True)
    for w in words:
        if w not in ws.dict:
            ws.dict[w] = len(ws.dict)
            new_words_count += 1

# 更新反向字典
ws.inverse_dict = dict(zip(ws.dict.values(), ws.dict.keys()))
new_vocab_size = len(ws)
print(f"新增词汇量: {new_words_count}")
print(f"新词表总大小: {new_vocab_size}")

if new_words_count == 0:
    print("没有发现新词汇，无需扩容。")
    sys.exit(0)

# 4. 载入旧模型权重
model_path = "brain/models/gpt_model.pkl"
old_model = CognitiveEngine(old_vocab_size).to(config.device)
old_model.load_state_dict(torch.load(model_path, map_location=config.device, weights_only=True))

# 5. 构建新模型
new_model = CognitiveEngine(new_vocab_size).to(config.device)

# 6. 张量外科手术：无损复制旧权重
print("正在执行张量拷贝 (Tensor Surgery)...")
old_state = old_model.state_dict()
new_state = new_model.state_dict()

for name, param in old_state.items():
    if name == 'embedding.weight' or name == 'fc_out.weight':
        # 扩展维度 0: [vocab_size, d_model]
        new_state[name][:old_vocab_size, :] = param
    elif name == 'fc_out.bias':
        # 扩展维度 0: [vocab_size]
        new_state[name][:old_vocab_size] = param
    else:
        # 其他层保持不变
        new_state[name] = param

new_model.load_state_dict(new_state)

# 7. 保存更新
with open(ws_path, "wb") as f:
    pickle.dump(ws, f)
torch.save(new_model.state_dict(), model_path)

print(f"=== 手术成功完成 ===")
print(f"模型和词汇表已无损扩容至 {new_vocab_size} 维！")
