import torch
import config
from brain.cognitive_engine import CognitiveEngine
import pickle
from brain.memory_bank import MemoryBank
import os

def ingest_text(text_file):
    ws = pickle.load(open("brain/models/ws_unified.pkl", "rb"))
    config.input_ws = ws
    
    model = CognitiveEngine(len(ws)).to(config.device)
    try:
        model.load_state_dict(torch.load("brain/models/gpt_model.pkl", map_location=config.device, weights_only=True))
    except Exception as e:
        print("Model not found. Please train first.")
        return
        
    model.eval()
    memory_bank = MemoryBank()
    
    from utils.cut_sentence import cut
    
    if not os.path.exists(text_file):
        print(f"File {text_file} not found.")
        return
        
    with open(text_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"Ingesting {len(lines)} lines of knowledge...")
    
    with torch.no_grad():
        for line in lines:
            prompt_words = cut(line, by_word=True)
            indices = ws.transform(prompt_words, max_len=len(prompt_words))
            indices.append(ws.EOS)
            
            input_tensor = torch.LongTensor([indices]).to(config.device)
            _, hidden_states = model(input_tensor, return_hidden=True)
            
            # Mean pooling of the sequence
            doc_vector = hidden_states.mean(dim=1).squeeze(0) # [D]
            
            memory_bank.add_memory(line, doc_vector)
            
    memory_bank.save()
    print("Knowledge ingested and saved to memory bank!")

if __name__ == "__main__":
    os.makedirs("corpus/knowledge", exist_ok=True)
    file_path = "corpus/knowledge/test_doc.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Antigravity是由Google Deepmind团队设计的一款高级AI代理助手。\n")
        f.write("Antigravity最擅长的领域是自主编程和重构底层算法。\n")
        f.write("它的核心能力是不需要修改源代码参数就能通过RAG自适应学习新知识。\n")
        
    ingest_text(file_path)
