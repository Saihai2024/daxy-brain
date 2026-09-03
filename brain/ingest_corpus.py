import torch
import config
from brain.cognitive_engine import CognitiveEngine
import pickle
from brain.memory_bank import MemoryBank
import os
import random

def ingest_corpus():
    ws = pickle.load(open("brain/models/ws_unified.pkl", "rb"))
    config.input_ws = ws
    
    model = CognitiveEngine(len(ws)).to(config.device)
    model.load_state_dict(torch.load("brain/models/gpt_model.pkl", map_location=config.device, weights_only=True), strict=False)
    model.eval()
    memory_bank = MemoryBank()
    
    from utils.cut_sentence import cut
    
    with open("corpus/brain/train_input.txt", "r", encoding="utf-8") as f:
        lines = [line.strip().replace(" ", "") for line in f.readlines() if line.strip()]
        
    # Take 1000 random lines from the actual corpus to build the cognitive baseline
    sampled_lines = random.sample(lines, min(1000, len(lines)))
    print(f"Ingesting {len(sampled_lines)} baseline conversation patterns...")
    
    with torch.no_grad():
        for line in sampled_lines:
            prompt_words = cut(line, by_word=True)
            indices = ws.transform(prompt_words, max_len=len(prompt_words))
            indices.append(ws.EOS)
            
            input_tensor = torch.LongTensor([indices]).to(config.device)
            _, hidden_states = model(input_tensor, return_hidden=True)
            # Use EOS token representation to align with SimCSE logic
            doc_vector = hidden_states[:, -1, :].squeeze(0)
            
            memory_bank.add_memory(line, doc_vector)
            
    memory_bank.save()
    print("Corpus ingested!")

if __name__ == "__main__":
    ingest_corpus()
