import torch
import config
from brain.cognitive_engine import CognitiveEngine
import pickle
import random
import os
from brain.memory_bank import MemoryBank
from brain.cognitive_surprise import CognitiveSurpriseEvaluator
from brain.cognitive_inference import generate_cognitive_response

if __name__ == "__main__":
    ws = pickle.load(open("brain/models/ws_unified.pkl", "rb"))
    config.input_ws = ws
    
    model = CognitiveEngine(len(ws)).to(config.device)
    try:
        model.load_state_dict(torch.load("brain/models/gpt_model.pkl", map_location=config.device, weights_only=True), strict=False)
    except Exception as e:
        print(f"Failed to load Cognitive Engine: {e}")
        exit()
        
    memory_bank = MemoryBank()
    # Using the new Dynamic Dual-CSI evaluator parameters
    evaluator = CognitiveSurpriseEvaluator(memory_bank, space_threshold=0.85, ppl_multiplier=1.3) 
    
    try:
        with open("corpus/brain/test_input.txt", "r", encoding="utf-8") as f:
            lines = [line.strip().replace(" ", "") for line in f.readlines() if line.strip()]
        test_questions = random.sample(lines, min(25, len(lines)))
    except:
        test_questions = ["你吃饭了吗", "火星有生命吗", "今天想做点什么", "我心情不好", "黑洞是什么", "能陪我聊天吗"] * 5

    # Always ensure some anomaly questions are present
    if "黑洞是什么" not in test_questions: test_questions[0] = "黑洞是什么"
    if "火星有生命吗" not in test_questions: test_questions[1] = "火星有生命吗"

    print(f"Loaded {len(test_questions)} questions. Generating cognitive responses...")
    
    with open("brain/test_25_cognitive_out.txt", "w", encoding="utf-8") as f:
        f.write("--- Cognitive Engine 25 Questions Test (MoE + Dual-CSI) ---\n\n")
        
        for i, q in enumerate(test_questions[:25]):
            ans = generate_cognitive_response(model, ws, memory_bank, evaluator, q, max_len=50)
            f.write(f"Q{i+1}: {q}\n")
            f.write(f"Engine: {ans}\n")
            f.write("-" * 40 + "\n")
            
            if (i+1) % 5 == 0:
                print(f"Processed {i+1}/25")
    
    print("Done! Results saved to brain/test_25_cognitive_out.txt")
