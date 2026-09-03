import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import config
from brain.cognitive_engine import CognitiveEngine
import pickle
from brain.memory_bank import MemoryBank
from brain.cognitive_surprise import CognitiveSurpriseEvaluator
from brain.cognitive_inference import generate_cognitive_response

if __name__ == "__main__":
    ws = pickle.load(open("brain/models/ws_unified.pkl", "rb"))
    config.input_ws = ws
    
    model = CognitiveEngine(len(ws)).to(config.device)
    model.load_state_dict(torch.load("brain/models/gpt_model.pkl", map_location=config.device, weights_only=True))
        
    memory_bank = MemoryBank()
    evaluator = CognitiveSurpriseEvaluator(memory_bank) 
    
    test_questions = [
        "引力波是什么",
        "什么是相对论",
        "奥巴马是谁"
    ]

    print("--- Neuro-Symbolic Inference Test ---")
    
    for q in test_questions:
        print(f"\nUser: {q}")
        ans = generate_cognitive_response(model, ws, memory_bank, evaluator, q, max_len=50)
        print(f"Engine: {ans}")
        print("-" * 40)
