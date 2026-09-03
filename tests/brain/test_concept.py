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
    try:
        model.load_state_dict(torch.load("brain/models/gpt_model.pkl", map_location=config.device, weights_only=True), strict=False)
    except Exception as e:
        print(f"Failed to load Cognitive Engine: {e}")
        exit()
        
    memory_bank = MemoryBank()
    evaluator = CognitiveSurpriseEvaluator(memory_bank, space_threshold=0.85, ppl_multiplier=1.3) 
    
    test_questions = [
        "你认识黑洞是什么吗？",
        "你能解释一下光速吗？",
        "你知道戴森球吗？",
        "你了解降维打击吗？",
        "什么是暗物质？",
        "量子纠缠是什么鬼？",
        "天空为什么是蓝色的？", # Test long sentence / QA mechanism
        "你听说过降维吗？" # Test generalization (降维 -> 降维打击)
    ]

    print(f"Loaded {len(test_questions)} conceptual test questions...")
    
    with open("brain/test_concept_out.txt", "w", encoding="utf-8") as f:
        f.write("--- 词汇级认知测试 (Phase 5) ---\n\n")
        
        for i, q in enumerate(test_questions):
            ans = generate_cognitive_response(model, ws, memory_bank, evaluator, q, max_len=50)
            f.write(f"Q{i+1}: {q}\n")
            f.write(f"Engine: {ans}\n")
            f.write("-" * 40 + "\n")
            print(f"Processed: {q}")
            
    print("Done! Results saved to brain/test_concept_out.txt")
