import torch
import config
from brain.cognitive_engine import CognitiveEngine
import pickle
import torch.nn.functional as F
from brain.memory_bank import MemoryBank
from brain.cognitive_surprise import CognitiveSurpriseEvaluator
import json
import os

def load_concept_dict():
    path = "corpus/knowledge/concept_dict.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                pass
    return {}

def extract_concepts(prompt, concept_dict, ws):
    found_concepts = {}
    unknown_nouns = []
    
    # 1. Exact string matching for known concepts
    for k in concept_dict:
        if k in prompt:
            found_concepts[k] = concept_dict[k]
            
    # 2. Use jieba to find unknown nouns
    try:
        import jieba.posseg as pseg
        pos_words = pseg.cut(prompt)
        for w, flag in pos_words:
            if flag.startswith('n') or flag == 'x': # noun or unknown
                if w not in concept_dict and w not in ws.dict and len(w) > 1:
                    unknown_nouns.append(w)
    except:
        pass
    return found_concepts, unknown_nouns

def generate_cognitive_response(model, ws, memory_bank, evaluator, prompt, max_len=50, temperature=0.65, top_k=50, top_p=0.85):
    model.eval()
    from utils.cut_sentence import cut
    
    # --- Phase 1: Lexical Cognition (词汇级认知) ---
    concept_dict = load_concept_dict()
    found_concepts, unknown_nouns = extract_concepts(prompt, concept_dict, ws)
    
    if unknown_nouns:
        # Phase 14: "外挂左脑" 神经符号系统 (Neuro-Symbolic Injection)
        print(f"  [词汇盲区] 发现未知概念: {unknown_nouns}")
        try:
            from brain.knowledge_graph import KnowledgeGraphEngine
            kg_engine = KnowledgeGraphEngine()
            
            for noun in unknown_nouns[:]:
                print(f"  [启动逻辑寻路] 正在全网检索 '{noun}' 的外部语义图谱...")
                # 1. 尝试基于拓扑重叠寻找关系 (旧逻辑)
                mapped_concept = None
                for known_concept in concept_dict.keys():
                    if noun in known_concept or known_concept in noun:
                        mapped_concept = known_concept
                        break
                
                # 2. 如果拓扑不匹配，调用外部维基图谱推理 (新逻辑)
                if not mapped_concept:
                    mapped_concept = kg_engine.infer_relationship(noun, concept_dict.keys())
                
                if mapped_concept:
                    found_concepts[mapped_concept] = concept_dict[mapped_concept]
                    unknown_nouns.remove(noun)
                    print(f"  [神经符号注入] 发现逻辑拓扑链：{noun} -> {mapped_concept}。强制激活“{mapped_concept}”神经元族群！")
        except Exception as e:
            print(f"  [神经符号系统异常] 逻辑寻路失败: {e}")
            pass
        
    if unknown_nouns:
        return f"[词汇盲区警告] 我的深层神经元尚未建立对“{'、'.join(set(unknown_nouns))}”的概念认知，且外部逻辑寻路失败，无法解析该词汇。"
    
    # Encode prompt
    prompt_words = cut(prompt, by_word=True)
    indices = ws.transform(prompt_words, max_len=len(prompt_words))
    indices.append(ws.EOS)
    input_tensor = torch.LongTensor([indices]).to(config.device)
    
    with torch.no_grad():
        logits, hidden_states = model(input_tensor, return_hidden=True)
        # Use EOS token representation
        query_vector = hidden_states[:, -1, :].squeeze(0)
        
        # Calculate Generative Perplexity (Prompt Loss)
        # Predict the prompt itself
        target_tensor = input_tensor[:, 1:].contiguous().view(-1)
        pred_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        
        if len(target_tensor) > 0:
            prompt_loss = F.cross_entropy(pred_logits, target_tensor).item()
        else:
            prompt_loss = 0.0
    
    # Evaluate Dual-CSI
    is_surprised, max_sim, csi_score, dyn_threshold = evaluator.evaluate_csi(query_vector, prompt_loss)
    
    # --- Override Mechanism ---
    # 如果在概念图谱中提取到了实体名词，说明模型“认识”这个核心词，强行压制语句级的认知意外。
    if found_concepts:
        is_surprised = False
        
    if is_surprised:
        # Save to anomaly buffer
        anomaly_record = {
            "query": prompt,
            "csi_score": float(csi_score),
            "max_sim": float(max_sim)
        }
        buffer_path = "corpus/knowledge/anomaly_buffer.json"
        buffer_data = []
        if os.path.exists(buffer_path):
            with open(buffer_path, "r", encoding="utf-8") as f:
                try:
                    buffer_data = json.load(f)
                except:
                    pass
        buffer_data.append(anomaly_record)
        with open(buffer_path, "w", encoding="utf-8") as f:
            json.dump(buffer_data, f, ensure_ascii=False, indent=2)
            
        print(f"  [Cognitive Anomaly Detected] CSI: {csi_score:.2f} (Sim: {max_sim:.2f}, Loss: {prompt_loss:.2f} > DynThresh: {dyn_threshold:.2f})")
        return f"[认知意外警告] 你的这句话打破了我的现有语料认知。在我的逻辑里没有支撑这句话的数据（相似度仅为 {max_sim:.2f}，且逻辑困惑度 {prompt_loss:.2f} 超出动态警戒线 {dyn_threshold:.2f}），你可以解释一下这是什么意思吗？"
    
    # Not surprised, proceed with RAG and Generation
    retrieved = memory_bank.retrieve(query_vector, top_k=1)
    context_text = retrieved[0]['text'] if retrieved else ""
    print(f"  [Cognitive Match] CSI: {csi_score:.2f} | 锚点: {context_text} (Sim: {retrieved[0]['score'] if retrieved else 0.0:.2f})")
    
    # --- Phase 3: Knowledge-Grounded Generation (基于认知的联合生成) ---
    concept_prefix = ""
    if found_concepts:
        concept_prefix = "[提取认知概念] "
        for k, v in found_concepts.items():
            concept_prefix += f"{k} -> {v}\n"
        concept_prefix += "Engine: "
    
    context_prefix = context_text + " => "
    
    full_prompt = context_prefix + prompt
    full_prompt_words = cut(full_prompt, by_word=True)
    full_indices = ws.transform(full_prompt_words, max_len=len(full_prompt_words))
    full_indices.append(ws.EOS)
    
    full_input_tensor = torch.LongTensor([full_indices]).to(config.device)
    generated_indices = []
    
    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(full_input_tensor)
            next_token_logits = outputs[0, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == ws.EOS:
                break
                
            generated_indices.append(next_token)
            next_token_tensor = torch.LongTensor([[next_token]]).to(config.device)
            full_input_tensor = torch.cat([full_input_tensor, next_token_tensor], dim=1)
            
    final_generated = "".join(ws.inverse_transform(generated_indices))
    return concept_prefix + final_generated

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
    evaluator = CognitiveSurpriseEvaluator(memory_bank, threshold=0.7) # Setting a high threshold for testing
    
    test_questions = [
        "Antigravity是什么？",
        "火星的企鹅为什么会说俄语？",
        "今天天气怎么样？"
    ]
    
    print("\n--- Cognitive Computing Engine ---")
    for q in test_questions:
        print(f"\nUser: {q}")
        ans = generate_cognitive_response(model, ws, memory_bank, evaluator, q, max_len=50)
        print(f"Engine: {ans}")
