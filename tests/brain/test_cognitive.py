import torch
import config
from brain.cognitive_engine import CognitiveEngine
import pickle
import torch.nn.functional as F
from brain.memory_bank import MemoryBank

def generate_text_with_rag(model, ws, memory_bank, prompt, max_len=50, temperature=0.8, top_k=50, top_p=0.9):
    model.eval()
    from utils.cut_sentence import cut
    
    # 1. Encode prompt for retrieval
    prompt_words = cut(prompt, by_word=True)
    indices = ws.transform(prompt_words, max_len=len(prompt_words))
    indices.append(ws.EOS)
    input_tensor = torch.LongTensor([indices]).to(config.device)
    
    with torch.no_grad():
        _, hidden_states = model(input_tensor, return_hidden=True)
        query_vector = hidden_states.mean(dim=1).squeeze(0)
    
    # 2. Retrieve top-1 memory
    retrieved = memory_bank.retrieve(query_vector, top_k=1)
    
    context_prefix = ""
    # We use a very low threshold because the model might not be perfectly trained
    if retrieved and retrieved[0]['score'] > 0.0: 
        context_text = retrieved[0]['text']
        print(f"  [RAG 检索到背景知识: {context_text} (相似度: {retrieved[0]['score']:.2f})]")
        # Append context to prompt (like a few-shot or context injection)
        context_prefix = context_text + " => "
    else:
        print("  [RAG 未检索到强相关背景知识]")
        
    # 3. Generate answer
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
            
            # Top-P / Top-K to avoid loops
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
            
    return ws.inverse_transform(generated_indices)

if __name__ == "__main__":
    ws = pickle.load(open("brain/models/ws_unified.pkl", "rb"))
    config.input_ws = ws
    
    model = CognitiveEngine(len(ws)).to(config.device)
    try:
        model.load_state_dict(torch.load("brain/models/gpt_model.pkl", map_location=config.device, weights_only=True))
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit()
        
    memory_bank = MemoryBank()
    
    test_questions = [
        "Antigravity是什么？",
        "Antigravity擅长做什么？",
        "今天天气怎么样？"
    ]
    
    print("\n--- RAG-Enhanced GPT Model Inference ---")
    for q in test_questions:
        print(f"\nUser: {q}")
        ans = generate_text_with_rag(model, ws, memory_bank, q, max_len=50)
        print(f"GPT:  {ans}")
