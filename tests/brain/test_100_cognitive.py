import torch
import config
from brain.cognitive_engine import CognitiveEngine
from brain.word_sequece import WordSequence
import pickle
import random
import torch.nn.functional as F

def generate_text(model, ws, prompt, max_len=50, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2):
    model.eval()
    from utils.cut_sentence import cut
    prompt_words = cut(prompt, by_word=True)
    
    indices = ws.transform(prompt_words, max_len=len(prompt_words))
    indices.append(ws.EOS)
    
    input_tensor = torch.LongTensor([indices]).to(config.device)
    generated_indices = []
    
    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(input_tensor)
            next_token_logits = outputs[0, -1, :]
            
            # Apply repetition penalty
            for idx in set(generated_indices):
                if next_token_logits[idx] < 0:
                    next_token_logits[idx] *= repetition_penalty
                else:
                    next_token_logits[idx] /= repetition_penalty
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Top-K filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
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
            input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
            
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
        
    # Read 100 random questions from test_input.txt or train_input.txt
    try:
        with open("corpus/brain/test_input.txt", "r", encoding="utf-8") as f:
            lines = [line.strip().replace(" ", "") for line in f.readlines() if line.strip()]
        test_questions = random.sample(lines, min(100, len(lines)))
    except:
        # Fallback if file doesn't exist or is empty
        test_questions = ["你吃饭了吗", "今天想做点什么", "我心情不好", "能陪我聊天吗"] * 25

    print(f"Loaded {len(test_questions)} questions. Generating answers...")
    
    with open("brain/test_100_out.txt", "w", encoding="utf-8") as f:
        f.write("--- GPT Model 100 Questions Test ---\n\n")
        
        for i, q in enumerate(test_questions):
            ans = generate_text(model, ws, q, max_len=50)
            f.write(f"Q{i+1}: {q}\n")
            f.write(f"GPT: {ans}\n")
            f.write("-" * 40 + "\n")
            
            # Print progress to console
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/100")
    
    print("Done! Results saved to brain/test_100_out.txt")
