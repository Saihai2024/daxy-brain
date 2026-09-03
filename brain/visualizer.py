import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import config
from brain.cognitive_engine import CognitiveEngine
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from utils.cut_sentence import cut

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows 默认支持的中文字体
plt.rcParams['axes.unicode_minus'] = False

def visualize_attention(model, ws, prompt, save_path="brain/attention_heatmap.png"):
    model.eval()
    
    prompt_words = cut(prompt, by_word=True)
    indices = ws.transform(prompt_words, max_len=len(prompt_words))
    indices.append(ws.EOS)
    labels = prompt_words + ["<EOS>"]
    
    input_tensor = torch.LongTensor([indices]).to(config.device)
    
    with torch.no_grad():
        _ = model(input_tensor)
        
    num_layers = len(model.layers)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Neural Attention Matrix across 12 Layers\nPrompt: "{prompt}"', fontsize=16)
    
    for i, layer in enumerate(model.layers):
        # last_attn shape: [B, nhead, L, L] -> we average across heads
        attn_matrix = layer.attn.last_attn[0].mean(dim=0).cpu().numpy()
        
        ax = axes[i // 4, i % 4]
        sns.heatmap(attn_matrix, xticklabels=labels, yticklabels=labels, cmap="viridis", ax=ax, cbar=False)
        ax.set_title(f"Layer {i+1} Attention")
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    print(f"[可视化系统] 12层神经网络突触热力图已成功保存至: {save_path}")

if __name__ == "__main__":
    try:
        ws = pickle.load(open("brain/models/ws_unified.pkl", "rb"))
        config.input_ws = ws
        
        model = CognitiveEngine(len(ws)).to(config.device)
        model.load_state_dict(torch.load("brain/models/gpt_model.pkl", map_location=config.device, weights_only=True))
        
        test_prompt = "火星有生命吗"
        visualize_attention(model, ws, test_prompt)
    except Exception as e:
        print(f"Failed: {e}")
