import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = d_model ** -0.5
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w2 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w3 = nn.Linear(dim_feedforward, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class MoESwiGLU(nn.Module):
    def __init__(self, d_model, dim_feedforward, num_experts=4, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, dim_feedforward) for _ in range(num_experts)])

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        final_output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i).any(dim=-1)
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = expert(expert_input)
                weight = routing_weights[expert_mask, 0].unsqueeze(-1)
                final_output[expert_mask] += expert_output * weight
                
        return final_output.view(B, L, D)

def apply_rotary_emb(q, k, seq_len, d_head):
    # RoPE implementation
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head)).to(q.device)
    t = torch.arange(seq_len, device=q.device).type_as(inv_freq)
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    
    cos = emb.cos().unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, d_head]
    sin = emb.sin().unsqueeze(0).unsqueeze(2)
    
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.d_head)
        k = self.k_proj(x).view(B, L, self.nhead, self.d_head)
        v = self.v_proj(x).view(B, L, self.nhead, self.d_head)
        
        # Apply RoPE
        q, k = apply_rotary_emb(q, k, L, self.d_head)
        
        q = q.transpose(1, 2) # [B, nhead, L, d_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            # mask: [L, L] or [B, 1, L, L]
            scores = scores + mask
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        self.last_attn = attn.detach()  # 供可视化模块挂载提取
        
        out = torch.matmul(attn, v) # [B, nhead, L, d_head]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)

class CognitiveReasoningLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = SelfAttention(d_model, nhead, dropout)
        self.norm1 = RMSNorm(d_model)
        
        self.ffn = MoESwiGLU(d_model, dim_feedforward, num_experts=4)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class CognitiveEngine(nn.Module):
    def __init__(self, vocab_size):
        super(CognitiveEngine, self).__init__()
        self.d_model = config.transformer_d_model
        self.num_layers = config.transformer_num_layers
        
        self.embedding = nn.Embedding(
            vocab_size, 
            self.d_model, 
            padding_idx=config.input_ws.PAD if hasattr(config, 'input_ws') and config.input_ws else 0
        )
        
        # 视觉皮层 (Vision Encoder) - 延迟初始化
        self.vision_encoder = None
        
        # True Deep Architecture (LLaMA style): Instantiate independent physical layers
        self.layers = nn.ModuleList([
            CognitiveReasoningLayer(
                d_model=self.d_model,
                nhead=config.transformer_nhead,
                dim_feedforward=config.transformer_dim_feedforward,
                dropout=config.transformer_dropout
            ) for _ in range(self.num_layers)
        ])
        
        self.final_norm = RMSNorm(self.d_model)
        self.fc_out = nn.Linear(self.d_model, vocab_size, bias=False)

    def _get_vision_encoder(self):
        if self.vision_encoder is None:
            from brain.vision_encoder import CognitiveVisionEncoder
            self.vision_encoder = CognitiveVisionEncoder(self.d_model).to(next(self.parameters()).device)
        return self.vision_encoder

    def generate_square_subsequent_mask(self, sz):
        # Creates a causal mask
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.unsqueeze(0).unsqueeze(0).to(config.device) # [1, 1, sz, sz] for multi-head broadcast

    def forward(self, src, image_tensor=None, return_hidden=False):
        """
        src: [batch_size, seq_len]
        image_tensor: [batch_size, 3, H, W]
        """
        # Word embedding + positional embedding is implicitly handled by RoPE inside attention
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # [多模态融合]：如果有图片输入，提取视觉Token并拼接到前面
        if image_tensor is not None:
            vision_encoder = self._get_vision_encoder()
            visual_tokens = vision_encoder(image_tensor) # [B, NumPatches, D]
            # 拼接: [B, NumPatches + L, D]
            x = torch.cat([visual_tokens, x], dim=1)
            
        seq_len = x.size(1)
        mask = self.generate_square_subsequent_mask(seq_len)
        
        # Loop over the INDEPENDENT deep layers
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        x = self.final_norm(x)
        logits = self.fc_out(x)
        
        # 如果拼接了视觉Token，我们只返回文本部分的 logits，方便原本的文本生成代码
        if image_tensor is not None:
            num_visual_tokens = visual_tokens.size(1)
            logits = logits[:, num_visual_tokens:, :]
            x = x[:, num_visual_tokens:, :]
        
        if return_hidden:
            return logits, x
        return logits
