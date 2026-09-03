import torch
import torch.nn.functional as F

def logic_contrastive_loss(hidden_states, input_ids, ws, temperature=0.1):
    """
    Computes a lightweight contrastive loss by aligning the question representation 
    and the answer representation within the same sequence.
    This runs in the same single forward pass, zero extra CPU cost.
    """
    B, L, D = hidden_states.shape
    eos_idx = ws.EOS
    
    # 1. Find the first EOS token index for each sequence (separates input and target)
    is_eos = (input_ids == eos_idx)
    # torch.max returns (values, indices). We want the indices of the first EOS.
    _, first_eos_idx = torch.max(is_eos.int(), dim=1)
    
    # Extract representation of the question
    h_q = hidden_states[torch.arange(B), first_eos_idx]
    
    # 2. Find the last valid token index (before padding)
    is_not_pad = (input_ids != ws.PAD).int()
    last_valid_idx = is_not_pad.sum(dim=1) - 1
    # Safety clamp
    last_valid_idx = torch.max(last_valid_idx, first_eos_idx)
    
    # Extract representation of the answer
    h_a = hidden_states[torch.arange(B), last_valid_idx]
    
    # 3. L2 Normalize for cosine similarity
    h_q = F.normalize(h_q, p=2, dim=1)
    h_a = F.normalize(h_a, p=2, dim=1)
    
    # 4. Compute cosine similarity matrix [B, B]
    cos_sim = torch.matmul(h_q, h_a.transpose(0, 1)) / temperature
    
    # 5. Contrastive InfoNCE Loss (Diagonal should be 1, rest 0)
    labels = torch.arange(B).to(hidden_states.device)
    loss_q2a = F.cross_entropy(cos_sim, labels)
    loss_a2q = F.cross_entropy(cos_sim.transpose(0, 1), labels)
    
    return (loss_q2a + loss_a2q) / 2
