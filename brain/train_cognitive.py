import torch
import torch.nn.functional as F
from brain.cognitive_engine import CognitiveEngine
from torch.optim import Adam
import torch.nn as nn
from brain.dataset_gpt import get_gpt_dataloader
from brain.loss_logic import logic_contrastive_loss
from tqdm import tqdm
import config
import numpy as np
import pickle
import os

def train_gpt(epoch, model, optimizer, loss_list, ws):
    data_loader = get_gpt_dataloader(train=True)
    bar = tqdm(data_loader, total=len(data_loader))

    for idx, (inputs, targets) in enumerate(bar):
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)

        optimizer.zero_grad()
        outputs, hidden_states = model(inputs, return_hidden=True) # [batch_size, seq_len, vocab_size]
        
        # 1. Causal LM Loss
        lm_loss = F.cross_entropy(
            outputs.reshape(-1, len(ws)), 
            targets.reshape(-1), 
            ignore_index=ws.PAD
        )
        
        # 2. Logic Contrastive Loss
        contrastive_loss = logic_contrastive_loss(hidden_states, inputs, ws)
        
        # Total Loss (weighted)
        loss = lm_loss + 0.5 * contrastive_loss
        
        loss.backward()
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        loss_list.append(loss.item())
        bar.set_description("GPT Epoch:{} idx:{} loss:{:.6f} grad_norm:{:.4f}".format(epoch, idx, np.mean(loss_list[-100:] if len(loss_list) > 100 else loss_list), total_norm))

        if idx % 100 == 0:
            torch.save(model.state_dict(), "brain/models/gpt_model.pkl")
            torch.save(optimizer.state_dict(), "brain/models/gpt_optimizer.pkl")
            pickle.dump(loss_list, open("brain/models/gpt_loss_list.pkl", "wb"))

if __name__ == '__main__':
    ws = pickle.load(open("brain/models/ws_unified.pkl", "rb"))
    config.input_ws = ws # Inject into config for PAD token usage in model
    
    model = CognitiveEngine(len(ws)).to(config.device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_list = []

    model_path = "brain/models/gpt_model.pkl"
    optimizer_path = "brain/models/gpt_optimizer.pkl"
    loss_list_path = "brain/models/gpt_loss_list.pkl"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=config.device, weights_only=True), strict=False)
        print("Loaded existing GPT model weights (strict=False).")
    # if os.path.exists(optimizer_path):
    #     optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))
    #     print("Loaded existing GPT optimizer state.")
    if os.path.exists(loss_list_path):
        loss_list = pickle.load(open(loss_list_path, "rb"))
        print(f"Loaded existing loss list. Starting from resumed epoch.")

    start_epoch = 0
    for i in range(start_epoch, 50):
        train_gpt(i, model, optimizer, loss_list, ws)
