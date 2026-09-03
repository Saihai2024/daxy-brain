"""
进行模型的训练
"""
import torch
import torch.nn.functional as F
from brain.seq2seq import Seq2Seq
from torch.optim import Adam
import torch.nn as nn
from brain.dataset import get_dataloader
from tqdm import tqdm
import config
import numpy as np
import pickle
# from matplotlib import pyplot as plt
# from eval import eval

# Variables moved to main block to prevent import issues

def train(epoch,model,optimizer,loss_list):
    data_loader = get_dataloader(train=True)
    bar = tqdm(data_loader,total=len(data_loader))

    for idx,(input,target,input_len,target_len) in enumerate(bar):
        input = input.to(config.device)
        target = target.to(config.device)
        # Shift target to create decoder input: [SOS, token1, token2, ...]
        batch_size = target.size(0)
        sos_tokens = torch.full((batch_size, 1), config.target_ws.SOS, dtype=torch.long, device=config.device)
        tgt_input = torch.cat([sos_tokens, target[:, :-1]], dim=1)

        optimizer.zero_grad()
        decoder_outputs = model(input, tgt_input) #[batch_size, max_len, vocab_size]
        loss = F.nll_loss(decoder_outputs.view(-1,len(config.target_ws)),target.view(-1),ignore_index=config.input_ws.PAD)
        loss.backward()
        
        # 观察梯度大小（计算所有参数梯度的 L2 范数）
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        #添加上梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(),5)

        optimizer.step()
        loss_list.append(loss.item())
        bar.set_description("epoch:{} idx:{} loss:{:.6f} grad_norm:{:.4f}".format(epoch,idx,np.mean(loss_list), total_norm))

        if idx % 100 == 0:
            torch.save(model.state_dict(),"brain/models/model.pkl")
            torch.save(optimizer.state_dict(),"brain/models/optimizer.pkl")
            pickle.dump(loss_list,open("brain/models/loss_list.pkl","wb"))


if __name__ == '__main__':
    import os
    model = Seq2Seq().to(config.device)
    optimizer = Adam(model.parameters())
    loss_list = []

    model_path = "brain/models/model.pkl"
    optimizer_path = "brain/models/optimizer.pkl"
    loss_list_path = "brain/models/loss_list.pkl"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded existing model weights.")
    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path))
        print("Loaded existing optimizer state.")
    if os.path.exists(loss_list_path):
        loss_list = pickle.load(open(loss_list_path, "rb"))
        print(f"Loaded existing loss list. Start from epoch {len(loss_list) // 371}.")

    start_epoch = len(loss_list) // 371 if len(loss_list) > 0 else 0
    for i in range(start_epoch, 100): # Allow up to 100 epochs
        train(i, model, optimizer, loss_list)
        # eval()

    # plt.figure(figsize=(50,8))
    # plt.plot(range(len(loss_list)),loss_list)
    # plt.show()

