import torch
import torch.nn as nn
import math
import config
import numpy as np
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.d_model = config.transformer_d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(
            len(config.input_ws), 
            self.d_model, 
            padding_idx=config.input_ws.PAD
        )
        self.tgt_embedding = nn.Embedding(
            len(config.target_ws), 
            self.d_model, 
            padding_idx=config.target_ws.PAD
        )
        
        self.pos_encoder = PositionalEncoding(
            self.d_model, 
            dropout=config.transformer_dropout,
            max_len=max(config.chatbot_input_max_len, config.chatbot_target_max_len) + 10
        )
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=config.transformer_nhead,
            num_encoder_layers=config.transformer_num_layers,
            num_decoder_layers=config.transformer_num_layers,
            dim_feedforward=config.transformer_dim_feedforward,
            dropout=config.transformer_dropout,
            batch_first=True
        )
        
        # Output layer
        self.fc_out = nn.Linear(self.d_model, len(config.target_ws))

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(config.device)

    def create_padding_mask(self, seq, pad_idx):
        return (seq == pad_idx).to(config.device)

    def forward(self, src, tgt):
        """
        src: [batch_size, src_len]
        tgt: [batch_size, tgt_len] (shifted right, starts with SOS)
        """
        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
        
        src_padding_mask = self.create_padding_mask(src, config.input_ws.PAD)
        tgt_padding_mask = self.create_padding_mask(tgt, config.target_ws.PAD)
        
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        outs = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask, tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # apply log_softmax to be compatible with nll_loss
        return F.log_softmax(self.fc_out(outs), dim=-1) # [batch_size, tgt_len, vocab_size]

    def evaluate(self, src, src_len):
        """
        Greedy decoding
        src: [batch_size, src_len]
        """
        self.eval()
        batch_size = src.size(0)
        
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        src_padding_mask = self.create_padding_mask(src, config.input_ws.PAD)
        
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        # Start with SOS token
        ys = torch.ones(batch_size, 1).fill_(config.target_ws.SOS).type_as(src.data)
        
        for i in range(config.chatbot_target_max_len - 1):
            tgt_emb = self.pos_encoder(self.tgt_embedding(ys) * math.sqrt(self.d_model))
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1))
            
            out = self.transformer.decoder(
                tgt_emb, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask
            )
            
            prob = self.fc_out(out[:, -1, :])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.unsqueeze(1)
            
            ys = torch.cat([ys, next_word], dim=1)
            
        # Returning dummy decoder_outputs (to match expected interface) and prediction
        decoder_outputs = torch.zeros(batch_size, config.chatbot_target_max_len, len(config.target_ws)).to(config.device)
        predict_result = ys[:, 1:].cpu().numpy() # Strip SOS
        
        return decoder_outputs, predict_result
    
    def evaluate_with_beam_search(self, src, src_len):
        # Fallback to greedy for now
        _, ys = self.evaluate(src, src_len)
        return ys[0] # Return the first sequence
