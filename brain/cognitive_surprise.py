import torch
import os
import pickle
import numpy as np

class CognitiveSurpriseEvaluator:
    def __init__(self, memory_bank, space_threshold=0.85, ppl_multiplier=1.3, max_ppl_floor=3.0):
        self.memory_bank = memory_bank
        self.space_threshold = space_threshold 
        self.ppl_multiplier = ppl_multiplier
        self.max_ppl_floor = max_ppl_floor
        
        self.baseline_loss = self._load_baseline_loss()
        self.dynamic_ppl_threshold = max(self.baseline_loss * self.ppl_multiplier, self.max_ppl_floor)
        
    def _load_baseline_loss(self):
        loss_list_path = "brain/models/gpt_loss_list.pkl"
        if os.path.exists(loss_list_path):
            try:
                loss_list = pickle.load(open(loss_list_path, "rb"))
                if len(loss_list) > 100:
                    return np.mean(loss_list[-100:])
                elif len(loss_list) > 0:
                    return np.mean(loss_list)
            except:
                pass
        return 4.0

    def evaluate_csi(self, query_vector, prompt_loss):
        if not self.memory_bank.memory:
            return True, 0.0, 1.0, self.dynamic_ppl_threshold
            
        retrieved = self.memory_bank.retrieve(query_vector, top_k=1)
        max_sim = retrieved[0]['score'] if retrieved else 0.0
        
        spatial_surprise = max(0.0, 1.0 - max_sim)
        generative_surprise = min(1.0, prompt_loss / (self.dynamic_ppl_threshold * 1.5))
        
        total_csi_score = 0.4 * spatial_surprise + 0.6 * generative_surprise
        
        is_surprised = (max_sim < self.space_threshold and prompt_loss > self.dynamic_ppl_threshold) or (total_csi_score > 0.8)
        
        return is_surprised, max_sim, total_csi_score, self.dynamic_ppl_threshold
