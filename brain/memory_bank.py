import torch
import os
import pickle
import torch.nn.functional as F

class MemoryBank:
    def __init__(self, path="brain/models/memory_bank.pkl", merge_threshold=0.90):
        self.path = path
        self.memory = [] # List of dicts: {"text": str, "vector": torch.Tensor, "count": int}
        self.merge_threshold = merge_threshold
        if os.path.exists(self.path):
            self.memory = pickle.load(open(self.path, "rb"))
            # Backwards compatibility
            for m in self.memory:
                if "count" not in m:
                    m["count"] = 1

    def add_memory(self, text, vector):
        vector = vector.cpu()
        if not self.memory:
            self.memory.append({"text": text, "vector": vector, "count": 1})
            return
            
        vectors = torch.stack([m["vector"] for m in self.memory])
        cos_sim = F.cosine_similarity(vector.unsqueeze(0), vectors)
        
        max_sim, max_idx = torch.max(cos_sim, dim=0)
        
        if max_sim.item() > self.merge_threshold:
            # Concept Clustering: Merge with existing memory
            idx = max_idx.item()
            existing_count = self.memory[idx]["count"]
            
            # Running average to fuse concepts
            new_vector = (self.memory[idx]["vector"] * existing_count + vector) / (existing_count + 1)
            self.memory[idx]["vector"] = new_vector
            self.memory[idx]["count"] += 1
            
            # Keep the shorter, more abstract text as the representative concept
            if len(text) < len(self.memory[idx]["text"]):
                self.memory[idx]["text"] = text
        else:
            self.memory.append({"text": text, "vector": vector, "count": 1})
        
    def save(self):
        pickle.dump(self.memory, open(self.path, "wb"))
        
    def retrieve(self, query_vector, top_k=1):
        if not self.memory:
            return []
            
        vectors = torch.stack([m["vector"] for m in self.memory])
        query_vector = query_vector.cpu()
        
        cos_sim = F.cosine_similarity(query_vector.unsqueeze(0), vectors)
        
        k = min(top_k, len(self.memory))
        top_scores, top_indices = torch.topk(cos_sim, k)
        
        results = []
        for score, idx in zip(top_scores, top_indices):
            results.append({
                "text": self.memory[idx.item()]["text"],
                "score": score.item(),
                "count": self.memory[idx.item()].get("count", 1)
            })
            
        return results
