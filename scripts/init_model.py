import torch
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from brain.cognitive_engine import CognitiveEngine

ws = pickle.load(open("brain/models/ws_unified.pkl", "rb"))
config.input_ws = ws
model = CognitiveEngine(len(ws)).to(config.device)
torch.save(model.state_dict(), "brain/models/gpt_model.pkl")
print("Successfully initialized fresh CognitiveEngine with deep architecture!")
