import os
import pickle
from brain.word_sequece import WordSequence
from tqdm import tqdm

def build_unified_vocab():
    print("Building unified vocabulary...")
    ws = WordSequence()
    
    # Read both inputs and targets
    files = ["./corpus/brain/train_input.txt", "./corpus/brain/train_target.txt"]
    for file in files:
        if os.path.exists(file):
            print(f"Reading {file}")
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in tqdm(lines, desc=f"Processing {os.path.basename(file)}"):
                    sentence = line.strip().split()
                    ws.fit(sentence)
        else:
            print(f"Warning: {file} not found.")

    # Build the vocabulary
    ws.build_vocab(min_count=2, max_features=50000)
    print(f"Unified vocabulary size: {len(ws)}")

    # Save to file
    os.makedirs("brain/models", exist_ok=True)
    with open("brain/models/ws_unified.pkl", "wb") as f:
        pickle.dump(ws, f)
    print("Saved unified vocabulary to brain/models/ws_unified.pkl")

if __name__ == "__main__":
    build_unified_vocab()
