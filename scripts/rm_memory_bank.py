import os
if os.path.exists("corpus/memory_bank/index.pt"):
    os.remove("corpus/memory_bank/index.pt")
if os.path.exists("corpus/memory_bank/metadata.json"):
    os.remove("corpus/memory_bank/metadata.json")
print("Removed old memory bank files!")
