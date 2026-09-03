import torch
from torch.utils.data import DataLoader, Dataset
import config
from brain.word_sequece import WordSequence
import pickle

class GPTDataset(Dataset):
    def __init__(self, train=True):
        input_path = "./corpus/brain/train_input.txt" if train else "./corpus/brain/test_input.txt"
        target_path = "./corpus/brain/train_target.txt" if train else "./corpus/brain/test_target.txt"
        self.input_data = open(input_path, encoding='utf-8').readlines()
        self.target_data = open(target_path, encoding='utf-8').readlines()
        assert len(self.input_data) == len(self.target_data), "Length mismatch"
        
        # Load unified vocab
        self.ws = pickle.load(open("brain/models/ws_unified.pkl", "rb"))
        self.max_len = 100 # Adjust as needed

    def __getitem__(self, idx):
        # Concatenate Input + <SEP> + Target + <EOS>
        input_words = self.input_data[idx].strip().split()
        target_words = self.target_data[idx].strip().split()
        
        # Instead of <SEP>, we can just use <SOS> then input then target then <EOS>
        # Let's define the sequence as: [SOS, input_1... input_n, EOS, target_1... target_m, EOS]
        # Or just [input_1...input_n, SEP, target_1...target_m, EOS]
        # To keep it simple with existing tokens:
        sentence = input_words + [self.ws.EOS_TAG] + target_words
        
        # Transform using the unified vocab
        indices = self.ws.transform(sentence, max_len=self.max_len, add_eos=True)
        return torch.LongTensor(indices)

    def __len__(self):
        return len(self.input_data)

def collate_fn_gpt(batch):
    # batch is a list of tensors
    # We want to return inputs (all but last token) and targets (all but first token)
    batch_tensor = torch.stack(batch)
    inputs = batch_tensor[:, :-1]
    targets = batch_tensor[:, 1:]
    return inputs, targets

def get_gpt_dataloader(train=True):
    batch_size = 128 # GPT batches usually need to be smaller if sequences are longer
    return DataLoader(GPTDataset(train), batch_size=batch_size, collate_fn=collate_fn_gpt, shuffle=True)
