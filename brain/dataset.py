"""
准备数据集
"""
import random
from tqdm import tqdm
import config
import torch
from torch.utils.data import DataLoader,Dataset



#1. 进行数据集的切分
def chatbot_data_split():
    f_train_input = open("./corpus/brain/train_input.txt","a")
    f_train_target = open("./corpus/brain/train_target.txt","a")
    f_test_input = open("./corpus/brain/test_input.txt","a")
    f_test_target = open("./corpus/brain/test_target.txt","a")
    input = open("./corpus/brain/input.txt").readlines()
    target = open("./corpus/brain/target.txt").readlines()
    for input,target in tqdm(zip(input,target),total=len(input)):
        if random.random()>0.8:
            #放入test
            f_test_input.write(input)
            f_test_target.write(target)
        else:
            f_train_input.write(input)
            f_train_target.write(target)
    f_train_input.close()
    f_train_target.close()
    f_test_input.close()
    f_test_target.close()



#2. 准备dataset

class ChatDataset(Dataset):
    def __init__(self,train=True):
        input_path = "./corpus/brain/train_input.txt" if train else "./corpus/brain/test_input.txt"
        target_path = "./corpus/brain/train_target.txt" if train else  "./corpus/brain/test_target.txt"
        self.input_data = open(input_path, encoding='utf-8').readlines()
        self.target_data = open(target_path, encoding='utf-8').readlines()
        assert len(self.input_data) == len(self.target_data),"input target长度不一致！！！"

    def __getitem__(self, idx):
        input = self.input_data[idx].strip().split()
        target = self.target_data[idx].strip().split()
        #获取真实长度
        input_len = len(input) if len(input)<config.chatbot_input_max_len else config.chatbot_input_max_len
        target_len = len(target) if len(target)<config.chatbot_target_max_len else config.chatbot_target_max_len

        input = config.input_ws.transform(input,max_len=config.chatbot_input_max_len)
        target = config.target_ws.transform(target,max_len=config.chatbot_target_max_len,add_eos=True)
        return input,target,input_len,target_len


    def __len__(self):
        return len(self.input_data)



#3. 准备dataloader
def collate_fn(batch):
    """
    :param batch:【（input,target,input_len,target_len），（），（一个getitem的结果）】
    :return:
    """
    #1. 对batch按照input的长度进行排序
    batch = sorted(batch,key=lambda x:x[-2],reverse=True)
    #2. 进行batch操作
    input, target, input_len, target_len = zip(*batch)
    #3. 把输入处理成LongTensor
    input = torch.LongTensor(input)
    target = torch.LongTensor(target)
    input_len = torch.LongTensor(input_len)
    target_len = torch.LongTensor(target_len)
    return input, target, input_len, target_len


def get_dataloader(train=True):
    batch_size = config.chatbot_train_batch_size if train else config.chatbot_test_batch_size
    return DataLoader(ChatDataset(train),batch_size=batch_size,collate_fn=collate_fn,shuffle=True)