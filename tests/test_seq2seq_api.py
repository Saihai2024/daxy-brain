"""
测试chatbot中所有的api
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根


from brain.word_sequece import WordSequence
from brain.dataset import get_dataloader
import torch
import pickle
from tqdm import tqdm
from brain.train import train
from brain.eval import eval,interface,interface_with_beamsearch
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_ws():
    ws = WordSequence()
    s = [["我","是","谁"],["你","好",'么']]
    for i in s:
        ws.fit(i)
    ws.build_vocab(min_count=1)
    print(ws.dict)
    ret = ws.transform(["你","好"],max_len=4,add_eos=False)
    print(ret)
    ret = ws.inverse_transform(ret)
    print(ret)


def prepar_ws():
    ws = WordSequence()
    for line in tqdm(open("./corpus/brain/input.txt", encoding='utf-8').readlines()):
        line = line.strip().split()
        ws.fit(line)
    ws.build_vocab()
    print(len(ws))  #3509
    pickle.dump(ws,open("./brain/models/ws_input.pkl","wb"))

    ws = WordSequence()
    for line in tqdm(open("./corpus/brain/target.txt", encoding='utf-8').readlines()):
        line = line.strip().split()
        ws.fit(line)
    ws.build_vocab()
    print(len(ws))  #4365
    pickle.dump(ws, open("./brain/models/ws_target.pkl","wb"))


def test_dataset():
    from brain.dataset import chatbot_data_split,ChatDataset
    # from brain.dataset import ChatDataset
    #train 296239 test:74440
    d_train = ChatDataset(train=False)
    input, target, input_len, target_len = d_train[100]
    # d_test = ChatDataset(train=False)
    # print(len(d_train))
    # print(len(d_test))
    print(target)
    # chatbot_data_split()


def test_dataloader():
    loader = get_dataloader()
    for idx,(input, target, input_len, target_len) in enumerate(loader):
        print(idx)
        print(input)
        print(target)
        print(input_len)
        print(target_len)
        break


def train_chatbot(): #进行模型的训练
    from brain.seq2seq import Seq2Seq
    from torch.optim import Adam
    from matplotlib import pyplot as plt
    import config
    model = Seq2Seq().to(config.device)
    optimizer = Adam(model.parameters())

    model_path = "brain/models/model.pkl"
    optimizer_path = "brain/models/optimizer.pkl"
    loss_list_path = "brain/models/loss_list.pkl"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        torch.save(model.state_dict(), model_path)  # 保存初始模型状态

    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path))
    else:
        torch.save(optimizer.state_dict(), optimizer_path)  # 保存初始优化器状态

    if os.path.exists(loss_list_path):
        loss_list = pickle.load(open(loss_list_path, "rb"))
    else:
        loss_list = []
        pickle.dump(loss_list, open(loss_list_path, "wb"))  # 保存初始损失列表

    # # 自定义初始化参数
    for name, param in model.named_parameters():
       if 'bias' in name:
           torch.nn.init.constant_(param, 0.0)
       elif 'weight' in name:
           torch.nn.init.xavier_normal_(param)

    for i in range(10000):
        train(i, model, optimizer, loss_list)
        eval()
    plt.figure(figsize=(50, 8))
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()


if __name__ == '__main__':
    prepar_ws()
    # test_dataset()
    # test_dataloader()
    # train_chatbot()
    # eval()
    # interface()
    # interface_with_beamsearch()
    # from brain.seq2seq import Seq2Seq
    # model = Seq2Seq()
    # for name, param in model.named_parameters():
    #     print(name, param.size(), "*"*30)









