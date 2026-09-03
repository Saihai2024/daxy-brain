"""
准备模型
"""
import fasttext
import numpy as np


def get_data_path(by_word=True,train=True):
    if by_word:
        return "./corpus/classify/data_by_word_train.txt" if train else "./corpus/classify/data_by_word_test.txt"
    else:
        return "./corpus/classify/data_train.txt" if train else "./corpus/classify/data_test.txt"


def prepar_model():
    data_path = get_data_path(by_word=True,train=True)
    model = fasttext.train_supervised(data_path,dim=100,epoch=20,wordNgrams=2)
    # model.save_model("./classify/models/classify_by_word_100_20_1.model") # 0.992064558311166
    model.save_model("./classify/models/classify_by_word_100_20_2.model")  #0.9952888912772837
    # model.save_model("./classify/models/classify_100_20_2.model")  #0.9942397448645937
    # model.save_model("./classify/models/classify_100_20_1.model")  #0.9927616078547304

def test_model():
    model = fasttext.load_model("./classify/models/classify_by_word_100_20_2.model")
    test_data_path = get_data_path(by_word=True,train=False)

    sentences = []
    labels = []
    for line in open(test_data_path).readlines():
        line = line.strip()
        temp_ret = line.split("\t")
        if len(temp_ret)==2:
            sentences.append(temp_ret[0])
            labels.append(temp_ret[1])

    ret = model.predict(sentences)[0]
    ret = [i[0] for i in ret]
    assert len(labels) == len(ret),"labels的长度和ret的长度不一致"
    acc = np.mean([1 if labels[i] == ret[i] else 0 for i in range(len(labels))])
    print(acc)
