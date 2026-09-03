"""
构造召回的模型
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from recall.bm25 import Bm25Vectorizer
from recall.fasttext_vectorizer import FasttextVectorizer
import pysparnn.cluster_index as ci
from utils import cut
import json

def prepar_recall_datas(method):
    qa_dict = json.load(open("./corpus/recall/qa_dict.json", encoding='utf-8'))
    q_list = []
    q_cut = []
    for i in qa_dict:
        q_list.append(i)
        q_cut.append(" ".join(qa_dict[i]["cut"])) #分词之后的问题 [sentence,sentence,....]
    if method == "tfidf":
        vectorizer = TfidfVectorizer()
    elif method == "bm25":
        vectorizer = Bm25Vectorizer()
    elif method == "fasttext":
        vectorizer = FasttextVectorizer()
    q_vector = vectorizer.fit_transform(q_cut) #得到问题的向量
    # print(q_vector)

    #准备搜索的索引
    cp = ci.MultiClusterIndex(q_vector,q_list)

    return vectorizer,cp,qa_dict


def get_search_result(input, method=None):
    tfidf_vec, cp, qa_dict = prepar_recall_datas(method)
    entity = []
    input_cut = []
    for word,seg in cut(input,by_word=False,use_seg=True):
        input_cut.append(word)
        if seg == "kc":
            entity.append(word)
    # 1. 得到用户问题的向量
    input_vector = tfidf_vec.transform([" ".join(input_cut)])
    #  2. 计算相似度
    result = cp.search(input_vector,k=10,k_clusters=10,return_distance=True)
    print(result)



def build_fasttext_model():
    """使用fasttext训练模型，得到词向量"""
    import fastText

    model = fastText.train_unsupervised("./corpus/recall/fasttxt_data.txt",wordNgrams=2,epoch=20)
    model.save_model("./recall/models/fasttext.model")



