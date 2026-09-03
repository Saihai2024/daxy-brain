"""
实现召回的相关的逻辑
"""
from recall.build_models import prepar_recall_datas
from utils import cut

class Recall:
    def __init__(self,method="tfidf"):
        self.method = method
        assert  self.method in ["tfidf","bm25","fasttext"],"method方法不合适"
        self.vectorizer,self.cp,self.qa_dict=prepar_recall_datas(method)

    def predict(self,input):
        """
        使用tfidf进行召回
        :param input: 用户输入的句子
        :return:
        """
        entity = []
        input_cut = []
        #TODO 以下操作可以提前在外层实现
        for word, seg in cut(input, by_word=False, use_seg=True):
            input_cut.append(word)
            if seg == "kc":
                entity.append(word)
        # 1. 得到用户问题的向量
        # print("input cut:",input_cut)
        input_vector = self.vectorizer.transform([" ".join(input_cut)])
        # print("user input_vector:\n",input_vector)
        # print("*"*20)
        #  2. 计算相似度
        result = self.cp.search(input_vector, k=10, k_clusters=10, return_distance=False)[0]
        # 3. 对结果通过主体过滤；用户问题没有 kc 实体时退化为 top-k
        if not entity:
            return list(result)
        final_result = []
        for temp_ret in result:
            ret_entity = self.qa_dict[temp_ret]["entity"]
            if len(set(entity) & set(ret_entity)) > 0:
                final_result.append(temp_ret)
        return final_result if final_result else list(result)