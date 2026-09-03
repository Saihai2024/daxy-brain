"""
实现fasttext的向量化对象
"""
import fasttext

class FasttextVectorizer:
    def __init__(self):
        self.model = fasttext.load_model("./recall/models/fasttext.model")

    def fit_transform(self,inputs):
        """
        把分词后的句子转化为向量
        :param inputs:[input,input...."wo shi shui "]
        :return:
        """
        result = []
        for input in inputs:
            result.append(self.transform(input))
        return result

    def transform(self,input):
        """
        把单独的一个句子转化为向量
        :param input:
        :return:
        """
        if isinstance(input,list):
            assert len(input) == 1,"如果input是list，那么他的长度应该为1"
            input = input[0]
        return self.model.get_sentence_vector(input)
