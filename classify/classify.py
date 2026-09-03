"""
提供接口
"""
import fasttext
import config

class Classify:
    def __init__(self):
        self.model = fasttext.load_model("classify/models/classify_100_20_2.model")

    def predict(self,sentence):
        """
        :param sentence: str ,分词后的句子
        :return:
        """

        ret = self.model.predict(sentence)  #(('__label__qa',), array([0.99764538]))
        predict_label = ret[0][0]  #预测结果
        predict_porb = ret[-1][0] #概率

        if predict_porb > config.predict_ratio:
            return predict_label