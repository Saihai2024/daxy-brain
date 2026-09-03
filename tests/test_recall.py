import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根

from recall.build_models import get_search_result,build_fasttext_model
from recall.recall import Recall


def test_bm25():
    from recall.bm25 import Bm25Vectorizer
    data = [
        'hello world',
        'oh hello there',
        'Play it',
        'Play it again Sam',
    ]

    vec = Bm25Vectorizer()
    data_vector = vec.fit_transform(data)
    search_data = [
        # 'oh there',
        'Play it again Frank'
    ]
    ret = vec.transform(search_data)
    print(ret)


if __name__ == '__main__':
    # get_search_result("python是什么")
    ret = Recall(method="bm25").predict("python")
    print(ret)
    # ret = Recall(method="tfidf").predict("python是啥")
    # print(ret)
    # test_bm25()
    # build_fasttext_model()
    # ret = Recall(method="fasttext").predict("python是啥")
    # print(ret)
