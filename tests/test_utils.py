"""测试utils中的代码"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根

# from utils.cut_sentence import _cut_sentence_by_word,_cut_sentence,cut
import time
from utils import cut


if __name__ == '__main__':
    s = "天气python不错人工智能+python啊"
    # ret = _cut_sentence_by_word(s)
    # ret = _cut_sentence(s,use_seg=True,use_stopwords=True)
    # t1 = time.time()
    ret = cut(s,by_word=False,use_stopwords=True,use_seg=True)
    # t2 = time.time()
    print(ret,t2-t1)
    ret = cut(s,by_word=False,use_stopwords=True,use_seg=True)
    print(ret,time.time()-t2)