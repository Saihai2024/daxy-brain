"""
测试classify下的所有的api
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根

from classify.build_model import prepar_model,test_model
from classify.classify import Classify
if __name__ == '__main__':
    prepar_model()
    test_model()
    # classify = Classify()
    # classify.predict("python 是 什么")