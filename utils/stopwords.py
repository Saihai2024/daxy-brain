"""获取停用词"""

stopwords_path = r"corpus/stopwords.txt"

stopwords = [i.strip() for i in open(stopwords_path, encoding='utf-8').readlines()]