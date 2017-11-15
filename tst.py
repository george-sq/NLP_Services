# coding = utf-8
"""
    @File   : tst.py
    @Author : NLP_QiangShen
    @Time   : 2017/9/19 15:48
    @Todo   : 
"""

import jieba.posseg as pos
from nltk import pos_tag
import textAnalysis as ta
import getData as bd
import gensim as gs
from gensim import corpora, models, similarities
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
from pprint import pprint

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 解决matplotlib中文显示问题
mpl.rcParams['font.sans-serif'] = ['STKaiti']
mpl.rcParams['font.serif'] = ['STKaiti']

'''
squares = [1, 4, 9, 16, 25]
plt.plot(squares, linewidth=1)
# 设置图表标题
plt.title("样本空间(Sample Space)", fontsize=24)
# 给坐标轴加上标签
plt.xlabel("Value", fontsize=14)
plt.ylabel("Square of Value", fontsize=14)
# 设置刻度标记的大小
plt.tick_params(axis='both', labelsize=14)
plt.show()
'''

# 获取文本语料库（分词后）
""
rows = bd.LoadData().getRawCorpus()[1]
corpus = [[word for word in row[4].split()] for row in rows]

# 去除只出现过一次的词
frequency = defaultdict(int)
for text in corpus:  # 词频统计
    for word in text:
        frequency[word] += 1

txtList = [[word for word in text if frequency[word] > 1] for text in corpus]
txtList = [txtList[i] for i in range(len(txtList)) if i==599 or i==9999 or i==19999 or i==19999+1]
dictionary = corpora.Dictionary(txtList)
# dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference
print('dictionary : ', dictionary)
print(dictionary.__len__())

print('*****'*20)
print('dictionary.token2id : ', dictionary.token2id)

inData = [dictionary.doc2bow(txt) for txt in txtList]
tfidf = models.TfidfModel(inData) # step 1 -- initialize a model

corpus_tfidf = tfidf[inData]
for doc in corpus_tfidf:
    print(doc)

index = gs.similarities.SparseMatrixSimilarity(tfidf[inData[:2]], num_features=dictionary.__len__())# dictionary.__len__()
# sims = index[tfidf[inData[0]]]
# print(list(enumerate(sims)))
sims = index[tfidf[inData[2]]]
print(list(enumerate(sims)))
sims = index[tfidf[inData[3]]]
print(list(enumerate(sims)))


print()