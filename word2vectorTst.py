# -*- coding: utf-8 -*-
"""
    @File   : word2vectorTst.py
    @Author : NLP_QingShen (275171387@qq.com)
    @Time   : 2017/9/22 8:56
    @Todo   : 
"""
import numpy as np
import jieba
import jieba.posseg as pos
import getData as bd
import gensim as gs
from gensim import corpora, models, similarities
from gensim.models.word2vec import LineSentence
import multiprocessing
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

txt1 = '五年来，习近平用文化自信提振“中国精气神”'
txt2 = '詹姆斯左脚踝扭伤缺训，被列入每日观察名单'
txt3 = '24日晚，由中国人民对外友好协会、中国戏剧家协会、江西省政府联合主办的汤显祖国际戏剧节暨国际戏剧交流月活动，在汤显祖故里江西抚州开幕。'
txt4 = '所谓餐厨垃圾，指的是餐饮服务单位在食品加工、饮食服务、单位供餐等活动中产生的食物残渣、食品加工废料和废弃食用油脂，也就是民间俗称的泔水。'
txts = [txt1, txt2, txt3, txt4]
for t in range(len(txts)):
    txts[t] = ' '.join(jieba.cut(txts[t]))
    print(txts[t])
txt1 = txts[0]
txt2 = txts[1]
txt3 = txts[2]
txt4 = txts[3]
stopWords = bd.LoadData().getStpwd()


# model = gs.models.Word2Vec(sentences=corpus,size=400,window=5,min_count=5,workers=multiprocessing.cpu_count()) # LineSentence()

def tokenization(txt):  # 去除停用词
    result = []
    for word in txt.split():
        if word not in stopWords:
            result.append(word)
    return result

# 夹角余弦距离公式
def cosdist(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

# 语料构建
corpus = [tokenization(txt1), tokenization(txt2), tokenization(txt3)]
print('**********' * 20)
for l in corpus:
    print('txt : ', l)
tst = tokenization(txt4)  ###################################################
print('----------' * 10)
print('tst : ', tst)
print('**********' * 20)
print()
# 构建词典
dictionary = corpora.Dictionary([tokenization(txt) for txt in txts])
print('**********' * 20)
print('dictionary : ')
print(dictionary.token2id)
print('**********' * 20)
print()
# 构建文本向量
all_bows_vectors = [dictionary.doc2bow(text) for text in [tokenization(txt) for txt in txts]]
txt_bows_vectors = [dictionary.doc2bow(text) for text in corpus]
tst_bow_vec = dictionary.doc2bow(tst)  ###################################################

def buildVec(ve,lens):
    newVec = np.zeros(lens)
    for ind,fre in ve:
        newVec[ind]=fre
    return newVec

print('**********' * 20)
print('夹角余弦距离计算 ： ')
a = buildVec(all_bows_vectors[0],dictionary.__len__())
b = buildVec(all_bows_vectors[1],dictionary.__len__())
print(cosdist(a,b))
print('**********' * 20)
print('**********' * 20)
for l in txt_bows_vectors:
    print('txt_bow : ', l)
print('----------' * 10)
print('tst_bow : ', tst_bow_vec)
print('**********' * 20)
print()
# ~~~~~~~~~~构建TF-IDF模型
tfidf = models.TfidfModel(corpus=txt_bows_vectors, dictionary=dictionary)  # !!!!
print('**********' * 20)
# print('tfidf model',tfidf)
all_vecs_tfidf = tfidf[all_bows_vectors]
txt_vecs_tfidf = tfidf[txt_bows_vectors]
tst_vec_tfidf = tfidf[tst_bow_vec]

for v in txt_vecs_tfidf:
    print("txt_tfidf : ")
    print(v)

print('----------' * 10)
print("tst_tfidf : ", tst_vec_tfidf)
print('**********' * 20)
print()
# 相似度分析
index = similarities.MatrixSimilarity(txt_vecs_tfidf)  # !!!!
# sims = index[tst_vec_tfidf]
print('**********' * 20)
# print('TF-IDF模型<相似度分析> :', list(enumerate(sims)))
print('**********' * 20)
print()

# ~~~~~~~~~~构建LSI模型
lsi = models.LsiModel(txt_vecs_tfidf, id2word=dictionary, num_topics=100)
lsi_vectors = lsi[txt_vecs_tfidf]
print('**********' * 20)
for vec in lsi_vectors:
    print('lsi_vector :', vec)
print('----------' * 10)
tst_lsi = lsi[tst_vec_tfidf]
print('tst_lsi :', tst_lsi)
print('**********' * 20)
print()
# print(list(lsi_vectors))
'''
a=list(lsi_vectors)
b=list(tst_lsi)
a.append(b)
all_lsi_vecs=a
x=[0,1,2,3,4]
y=[]
for yv in all_lsi_vecs:
    yy=[0]
    for lv in yv:
        yy.append(lv[1])
    y.append(yy)
'''
index = similarities.MatrixSimilarity(lsi_vectors)
sims = index[tst_lsi]
a = enumerate(sims)
aa = list(a)
print('**********' * 20)
print('LSI模型<相似度分析> :', list(enumerate(sims)))
print('**********' * 20)
print()
