# -*- coding: utf-8 -*-
"""
    @File   : vectorSpace.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/9/22 10:05
    @Todo   : 
"""

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import jieba.posseg as pos
import gensim
from gensim import corpora, models, similarities


# 停用词处理，中文停用词 and 英文停用词
def getStopWords():  # step-0
    """
        :return: list => [stopWord,...]
    """
    return bd.LoadData().getStpwd()


# 构建语料词典
def buildDict(docList):  # step-1
    """
        :param docList: 文本列表 => [text,...]
        :return: Dictionary
    """
    return gensim.corpora.Dictionary(docList)


# 词包模型
def buildBow(wordList):  # step-2
    """
        :param wordList: 文本词列表 => [word,...]
        :return: 词包模型 => [(词典索引，词频),...]
    """
    return gensim.corpora.Dictionary.doc2bow(wordList)


# TF-IDF向量
def buildTFIDF(bows):  # step-3.1
    """
        :param bows: 文本词包模型列表 => [bow,...]
        :return: tfidf model对象
        Return tf-idf representation of the input vector and/or corpus.
    """
    tfidf = gensim.models.TfidfModel(bows)
    # tfidf_vectors = tfidf[bows]
    return tfidf


def getTFIDFVectors(tfidf, someSample):  # step-3.2
    """
        :param tfidf: tfidf model对象
        :param someSample: bow or corpus=[bow,...].
        :return:
    """
    if isinstance(tfidf, gensim.models.TfidfModel):
        return tfidf[someSample]
    else:
        print(tfidf, "参数类型错误。")
        exit(1)


# Latent Semantic Indexing\Analysis,(LSI or LSA)浅层语义分析模型
def buildLsiModel(tfidf_vectors, dictionary, num):  # step-4
    """
        :param tfidf_vectors: 文档的tfidf向量
        :param dictionary: 文档字典
        :param num: num_topics
        :return: lsi模型向量
    """
    lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=num)
    lsi_vectors = lsi[tfidf_vectors]
    return lsi_vectors

def getSimilarities(lsi_vectors,tst_lsi):
    """
        :param lsi_vectors: lsi模型向量
        :param tst_lsi: lsi模型向量
        :return: 相似度列表
    """
    # tst_lsi = lsi[tst_vec]
    print('tst_lsi :', tst_lsi)
    index = similarities.MatrixSimilarity(lsi_vectors)
    sims = index[tst_lsi]
    return sims

if __name__ == '__main__':
    pass
