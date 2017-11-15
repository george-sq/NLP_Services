# -*- coding: utf-8 -*-
"""
    @File   : 2.py.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/10/13 14:07
    @Todo   :
            2、数据分析
                    a、标准化输入输出
                        1)、输入数据标准
                        2)、输出数据标准
                    b、数据向量化
                        1)、构建数据词典
                        2)、输入数据词袋化
                        3)、词袋向量权重化
                    c、构建任务算法
                        1)、算法构建
                        2)、模型训练
                    d、算法性能评估
"""

import os
import time
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim import corpora
from gensim import models
import numpy as np
from wordcloud import WordCloud
import  matplotlib as plt
import multiprocessing
from multiprocessing import Pool
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


########################################################################################################################
def timeCost(func):
    def cost(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        useTime = time.time() - startTime
        logging.info('>>>>' * 30)
        logging.info('\t\t\t\t\t\t\t\t%s  [ %s ]  cost time  %s s' % (
            func.__class__, func.__name__, useTime))
        logging.info('<<<<' * 30)
        return result

    return cost


########################################################################################################################
def doMultiProcess(func, argsItr):
    pool = Pool(multiprocessing.cpu_count())
    results = pool.imap(func, argsItr)
    pool.close()
    pool.join()
    return results


def buildWordCloudWithFreq(dicts, imgName):
    """
        :param dicts: dict from string to float. {'word':freq,}
        :param imgName:
        :return:
    """
    # Generate a word cloud image
    wordcloud = WordCloud(max_words=2000, width=1300, height=600, background_color="white",
                          font_path='C:/Windows/Fonts/STSONG.TTF').generate_from_frequencies(
        dicts)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    wordcloud.to_file('./Out/' + imgName)


########################################################################################################################
def loadData():  # 加载预处理后的数据
    dictionary = corpora.Dictionary.load('./Out/Dicts/pamo_dicts.dict')
    corpus = corpora.MmCorpus('./Out/Corpus/pamo_gaCorpus.mm')
    return dictionary, corpus


def convertTfidf(dictionary=None, corpus=None):  # 将原始语料转换成tfidf向量空间
    if dictionary is not None:
        if corpus is not None:
            vecs_tfidf = models.TfidfModel(corpus)
            vecs_tfidf_corpus = vecs_tfidf[corpus]
            return vecs_tfidf_corpus
            # len_vec = len(dictionary)
            # zv = np.zeros(len_vec)
        else:
            print('语料库为空！！！')
    else:
        print('字典为空！！！')


########################################################################################################################
"""
    # 主函数
"""


def main():
    """
        2、数据分析          a、标准化输入输出    b、数据向量化    c、构建任务算法    d、算法性能评估
    """
    dictionary,corpus = loadData()

    vecs = convertTfidf(dictionary, corpus)




if __name__ == '__main__':
    main()
