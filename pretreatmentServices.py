# -*- coding: utf-8 -*-
"""
    @File   : pretreatmentServices.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/17 9:11
    @Todo   : 提供关于文本预处理的服务
"""

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora
from gensim import models
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF向量生成类
from collections import defaultdict


class BaseStructData(object):
    """数据预处理类"""

    def __init__(self):
        self.dictObj = None
        self.freDictObj = defaultdict(int)
        self.bunch4bow = None

    def buildWordFrequencyDict(self, dataSets):
        """ 生成数据集中最小单元的频率字典
            :param dataSets: 输入数据集 --> [[column0,column1,],]
            :return: self
        """
        for record in dataSets:
            for column in record:
                self.freDictObj[column] += 1
        return self.freDictObj

    def buildGensimDict(self, dataSets):
        """ 生成数据集的字典
            :param dataSets: 输入数据集 --> [[column0,column1,],]
            :return: self
        """
        self.dictObj = corpora.Dictionary(dataSets)
        return self.dictObj

    def buildBow2Bunch(self, wordSeqs=None):
        """ 生成Bunch对象的BOW
            :param wordSeqs: 词序列集合 --> [[column,],]
            :return:
        """
        if wordSeqs is not None:
            self.bunch4bow = Bunch(txtIds=[], classNames=[], labels=[], contents=[])
            for record in wordSeqs:
                if isinstance(record, list):
                    self.bunch4bow.contents.append(' '.join(record))
                else:
                    print("wordSeqs 的内容结构错误！wordSeqs --> [[column,],]")
        else:
            print("wordSeqs is None！请输入正确的参数 wordSeqs --> [[column,],]")
        return self.bunch4bow

    def buildGensimCorpus2MM(self, dataSets=None, dictObj=None):
        """ 生成语料库文件
            :param dataSets: 输入数据集 --> [[column0,column1,],]
            :param dictObj: Gensim字典对象 --> corpora.Dictionary
            :return: corpus --> [[(wordIndex,wordFreq),],]
        """
        corpus = None
        if isinstance(dictObj, corpora.Dictionary):
            corpus = [dictObj.doc2bow(record) for record in dataSets]
        elif isinstance(self.dictObj, corpora.Dictionary):
            corpus = [self.dictObj.doc2bow(record) for record in dataSets]
        else:
            print("非法的dictObj对象 (%s)，需要有效的 corpora.Dictionary对象！！！" % dictObj)
        return corpus


class StatisticalData(object):
    """统计数据生成类"""

    def __init__(self):
        self.TFIDF_Train_Vecs = None
        self.TFIDF_Test_Vecs = None
        self.TFIDF_Vecs = None

    def buildTFIDF2Train(self, bowObj=None, dictObj=None):
        """ 生成训练集的TFIDF向量空间（Bunch对象）
            :param bowObj: Bunch(txtIds=[], classNames=[], labels=[], contents=[[],])
            :param dictObj: Gensim字典对象 --> corpora.Dictionary
            :return: self.TFIDF_Train_Vecs
        """
        if isinstance(bowObj, Bunch):
            self.TFIDF_Train_Vecs = Bunch(txtIds=[], classNames=[], labels=[], tdm=[], vocabulary=[])
            self.TFIDF_Train_Vecs.txtIds.extend(bowObj.txtIds)
            self.TFIDF_Train_Vecs.classNames.extend(bowObj.classNames)
            self.TFIDF_Train_Vecs.labels.extend(bowObj.labels)
            if isinstance(dictObj, corpora.Dictionary):
                self.TFIDF_Train_Vecs.vocabulary = dictObj.token2id
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                         vocabulary=self.TFIDF_Train_Vecs.vocabulary)  # 将测试集文本映射到训练集词典中
            self.TFIDF_Train_Vecs.tdm = vectorizer.fit_transform(bowObj.contents)
        else:
            print("参数bowObj 类型错误 (%s)！请输入正确的bowObj参数。" % bowObj)
        return self.TFIDF_Train_Vecs

    def buildTFIDF2Test(self, bowObj=None, trainTfidfObj=None):
        """ 生成测试集的TFIDF向量空间（Bunch对象）
            :param bowObj: Bunch(txtIds=[], classNames=[], labels=[], contents=[])
            :param trainTfidfObj: Bunch(txtIds=[], classNames=[], labels=[], tdm=[], vocabulary=[])
            :return: self.TFIDF_Test_Vecs
        """
        if isinstance(bowObj, Bunch):
            if isinstance(trainTfidfObj, Bunch):
                self.TFIDF_Test_Vecs = Bunch(txtIds=[], classNames=[], labels=[], tdm=[], vocabulary=[])
                self.TFIDF_Test_Vecs.txtIds.extend(bowObj.txtIds)
                self.TFIDF_Test_Vecs.classNames.extend(bowObj.classNames)
                self.TFIDF_Test_Vecs.labels.extend(bowObj.labels)

                self.TFIDF_Test_Vecs.vocabulary = trainTfidfObj.vocabulary
                vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                             vocabulary=trainTfidfObj.vocabulary)  # 将测试集文本映射到训练集词典中
                self.TFIDF_Test_Vecs.tdm = vectorizer.fit_transform(bowObj.contents)
        else:
            print("参数bowObj 类型错误 (%s)！请输入正确的bowObj参数。" % bowObj)
        return self.TFIDF_Test_Vecs

    def buildGensimTFIDF(self, initCorpus=None, **kwargs):
        """ 生成数据集的TFIDF向量空间
            :param initCorpus: 初始化TFIDF向量工具模型的数据 --> [[doc2bow的处理结果(wordIndex,wordFreq),],]
            :param kwargs: record or corpus
            :return:
        """
        retVal = None
        record = kwargs.get("record", None)
        corpus = kwargs.get("corpus", None)
        if initCorpus is not None:
            self.TFIDF_Vecs = models.TfidfModel(initCorpus)
            if isinstance(record, list):
                retVal = self.TFIDF_Vecs[record]
            elif isinstance(corpus, list):
                retVal = self.TFIDF_Vecs[corpus]
            else:
                print("TFIDF向量空间生成失败。 (%s, %s)" % (record, corpus))
        else:
            print("参数initCorpus 类型错误 (%s)！请输入正确的initCorpus参数。" % initCorpus)

        return retVal


def main():
    pass


if __name__ == '__main__':
    main()
