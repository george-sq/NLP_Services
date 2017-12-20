# -*- coding: utf-8 -*-
"""
    @File   : textProcess.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/20 15:05
    @Todo   : 提供关于文本处理的服务
"""

import services_database as dbs
import services_fileIO as fs
import services_structdata as structdata
import jieba
import random
from gensim import corpora
import multiprocessing
from multiprocessing import Pool
import numpy as np
from scipy.sparse.csr import csr_matrix
import logging

jieba.setLogLevel(log_level=logging.INFO)
logger = logging.getLogger(__name__)


def getStopWords():
    stopWords = []
    stopWords_EN = fs.FileServer().loadTextByUTF8('./StopWords/', 'stopWords_EN.txt')
    stopWords_CN = fs.FileServer().loadTextByUTF8('./StopWords/', 'stopWords_CN.txt')
    stopWords.extend(stopWords_EN)
    stopWords.extend(stopWords_CN)
    stopWords.append(' ')
    return set(stopWords)


def doCutWord(record):
    """
        :param record: str
        :return:
    """
    retVal = []
    wordSeqs = jieba.cut(record.replace('\r\n', '').replace('\n', '').replace(' ', ''))
    retVal.extend(list(wordSeqs))
    return retVal


def splitTxt(docs=None):
    logger.info("对数据进行分词处理")
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    dataSets = pool.map(doCutWord, docs)
    pool.close()
    pool.join()
    logger.info("分词处理完成")
    return dataSets


def baseProcess():
    # 初始化Mysql数据库连接
    mysqls = dbs.MysqlServer()
    mysqls.setConnect(user="pamo", passwd="pamo", db="textcorpus")

    # 获取原始语料库数据
    result_query = mysqls.executeSql("SELECT * FROM tb_txtcate ORDER BY txtId")
    labels = [record[2] for record in result_query[1:]]
    txts = [record[3] for record in result_query[1:]]
    stopWords = getStopWords()

    # 分词处理
    dataSets = splitTxt(txts)

    # 数据标准化
    structDataHandler = structdata.BaseStructData()

    # 频率信息
    itermFreqs = structDataHandler.buildWordFrequencyDict(dataSets)
    freqData = []
    wordFreq = sorted(itermFreqs.items(), key=lambda twf: twf[1], reverse=True)
    for w, f in wordFreq:
        freqData.append(str(w) + '\t' + str(f) + '\n')

    # 语料库词典
    dicts4corpus = structDataHandler.buildGensimDict(dataSets)
    fileHandler = fs.FileServer()
    dicts4stopWords = structDataHandler.buildGensimDict([list(stopWords)])
    fileHandler.saveGensimDict(path="./Out/Dicts/", fileName="stopWords.dict", dicts=dicts4stopWords)

    # 去停用词
    for i in range(len(dataSets)):
        txt = dataSets[i]
        newTxt = []
        for j in range(len(txt)):
            word = txt[j]
            if word not in stopWords:
                newTxt.append(word)
        dataSets[i] = newTxt

    # 标准化语料库
    corpus = structDataHandler.buildGensimCorpusByCorporaDicts(dataSets, dicts4corpus)

    # 统计TFIDF数据
    statDataHandler = structdata.StatisticalData()
    tfidf4corpus = statDataHandler.buildGensimTFIDF(initCorpus=corpus, corpus=corpus)
    tfidfModel = statDataHandler.TFIDF_Vecs

    return labels, corpus, dicts4corpus, tfidfModel, tfidf4corpus, freqData


def storeData(path, fileName, **kwargs):
    fileHandler = fs.FileServer()
    lines = kwargs.get("lines", None)
    content = kwargs.get("content", None)
    dicts = kwargs.get("dicts", None)
    inCorpus = kwargs.get("inCorpus", None)
    # 本地存储
    if lines is not None and isinstance(lines, list):
        fileHandler.saveText2UTF8(path=path, fileName=fileName, lines=lines)
    elif content is not None and isinstance(content, str):
        fileHandler.saveText2UTF8(path=path, fileName=fileName, content=content)
    elif dicts is not None and isinstance(dicts, corpora.Dictionary):
        fileHandler.saveGensimDict(path=path, fileName=fileName, dicts=dicts)
    elif inCorpus is not None and isinstance(inCorpus, list):
        fileHandler.saveGensimCourpus2MM(path=path, fileName=fileName, inCorpus=inCorpus)
    else:
        print("缺少本地写入内容。")


def vecs2csrm(vecs, columns=None):
    """
        :param vecs: 需要转换的向量空间[[iterm,],]
        :param columns: 列的维度
        :return: csr_matrix
    """
    data = []  # 存放的是非0数据元素
    indices = []  # 存放的是data中元素对应行的列编号（列编号可重复）
    indptr = [0]  # 存放的是行偏移量

    for vec in vecs:  # 遍历数据集
        for colInd, colData in vec:  # 遍历单个数据集
            indices.append(colInd)
            data.append(colData)
        indptr.append(len(indices))

    if columns is not None:
        retCsrm = csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, columns), dtype=np.double)
    else:
        retCsrm = csr_matrix((data, indices, indptr), dtype=np.double)

    retCsrm.sort_indices()

    return retCsrm


def splitDataSet(labels, vectorSpace):
    """
        :param labels: [label,]
        :param vectorSpace:
        :return:
    """
    # 划分训练集和测试集
    trainSet = []
    testSet = []
    labelsA = []  # 非电信诈骗类型的索引集合
    labelsB = []  # 电信诈骗类型的索引集合
    for i in range(len(labels)):
        if "电信诈骗" != labels[i]:
            labelsA.append(i)
        else:
            labelsB.append(i)
    trainSet.extend(random.sample(labelsA, int(len(labelsA) * 0.9)))
    trainSet.extend(random.sample(labelsB, int(len(labelsB) * 0.9)))
    testSet.extend([index for index in labelsA if index not in trainSet])
    testSet.extend([index for index in labelsB if index not in trainSet])

    trainLabel = [labels[index] for index in trainSet]
    testLabel = [(index, labels[index]) for index in testSet]
    trainSet = [vectorSpace[index] for index in trainSet]
    testSet = [vectorSpace[index] for index in testSet]

    return [trainLabel, testLabel], [trainSet, testSet]


def main():
    # 预处理
    labels, corpus, dicts, tfidfModel, tfidfVecs, freqFile = baseProcess()
    storeData(path="./Out/StatFiles/", fileName="statFreqData.txt", lines=freqFile)
    storeData(path="./Out/Dicts/", fileName="corpusDicts.dict", dicts=dicts)
    storeData(path="./Out/Corpus/", fileName="corpus.mm", inCorpus=corpus)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
