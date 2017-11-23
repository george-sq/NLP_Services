# -*- coding: utf-8 -*-
"""
    @File   : textProcess.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/20 15:05
    @Todo   : 提供关于文本处理的服务
"""

import databaseServices as dbs
import fileServices as fs
import pretreatmentServices as pts
import jieba
import multiprocessing
from multiprocessing import Pool
import numpy as np
from scipy.sparse.csr import csr_matrix
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
jieba.setLogLevel(log_level=logging.INFO)


def getStopWords():
    stopWords = []
    stopWords_EN = fs.FileServer().loadLocalTextByUTF8('./StopWords/', 'stopWords_EN.txt')
    stopWords_CN = fs.FileServer().loadLocalTextByUTF8('./StopWords/', 'stopWords_CN.txt')
    stopWords.extend(stopWords_EN)
    stopWords.extend(stopWords_CN)
    stopWords.append(' ')
    return set(stopWords)


def doCutWord(record):
    """
        :param record: [txtid,label,content]
        :return:
    """
    retVal = []
    txtid = record[0]
    label = record[1]
    wordSeqs = jieba.cut(record[2].replace('\r\n', '').replace('\n', '').replace(' ', ''))
    # content = set([word for word in list(wordSeqs) if word not in record[3]])
    retVal.extend([txtid, label, list(wordSeqs)])
    return retVal


def baseProcess():
    # 初始化Mysql数据库连接
    mysqls = dbs.MysqlServer()
    mysqls.setConnect(user="pamo", passwd="pamo", db="textcorpus")

    # 获取原始语料库数据
    qs = mysqls.executeSql("SELECT * FROM tb_txtcate ORDER BY txtId")
    records = [[str(record[0]), record[2], record[3]] for record in qs[1:]]
    stopWords = getStopWords()

    # 分词处理
    pool = Pool(multiprocessing.cpu_count())
    dataSets = pool.map(doCutWord, records)
    pool.close()
    pool.join()

    # 数据标准化
    structDataHandler = pts.BaseStructData()
    fileHandler = fs.FileServer()
    # 原始文本集
    rawCorpus = [record[2] for record in dataSets]
    labels = [record[1] for record in dataSets]
    # 频率信息
    itermFreqs = structDataHandler.buildWordFrequencyDict(rawCorpus)
    freqFile = []
    wordFreq = sorted(itermFreqs.items(), key=lambda twf: twf[1], reverse=True)
    for w, f in wordFreq:
        freqFile.append(str(w) + '\t' + str(f) + '\n')
    # 语料库词典
    dicts4corpus = structDataHandler.buildGensimDict(rawCorpus)
    dicts4stopWords = structDataHandler.buildGensimDict([list(stopWords)])
    # 去停用词
    for i in range(len(rawCorpus)):
        txt = rawCorpus[i]
        newTxt = []
        for j in range(len(txt)):
            word = txt[j]
            if word not in stopWords:
                newTxt.append(word)
        rawCorpus[i] = newTxt
    # 标准化语料库
    corpus2MM = structDataHandler.buildGensimCorpus2MM(rawCorpus, dicts4corpus)
    # 本地存储
    fileHandler.saveText2UTF8(path="./Out/StatFiles/", fileName="statFreqData.txt", lines=freqFile)
    fileHandler.saveGensimDict(path="./Out/Dicts/", fileName="stopWords.dict", dicts=dicts4stopWords)
    fileHandler.saveGensimDict(path="./Out/Dicts/", fileName="corpusDicts.dict", dicts=dicts4corpus)
    fileHandler.saveGensimCourpus2MM(path="./Out/Corpus/", fileName="corpus.mm", inCorpus=corpus2MM)

    # 统计TFIDF数据
    statDataHandler = pts.StatisticalData()
    tfidf4corpus = statDataHandler.buildGensimTFIDF(initCorpus=corpus2MM, corpus=corpus2MM)
    tfidf4corpus = list(tfidf4corpus)
    return dicts4corpus, labels, tfidf4corpus


def vecs2csrm(vecs):
    data = []  # 存放的是非0数据元素
    indices = []  # 存放的是data中元素对应行的列编号（列编号可重复）
    indptr = [0]  # 存放的是行偏移量

    for vec in vecs:  # 遍历数据集
        for colInd, colData in vec:  # 遍历单个数据集
            indices.append(colInd)
            data.append(colData)
        indptr.append(len(indices))

    retCsrm = csr_matrix((data, indices, indptr), dtype=np.double)
    retCsrm.sort_indices()

    return retCsrm


def main():
    dicts, labels, tfidfVecs = baseProcess()
    csrm_tfidf = vecs2csrm(tfidfVecs)


if __name__ == '__main__':
    main()