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
from gensim import corpora
import multiprocessing
from multiprocessing import Pool
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


def main():
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
    del records
    del qs
    del mysqls
    del pool

    # 数据标准化
    structDataHandler = pts.BaseStructData()
    fileHandler = fs.FileServer()
    # 原始文本集
    rawCorpus = [record[2] for record in dataSets]
    # 频率信息
    itermFreqs = structDataHandler.buildWordFrequencyDict(rawCorpus)
    freqFile = []
    wordFreq = sorted(itermFreqs.items(), key=lambda twf: twf[1], reverse=True)
    for w, f in wordFreq:
        freqFile.append(str(w) + '\t' + str(f) + '\n')
    # 语料库词典
    dicts4corpus = structDataHandler.buildGensimDict(rawCorpus)
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
    fileHandler.saveGensimDict(path="./Out/Dicts/", fileName="corpusDicts.dict", dicts=dicts4corpus)
    fileHandler.saveGensimCourpus2MM(path="./Out/Corpus/", fileName="corpus.mm", inCorpus=corpus2MM)
    del stopWords


if __name__ == '__main__':
    main()
