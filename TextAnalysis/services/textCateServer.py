# -*- coding: utf-8 -*-
"""
    @File   : textCateServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/2/1 17:32
    @Todo   : 
"""

from bases.mysqlServer import MysqlServer
from bases.fileServer import FileServer
from basicTextProcessingServer import BasicTextProcessing


def getStopWords():
    stopWords = []
    stopWords_EN = FileServer().loadTextByUTF8('../../StopWords/', 'stopWords_EN.txt')
    stopWords_CN = FileServer().loadTextByUTF8('../../StopWords/', 'stopWords_CN.txt')
    stopWords.extend(stopWords_EN)
    stopWords.extend(stopWords_CN)
    stopWords.append(' ')
    return set(stopWords)


def baseProcess():
    # 初始化Mysql数据库连接
    mysqls = MysqlServer(host="10.0.0.247", db="db_pamodata", user="pamo", passwd="pamo")

    # 获取原始语料库数据
    result_query = mysqls.executeSql(sql="SELECT * FROM tb_txtcate ORDER BY txtId")
    labels = [record[2] for record in result_query[1:]]
    txts = [record[3] for record in result_query[1:]]
    stopWords = getStopWords()

    # 分词处理
    userDict = "../../Dicts/dict_jieba_check.txt"
    newWords = "../../Dicts/newWords.txt"
    btp = BasicTextProcessing(dict=userDict, words=newWords)
    dataSets = btp.batchWordSplit(contentList=txts)

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


def main():
    pass


if __name__ == '__main__':
    main()
