# -*- coding: utf-8 -*-
"""
    @File   : textCateServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/2/1 17:32
    @Todo   : 
"""

from bases.mysqlServer import MysqlServer
from bases.fileServer import FileServer
from basicTextProcessingServer import BasicTextProcessing, TfidfVecSpace


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
    result_query = mysqls.executeSql(sql="SELECT * FROM corpus_rawtxts ORDER BY txtId")
    labels = [record[2] for record in result_query[1:]]
    txts = [record[3] for record in result_query[1:]]
    stopWords = getStopWords()

    # 生成文本预处理器
    userDict = "../../Dicts/dict_jieba_check.txt"
    newWords = "../../Dicts/newWords.txt"
    textHandler = BasicTextProcessing(dict=userDict, words=newWords)

    # 对原始语料库样本进行分词处理
    dataSets = textHandler.batchWordSplit(contentList=txts)

    # 生成原始语料库的语料库词典
    dicts4corpus = textHandler.buildGensimDict(dataSets)
    fileHandler = FileServer()
    dicts4stopWords = textHandler.buildGensimDict([list(stopWords)])
    fileHandler.saveGensimDict(path="./Out/Dicts/", fileName="stopWords.dict", dicts=dicts4stopWords)

    # 生成原始语料库的词频字典
    itermFreqs = textHandler.buildWordFrequencyDict(dataSets)
    freqData = []
    wordFreq = sorted(itermFreqs.items(), key=lambda twf: twf[1], reverse=True)
    for w, f in wordFreq:
        freqData.append(str(w) + '\t' + str(f) + '\n')

    # 对原始语料库进行去停用词处理
    for i in range(len(dataSets)):
        txt = dataSets[i]
        dataSets[i] = [txt[j] for j in range(len(txt)) if txt[j] not in stopWords]

    # 将原始语料库的样本进行数字化处理，生成数字化语料库
    corpus = textHandler.buildGensimCorpusByCorporaDicts(dataSets, dicts4corpus)

    # 根据数字化语料库生成TFIDF向量空间
    statDataHandler = TfidfVecSpace()
    tfidf4corpus = statDataHandler.buildVecsByGensim(initCorpus=corpus, corpus=corpus)
    tfidfModel = statDataHandler.TFIDF_Vecs

    return labels, corpus, dicts4corpus, tfidfModel, tfidf4corpus, freqData


def main():
    pass


if __name__ == '__main__':
    main()
