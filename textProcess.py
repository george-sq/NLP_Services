# -*- coding: utf-8 -*-
"""
    @File   : textProcess.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/20 15:05
    @Todo   : 提供关于文本处理的服务
"""

import databaseServices as dbs
import fileServices as fs
import jieba
from gensim import corpora
import multiprocessing
from multiprocessing import Queue
from multiprocessing import Pool
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def getStopWords():
    stopWords = []
    stopWords_EN = fs.FileServer().loadLocalTextByUTF8('./StopWords/', 'stopWords_EN.txt')
    stopWords_CN = fs.FileServer().loadLocalTextByUTF8('./StopWords/', 'stopWords_CN.txt')
    stopWords.extend(stopWords_EN)
    stopWords.extend(stopWords_CN)
    stopWords.append(' ')
    return set(stopWords)


def doCutWord(qq, record):
    """
        :param record: [txtid,label,content,stopwords]
        :return:
    """
    retVal = []
    print("Queue size :", qq.qsize())
    gd = qq.get()
    print("Queue size :", qq.qsize())
    qq.put("doCutWord")
    print("Queue size :", qq.qsize())
    txtid = record[0]
    label = record[1]
    wordSeqs = jieba.cut(record[2])
    g_dicts.add_documents(wordSeqs)
    content = set([word for word in list(wordSeqs) if word not in record[3]])
    retVal.extend([txtid, label, list(content)])
    return retVal


g_dicts = corpora.Dictionary()


def main():
    # 初始化Mysql数据库连接
    mysqls = dbs.MysqlServer()
    mysqls.setConnect(user="pamo", passwd="pamo", db="textcorpus")

    # 获取原始语料库数据
    qs = mysqls.executeSql("SELECT * FROM tb_txtcate ORDER BY txtId")
    records = [[str(record[0]), record[2], record[3]] for record in qs[1:]]
    stopWords = getStopWords()
    for line in records:
        line[2] = line[2].replace('\r\n', '').replace('\n', '').replace(' ', '')
        line.append(stopWords)

    # 分词处理
    que = Queue()
    que.put(g_dicts)

    for line in records:
        p = multiprocessing.Process(target=doCutWord, args=(que, line))
        p.start()
        # print(que.get())
        p.join()
        # ds = doCutWord(line)
    pool = Pool(multiprocessing.cpu_count())
    dataSets = pool.imap(doCutWord, records)
    pool.close()
    pool.join()
    del records
    del stopWords


if __name__ == '__main__':
    main()
