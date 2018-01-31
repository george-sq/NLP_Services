# -*- coding: utf-8 -*-
"""
    @File   : basicTextProcessingServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/19 10:05
    @Todo   : 
"""

from mysqlServer import MysqlServer
import keyWordRecognitionServer as kws
import multiprocessing
import logging

logger = logging.getLogger(__name__)
import jieba
from jieba import posseg


def buildTaggedTxtCorpus():
    # 数据库连接
    mysql = MysqlServer(host="10.0.0.247", db="db_pamodata", user="pamo", passwd="pamo")
    # 查询结果
    sql = "SELECT * FROM corpus_rawtxts WHERE txtLabel<>'电信诈骗' ORDER BY txtId LIMIT 100"
    # sql = "SELECT * FROM corpus_rawtxts ORDER BY txtId"
    queryResult = mysql.executeSql(sql=sql)

    # 切分标注
    queryResult = [(record[0], record[2], record[3].replace(" ", "")) for record in queryResult[1:]]
    pool = multiprocessing.Pool(4)
    params = [[tid, txt] for tid, l, txt in queryResult]
    retVal = pool.map(kws.fullMatch, params)
    pool.close()
    pool.join()
    raw_root = "../../Out/文本分类语料库/"
    print(raw_root)
    csv_header = ["word", "pos", "ner"]
    csv_data = []
    for i in retVal:
        for a in i:
            if isinstance(a, list):
                for s in a:
                    print(s)
            else:
                print(a)
        print("##########" * 15)
    pass


class BasicTextProcessing(object):
    def __init__(self, **kwargs):
        jieba.setLogLevel(logging.INFO)
        userDict = kwargs.get("dict", None)
        newWords = kwargs.get("words", None)
        try:
            if userDict:
                jieba.set_dictionary(userDict)
                logger.info("Add custom dictionary successed.")
            if newWords:
                with open(newWords, "r", encoding="utf-8") as nw:
                    wordsSet = nw.readlines()
                    for line in wordsSet:
                        w, t = line.split()
                        jieba.add_word(word=w, tag=t.strip())
                        logger.info("Add word=%s(%s) freq : %s" % (w, t.strip(), str(jieba.suggest_freq(w))))
                    logger.info("Add user new words finished.")

        except Exception as e:
            logger.error("Error:%s" % e)
            logger.warning("Use custom dictionary failed, use default dictionary.")
        finally:
            self.__jieba = jieba
            self.__posseg = posseg

    def doWordSplit(self, content="", contents=()):
        retVal = []
        if content:
            retVal = self.__jieba.lcut(content)
        elif contents:
            for li in contents:
                retVal.append(self.__jieba.lcut(li))
        else:
            logger.warning("None content for splitting word")
        return retVal

    def batchWordSplit(self, contentList):
        retVal = []
        if contentList:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            retVal = pool.map(self.__jieba.lcut, contentList)
            pool.close()
            pool.join()
        else:
            logger.warning("None content for splitting word")
        return retVal


def main():
    # 数据库连接
    mysql = MysqlServer(host="10.0.0.247", db="db_pamodata", user="pamo", passwd="pamo")

    # 查询结果
    # sql = "SELECT * FROM corpus_rawtxts WHERE txtLabel<>'电信诈骗' ORDER BY txtId LIMIT 100"
    sql = "SELECT * FROM corpus_rawtxts ORDER BY txtId LIMIT 10"
    queryResult = mysql.executeSql(sql=sql)
    queryResult = [(record[0], record[2], record[3].replace(" ", "")) for record in queryResult[1:]]
    ids = [r[0] for r in queryResult]
    labels = [r[1] for r in queryResult]
    txts = [r[2] for r in queryResult]

    # 切分标注
    userDict = "../../Dicts/dict_jieba_check.txt"
    newWords = "../../Dicts/newWords.txt"

    # 功能类测试
    btp = BasicTextProcessing(dict=userDict, words=newWords)
    r = btp.doWordSplit(content=txts[0])
    print("*********" * 15)
    print(r)
    r = btp.doWordSplit(contents=txts)
    print("*********" * 15)
    for l in r:
        print(l)
        # r = btp.batchWordSplit(contentList=txts)
        # print("*********" * 15)
        # for l in r:
        #     print(l)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
