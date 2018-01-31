# -*- coding: utf-8 -*-
"""
    @File   : baseTextProcessingServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/19 10:05
    @Todo   : 
"""

from mysqlServer import MysqlServer
import keyWordRecognitionServer as kws
import multiprocessing
import logging

logger = logging.getLogger(__name__)


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


def doWordSplit():
    pass


def main():
    # buildTaggedTxtCorpus()

    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
