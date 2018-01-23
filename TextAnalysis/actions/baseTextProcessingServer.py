# -*- coding: utf-8 -*-
"""
    @File   : baseTextProcessingServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/19 10:05
    @Todo   : 
"""

from mysqlServer import MysqlServer
import keyWordRecognitionServer as kws
import logging

logger = logging.getLogger(__name__)


def main():
    mysql = MysqlServer(host="10.0.0.247", db="db_pamodata", user="pamo", passwd="pamo")
    sql = "SELECT * FROM corpus_rawtxts WHERE txtLabel<>'电信诈骗' ORDER BY txtId LIMIT 10"
    queryResult = mysql.executeSql(sql=sql)
    queryResult = [(record[0], record[2], record[3]) for record in queryResult[1:]]
    retVal = []
    for tid, l, txt in queryResult:
        retVal.append(kws.fullMatch([tid, txt]))

    print(type(retVal))
    print(len(retVal))
    for i in retVal:
        print(i)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
