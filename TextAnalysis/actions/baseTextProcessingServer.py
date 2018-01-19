# -*- coding: utf-8 -*-
"""
    @File   : baseTextProcessingServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/19 10:05
    @Todo   : 
"""

from mysqlServer import MysqlServer
import jieba
from jieba import posseg
import logging

jieba.setLogLevel(logging.INFO)
jieba.enable_parallel(4)
logger = logging.getLogger(__name__)


def main():
    mysql = MysqlServer(host="10.0.0.247", db="db_pamodata", user="pamo", passwd="pamo")
    sql = "SELECT * FROM corpus_rawtxts WHERE txtLabel<>'电信诈骗' ORDER BY txtId LIMIT 10"
    retVal = mysql.executeSql(sql=sql)
    retVal = [(record[0], record[2], posseg.lcut(record[3])) for record in retVal[1:]]
    # retVal = [(line[3].strip(), line[0]) for line in retVal[1:]]
    # sql = "UPDATE corpus_rawtxts SET txtContent=%s WHERE txtId=%s"
    # retVal = mysql.executeSql(sql=sql, args=retVal)
    print(retVal)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
