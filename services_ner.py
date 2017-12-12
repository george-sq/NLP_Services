# -*- coding: utf-8 -*-
"""
    @File   : services_ner.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/12/11 10:17
    @Todo   : 
"""

import logging
import multiprocessing
import services_database as dbs
import jieba
from jieba import posseg

jieba.setLogLevel(log_level=logging.INFO)
logger = logging.getLogger(__name__)


def doCutPos(record):
    """
        :param record: str
        :return:
    """
    retVal = []
    wordSeqs = posseg.cut(record.replace('\r\n', '').replace('\n', '').replace(' ', ''))
    retVal.extend(list(wordSeqs))
    return retVal


def splitTxt(docs=None):
    logger.info("对数据进行分词处理")
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    dataSets = pool.map(doCutPos, docs)
    pool.close()
    pool.join()
    logger.info("分词处理完成")
    return dataSets


def main():
    # # 初始化Mysql数据库连接
    # mysqls = dbs.MysqlServer()
    # mysqls.setConnect(user="pamo", passwd="pamo", db="textcorpus")
    #
    # # 获取原始语料库数据
    # result_query = mysqls.executeSql("SELECT * FROM tb_txtcate WHERE txtLabel='电信诈骗' ORDER BY txtId")
    # txts = [record[3] for record in result_query[1:]]
    # logger.info("获得案件记录数据,记录总数 %s records" % len(txts))
    #
    # # 分词处理
    # dataSets = splitTxt(txts)
    #
    # # 原始文本集
    # txtIds = [str(record[0]) for record in result_query[1:]]
    txt = "在网上购物对方让其退款点击链接发现损失1904元，半小时前操作，对方手机号：13566018281，己方帐号6217000060006070435，请妥善处置（电信诈骗已骗成）"
    result = doCutPos()
    pass


if __name__ == '__main__':
    # 创建一个handler，用于写入日志文件
    logfile = "./Logs/log_ner.log"
    fileLogger = logging.FileHandler(filename=logfile, encoding="utf-8")
    fileLogger.setLevel(logging.NOTSET)

    # 再创建一个handler，用于输出到控制台
    stdoutLogger = logging.StreamHandler()
    stdoutLogger.setLevel(logging.INFO)  # 输出到console的log等级的开关

    logging.basicConfig(level=logging.NOTSET,
                        format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S", handlers=[fileLogger, stdoutLogger])
    main()
