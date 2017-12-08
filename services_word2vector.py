# -*- coding: utf-8 -*-
"""
    @File   : services_word2vector.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/12/8 14:32
    @Todo   : 
"""

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
import gensim
import multiprocessing
import services_database as dbs
import services_textProcess as tp
import services_fileIO as fs
from functools import reduce
from sklearn.datasets.base import Bunch
import jieba

jieba.setLogLevel(log_level=logging.INFO)
logger = logging.getLogger(__name__)


def buildWord2Vector(**kwargs):
    retVal = None
    corpus = kwargs.get("dataSet", None)
    if corpus is not None:
        logger.info("开始构建Word2Vector模型")
        model_w2v = gensim.models.Word2Vec(sentences=corpus, min_count=1, hs=1, workers=multiprocessing.cpu_count())
        fs.FileServer().saveWord2VectorModel(path="./Out/Word2Vector/", fileName="word2vector_all.w2v",
                                             wvmodel=model_w2v)
        logger.info("构建Word2Vector模型完成(%s)" % model_w2v)
        retVal = model_w2v
    else:
        logger.warning("参数错误（dataSet=%s），需要输入参数dataSet" % corpus)
    return retVal


def convertTxtVectorByWord2Vector(record=None):
    retVal = None
    if record is not None:
        wordSeqs = record[0]
        model_w2v = record[1]
        w2vs = []
        for word in wordSeqs:
            w2vs.append(model_w2v[word])
        retVal = reduce(lambda x, y: x + y, w2vs) / len(wordSeqs)
        logger.info("文本向量转化 +1")
    else:
        logger.warning("参数错误（record=%s）" % record)

    return retVal


def doTxtQuantizationByWord2Vector(**kwargs):
    retVal = None
    model_w2v = kwargs.get("model", None)
    corpus = kwargs.get("dataSet", None)
    if corpus is not None and isinstance(model_w2v, gensim.models.Word2Vec):
        dataSet = [(wordSeqs, model_w2v) for wordSeqs in corpus]
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        result_convertor = pool.map(convertTxtVectorByWord2Vector, dataSet)
        pool.close()
        pool.join()
        logger.info("文本向量转化完成, 转化总数: %s records" % len(result_convertor))
        retVal = result_convertor
    else:
        logger.warning("参数错误（corpus=%s , model_w2v=%s）" % (corpus, model_w2v))
    return retVal


def splitTxt(docs=None):
    logger.info("对数据进行分词处理")
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    dataSets = pool.map(tp.doCutWord, docs)
    pool.close()
    pool.join()
    logger.info("分词处理完成")
    return dataSets


def main():
    # 初始化Mysql数据库连接
    mysqls = dbs.MysqlServer()
    mysqls.setConnect(user="pamo", passwd="pamo", db="textcorpus")

    # 获取原始语料库数据
    result_query = mysqls.executeSql("SELECT * FROM tb_txtcate ORDER BY txtId")
    txts = [record[3] for record in result_query[1:]]
    logger.info("获得数据总数 %s records" % len(txts))

    # 分词处理
    dataSets = splitTxt(txts)

    # 原始文本集
    txtIds = [str(record[0]) for record in result_query[1:]]
    raw = Bunch(txtIds=txtIds, rawCorpus=dataSets)

    # 构建Word2Vector
    model_w2v = buildWord2Vector(dataSet=dataSets)

    # 文本数字化
    txtVecs = doTxtQuantizationByWord2Vector(dataSet=dataSets, model=model_w2v)
    pass


if __name__ == '__main__':
    # 创建一个handler，用于写入日志文件
    logfile = "./Logs/log_word2vector.log"
    fileLogger = logging.FileHandler(filename=logfile, encoding="utf-8")
    fileLogger.setLevel(logging.NOTSET)

    # 再创建一个handler，用于输出到控制台
    stdoutLogger = logging.StreamHandler()
    stdoutLogger.setLevel(logging.INFO)  # 输出到console的log等级的开关

    logging.basicConfig(level=logging.NOTSET,
                        format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S", handlers=[fileLogger, stdoutLogger])
    main()
