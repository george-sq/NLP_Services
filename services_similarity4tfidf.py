# -*- coding: utf-8 -*-
"""
    @File   : services_similarity4tfidf.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/12/6 11:26
    @Todo   : 
"""

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
import gensim
import services_database as dbs
import services_structdata as structdata
import services_textProcess as tp
import services_fileIO as fs
import multiprocessing
from sklearn.datasets.base import Bunch
import jieba

jieba.setLogLevel(log_level=logging.INFO)
logger = logging.getLogger(__name__)


def splitTxt(docs=None):
    logger.info("对数据进行分词处理")
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    dataSets = pool.map(tp.doCutWord, docs)
    pool.close()
    pool.join()
    logger.info("分词处理完成")
    return dataSets


def getRawCorpus():
    # 初始化Mysql数据库连接
    mysqls = dbs.MysqlServer()
    # mysqls.setConnect(user="pamo", passwd="pamo", db="textcorpus")

    # 获取原始语料库数据
    result_query = mysqls.executeSql("SELECT * FROM tb_tinfo ORDER BY tId")
    txts = []
    txtIds = []
    for record in result_query[1:]:
        txtIds.append(str(record[0]))
        txts.append(record[1])
    logger.info("获得案件记录数据,记录总数 %s records" % len(txts))

    # 分词处理
    dataSets = splitTxt(txts)

    # 原始文本集
    raw = Bunch(txtIds=txtIds, rawCorpus=dataSets)

    # 数据标准化
    structDataHandler = structdata.BaseStructData()
    dicts4corpus = structDataHandler.buildGensimDict(dataSets)

    # 标准化语料库
    corpus = structDataHandler.buildGensimCorpusByCorporaDicts(dataSets, dicts4corpus)

    # 统计TFIDF数据
    statDataHandler = structdata.StatisticalData()
    model_tfidf = statDataHandler.buildGensimTFIDF(initCorpus=corpus)

    return raw, corpus, dicts4corpus, model_tfidf


def main():
    # 预处理
    raw, corpus, dicts, tfidfModel = getRawCorpus()
    tfidfVecs = tfidfModel[corpus]
    num_features = len(dicts)
    fileHandler = fs.FileServer()
    fileHandler.saveGensimDict(path="./Out/Dicts/", fileName="dict_anjian.dict", dicts=dicts)
    fileHandler.saveGensimTfidfModel(path="./Out/Models/", fileName="model_TfidfModel_anjian.mdl", tfidf=tfidfModel)
    fileHandler.saveGensimCourpus2MM(path="./Out/Corpus/", fileName="corpus_anjian.mm", inCorpus=corpus)
    fileHandler.savePickledObjFile(path="./Out/", fileName="raw_anjian.dat", writeContentObj=raw)

    # tfidf相似性
    indexTfidf = gensim.similarities.SparseMatrixSimilarity(tfidfVecs, num_features=num_features)
    fileHandler.saveIndex4tfidfSimilarity(path="./Out/Indexs/", fileName="index_TfidfModel_anjian.idx",
                                          index=indexTfidf)

    queryTxt = "微信上买手机，转账被骗3100 现在此地，请妥处 嫌疑人微信号qq421149709 "
    bow_query = dicts.doc2bow(list(jieba.cut(queryTxt)))

    # 数字向量化
    tfidf_query = tfidfModel[bow_query]

    # tfidf相似性
    sim_tfidf_query = indexTfidf[tfidf_query]
    results = sorted(enumerate(sim_tfidf_query), key=lambda item: -item[1])[:5]
    print(results)
    txtIds = raw.txtIds
    results = [(txtIds[index], freq) for index, freq in results]
    print("query tfidf相似性：", results)


if __name__ == '__main__':
    # 创建一个handler，用于写入日志文件
    logfile = "./Logs/log_similarity4tfidf.log"
    fileLogger = logging.FileHandler(filename=logfile, encoding="utf-8")
    fileLogger.setLevel(logging.NOTSET)

    # 再创建一个handler，用于输出到控制台
    stdoutLogger = logging.StreamHandler()
    stdoutLogger.setLevel(logging.INFO)  # 输出到console的log等级的开关

    logging.basicConfig(level=logging.NOTSET,
                        format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S", handlers=[fileLogger, stdoutLogger])
    main()
