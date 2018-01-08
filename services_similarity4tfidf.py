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
import services_ner as ner
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


def getRawCorpus(sql=""):
    # 初始化Mysql数据库连接
    mysqls = dbs.MysqlServer()

    # 获取原始语料库数据
    if len(sql) > 0:
        result_query = mysqls.executeSql(sql)
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


def buildTfidfModel(sql="", path="", mname=""):
    # 预处理
    if len(sql) > 0:
        raw, corpus, dicts, tfidfModel = getRawCorpus(sql=sql)
        tfidfVecs = tfidfModel[corpus]
        num_features = len(dicts)
        fileHandler = fs.FileServer()

        # tfidf相似性
        indexTfidf = gensim.similarities.SparseMatrixSimilarity(tfidfVecs, num_features=num_features)
        similarityModel = Bunch(txtIds=raw.txtIds, dicts=dicts, tfidfModel=tfidfModel, indexTfidf=indexTfidf)

        if len(path) > 0 and len(mname) > 0:
            fileHandler.savePickledObjFile(path=path, fileName=mname, writeContentObj=similarityModel)


def tfidfSimilartyProcess(queryTxt, sql="", path="", mname=""):
    if len(path) > 0 and len(mname) > 0:
        # 加载数据
        fileHandler = fs.FileServer()
        similarityModel = fileHandler.loadPickledObjFile(path=path, fileName=mname)
        txtIds = similarityModel.txtIds
        dicts = similarityModel.dicts
        tfidfModel = similarityModel.tfidfModel
        indexTfidf = similarityModel.indexTfidf

        # 数字向量化
        bow_query = dicts.doc2bow(list(jieba.cut(queryTxt)))
        tfidf_query = tfidfModel[bow_query]

        # tfidf相似性
        sim_tfidf_query = indexTfidf[tfidf_query]
        results = sorted(enumerate(sim_tfidf_query), key=lambda item: -item[1])[:10]
        results = [(txtIds[index], freq) for index, freq in results]

        print("query tfidf相似性：")
        # print(results)
        sim_results = []
        if len(sql) > 0:
            for r in results:
                q = dbs.MysqlServer().executeSql(sql % r[0])
                print(q[1:][0][:2])
                sim_results.append(q[1:][0][:2])
        # results = [int(item[0]) for item in results]
        # print(results)
        return sim_results


def main():
    queryTxt = "微信上买手机，转账被骗3100 现在此地，请妥处 嫌疑人微信号qq421149709"

    # 生成数据
    # sql = "SELECT * FROM tb_ajinfo ORDER BY tId"
    path = "./Out/"
    mname = "model_tfidfSimilarity_anjian.pickle"
    # buildTfidfModel(sql=sql, path=path, mname=mname)

    # 相似度分析
    sql = "SELECT * FROM tb_ajinfo WHERE tid=%s"
    res_sim = tfidfSimilartyProcess(queryTxt, sql=sql, path=path, mname=mname)
    results = []
    for tid, txt in res_sim:
        index, content = ner.fullMatch((tid, txt))
        content = "\t".join([" | ".join(wp) for wp in content])
        results.append((index, content))

    print()

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
