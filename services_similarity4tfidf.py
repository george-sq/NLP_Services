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
import services_pretreatment as pts
import services_fileIO as fs
import multiprocessing
from functools import reduce
from sklearn.datasets.base import Bunch
import jieba

jieba.setLogLevel(log_level=logging.INFO)
logger = logging.getLogger(__name__)


def doCutWord(record):
    """
        :param record: [txtid,label,content]
        :return:
    """
    retVal = []
    txtid = record[0]
    wordSeqs = jieba.cut(record[1].replace('\r\n', '').replace('\n', '').replace(' ', ''))
    retVal.extend([txtid, list(wordSeqs)])
    return retVal


def buildWord2Vector(**kwargs):
    retVal = None
    corpus = kwargs.get("dataSet", None)
    if corpus is not None:
        logger.info("基于案件记录数据,构建Word2Vector模型")
        model_w2v = gensim.models.Word2Vec(sentences=corpus, min_count=1, hs=1, workers=multiprocessing.cpu_count())
        fs.FileServer().saveWord2VectorModel(path="./Out/Word2Vector/", fileName="word2vector_anjian.w2v",
                                             wvmodel=model_w2v)
        logger.info("构建Word2Vector模型完成")
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


def getRawCorpus():
    # 初始化Mysql数据库连接
    mysqls = dbs.MysqlServer()
    mysqls.setConnect(user="pamo", passwd="pamo", db="textcorpus")

    # 获取原始语料库数据
    result_query = mysqls.executeSql("SELECT * FROM tb_txtcate WHERE txtLabel='电信诈骗' ORDER BY txtId")
    records = [[str(record[0]), record[3]] for record in result_query[1:]]
    logger.info("获得案件记录数据,记录总数 %s records" % len(records))

    # 分词处理
    logger.info("对案件记录数据进行分词处理")
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    dataSets = pool.map(doCutWord, records)
    pool.close()
    pool.join()
    logger.info("分词处理完成")

    # 原始文本集
    txtIds = [record[0] for record in dataSets]
    rawCorpus = [record[1] for record in dataSets]
    raw = Bunch(txtIds=txtIds, rawCorpus=rawCorpus)

    # 构建Word2Vector
    # model_w2v = buildWord2Vector(dataSet=rawCorpus)
    # model_w2v = fs.FileServer().loadWord2VectorModel(path="./Out/Word2Vector/", fileName="word2vector_anjian.w2v")

    # 文本数字化
    # txtVecs = doTxtQuantizationByWord2Vector(dataSet=rawCorpus, model=model_w2v)

    # 数据标准化
    structDataHandler = pts.BaseStructData()
    dicts4corpus = structDataHandler.buildGensimDict(rawCorpus)

    # 标准化语料库
    corpus = structDataHandler.buildGensimCorpusByCorporaDicts(rawCorpus, dicts4corpus)

    # 统计TFIDF数据
    statDataHandler = pts.StatisticalData()
    model_tfidf = statDataHandler.buildGensimTFIDF(initCorpus=corpus)

    return raw, corpus, dicts4corpus, model_tfidf


def main():
    # 预处理
    raw, corpus, dicts, tfidfModel = getRawCorpus()
    tfidfVecs = tfidfModel[corpus]
    num_features = len(dicts)
    fileHandler = fs.FileServer()
    fileHandler.saveGensimDict(path="./Out/Dicts/", fileName="dict_anjian.dict", dicts=dicts)
    fileHandler.saveGensimTfidfModel(path="./Out/Models/", fileName="tfidf_anjian.mdl", tfidf=tfidfModel)
    fileHandler.savePickledObjFile(path="./Out/", fileName="raw_anjian.dat", writeContentObj=raw)
    fileHandler.saveGensimCourpus2MM(path="./Out/Corpus/", fileName="corpus_anjian.mm", inCorpus=corpus)

    # tfidf相似性
    indexTfidf = gensim.similarities.SparseMatrixSimilarity(tfidfVecs, num_features=num_features)
    fileHandler.saveIndex4tfidfSimilarity(path="./Out/Indexs/", fileName="Index_TFIDF_anjian.idx",
                                          index=indexTfidf)

    queryTxt = "事情是这样的，我今天收到一组协查公文，北京东城公安局今年1月在酒店发现一名吸毒过量的女性死者，据现场勘验死者为女性，" \
               "36岁。我问你，为什么在死者的信用卡中有一张是以你的名义办的？"
    # queryTxt = "收到假的工行积分兑换，点击链接(www.95588oy.cc)，输入号码和验证码，被骗1260元"
    bow_query = dicts.doc2bow(list(jieba.cut(queryTxt)))

    # 数字向量化
    tfidf_query = tfidfModel[bow_query]

    # tfidf相似性
    sim_tfidf_query = indexTfidf[tfidf_query]

    print("query tfidf相似性：", sorted(enumerate(sim_tfidf_query), key=lambda item: -item[1])[:5])


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
