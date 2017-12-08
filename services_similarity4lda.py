# -*- coding: utf-8 -*-
"""
    @File   : services_similarity4lda.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/12/6 11:26
    @Todo   : 
"""

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
from gensim import models
from gensim import similarities
import services_fileIO as fs
import jieba

jieba.setLogLevel(log_level=logging.INFO)


def main():
    # 加载数据
    fileHandler = fs.FileServer()
    corpus = fileHandler.loadLocalMmCorpus(path="./Out/Corpus/", fileName="corpus_anjian.mm")
    dicts = fileHandler.loadLocalGensimDict(path="./Out/Dicts/", fileName="dict_anjian.dict")
    tfidfModel = fileHandler.loadGensimTfidfModel(path="./Out/Models/", fileName="model_TfidfModel_anjian.mdl")
    raw = fileHandler.loadPickledObjFile(path="./Out/", fileName="raw_anjian.dat")

    # lda相似性
    tfidfVecs = tfidfModel[corpus]
    lda = models.LdaModel(tfidfVecs, id2word=dicts)
    fileHandler.saveTopicModel(path="./Out/Models/", fileName="model_LdaTopicModel_anjian.mdl", tmodel=lda)
    indexLda = similarities.MatrixSimilarity(lda[tfidfVecs])
    fileHandler.saveIndex4topicSimilarity(path="./Out/Indexs/", fileName="Index_LdaTopicModel_anjian.idx",
                                          index=indexLda)

    queryTxt = "微信上买手机，转账被骗3100 现在此地，请妥处 嫌疑人微信号qq421149709 "

    bow_query = dicts.doc2bow(list(jieba.cut(queryTxt)))

    # 数字向量化
    tfidf_query = tfidfModel[bow_query]

    # lda相似性
    lda_query = lda[tfidf_query]

    sim_lda_query = indexLda[lda_query]

    results = sorted(enumerate(sim_lda_query), key=lambda item: -item[1])[:5]

    print(results)
    txtIds = raw.txtIds
    results = [(txtIds[index], freq) for index, freq in results]

    print("query lda相似性：", results)


if __name__ == '__main__':
    # 创建一个handler，用于写入日志文件
    logfile = "./Logs/log_similarity4lda.log"
    fileLogger = logging.FileHandler(filename=logfile, encoding="utf-8")
    fileLogger.setLevel(logging.NOTSET)

    # 再创建一个handler，用于输出到控制台
    stdoutLogger = logging.StreamHandler()
    stdoutLogger.setLevel(logging.INFO)  # 输出到console的log等级的开关

    logging.basicConfig(level=logging.NOTSET,
                        format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S", handlers=[fileLogger, stdoutLogger])
    main()
