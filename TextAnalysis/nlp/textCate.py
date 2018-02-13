# -*- coding: utf-8 -*-
"""
    @File   : textCateServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/2/1 17:32
    @Todo   : 
"""

import logging
import random

import numpy as np
from scipy.sparse.csr import csr_matrix
from sklearn.datasets.base import Bunch

from bases.fileServer import FileServer
from bases.mysqlServer import MysqlServer
from basicTextProcessing import BasicTextProcessing, TfidfVecSpace
from naiveBayes4txtCate import MultinomialNB2TextCates

logger = logging.getLogger(__name__)


def __getStopWords():
    """ 获取停用词
    :return:
    """
    stopWords = []
    stopWords_EN = FileServer().loadTextByUTF8('../../StopWords/', 'stopWords_EN.txt')
    stopWords_CN = FileServer().loadTextByUTF8('../../StopWords/', 'stopWords_CN.txt')
    stopWords.extend(stopWords_EN)
    stopWords.extend(stopWords_CN)
    stopWords.append("")
    stopWords.append(" ")
    return set(stopWords)


def __baseProcess():
    """ 语料库预处理
    :return:
    """
    # 初始化Mysql数据库连接
    # dbHandler = MysqlServer(host="10.0.0.247", db="db_pamodata", user="pamo", passwd="pamo")
    dbHandler = MysqlServer(host="192.168.0.113", db="TextCorpus", user="root", passwd="mysqldb")

    # 获取原始语料库数据
    result_query = dbHandler.executeSql(sql="SELECT * FROM corpus_rawtxts ORDER BY txtId")
    labelsIndex = [(str(record[0]), record[2]) for record in result_query[1:]]
    txts = [record[3] for record in result_query[1:]]

    # 生成文本预处理器
    textHandler = BasicTextProcessing()

    # 获取停用词库
    stopWords = __getStopWords()
    # dicts4stopWords = textHandler.buildGensimDict([list(stopWords)], stored=(True, "../../Out/Dicts/stopWords.dict"))
    textHandler.buildGensimDict([list(stopWords)], stored=(True, ("../../Out/Dicts/", "stopWords.dict")))

    # 对原始语料库样本进行分词处理
    dataSets = list(textHandler.batchWordSplit(contentList=txts))

    # 对原始语料库进行去停用词处理
    for i in range(len(dataSets)):
        txt = dataSets[i]
        dataSets[i] = [txt[j] for j in range(len(txt)) if txt[j] not in stopWords]

    # 生成原始语料库的语料库词典
    dicts4corpus = textHandler.buildGensimDict(dataSets)

    # 将原始语料库的样本进行数字化处理，生成数字化语料库
    corpus = textHandler.buildGensimCorpusByCorporaDicts(dataSets, dicts4corpus)

    # 根据数字化语料库生成TFIDF向量空间
    tfidf = TfidfVecSpace()
    tfidf.buildVecsByGensim(initCorpus=corpus, corpus=corpus)
    tfidfModel = tfidf.TFIDF_Model
    tfidf4corpus = tfidf.TFIDF_Vecs

    return labelsIndex, corpus, dicts4corpus, tfidfModel, tfidf4corpus


class TextCateServer(object):
    def __init__(self):
        pass

    @staticmethod
    def vecs2csrm(vecs, columns=None):
        """ 转换成稀疏矩阵
        :param vecs: 需要转换的向量空间[[iterm,],]
        :param columns: 列的维度
        :return: csr_matrix
        """
        data = []  # 存放的是非0数据元素
        indices = []  # 存放的是data中元素对应行的列编号（列编号可重复）
        indptr = [0]  # 存放的是行偏移量

        for vec in vecs:  # 遍历数据集
            for colInd, colData in vec:  # 遍历单个数据集
                indices.append(colInd)
                data.append(colData)
            indptr.append(len(indices))

        if columns is not None:
            retCsrm = csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, columns), dtype=np.double)
        else:
            retCsrm = csr_matrix((data, indices, indptr), dtype=np.double)

        retCsrm.sort_indices()

        return retCsrm

    def buildCateModel(self, **kwargs):
        """ 构建本地数据模型
        :param kwargs: labels=, vecsSet=, dicts=, tfidfModel=
        :return:
        """
        # 参数解析
        labels = kwargs.get("labels", None)
        tfidfVecs = kwargs.get("vecsSet", None)
        dicts = kwargs.get("dicts", None)
        tfidfModel = kwargs.get("tfidfModel", None)
        len_labels = len(labels)
        len_tfidfVecs = len(tfidfVecs)

        # 标准化（数字化）
        if tfidfVecs is not None and len_labels == len_tfidfVecs:
            corpusVecs = self.vecs2csrm(tfidfVecs)

            # 构建本地模型
            bayesTool = MultinomialNB2TextCates()
            bayesTool.buildModel(labels=labels, tdm=corpusVecs)
            logger.info("Build naive bayes model for classifying text SUCCESSED!")

            if dicts is not None and tfidfModel is not None:
                # bayesTool.dicts = dicts
                # bayesTool.tfidfModel = tfidfModel
                logger.info("Stored naive bayes model")
                textCate = Bunch(dicts=dicts, tfidf=tfidfModel, nbayes=bayesTool)
                # 本地存储
                fileHandler = FileServer()
                fileHandler.savePickledObjFile(path="../../Out/Models/", fileName="nbTextCate-2018.pickle",
                                               writeContentObj=textCate)
                logger.info("Stored naive bayes model SUCCESSED!")
                return textCate

            else:
                logger.error("Params missing! (dicts=%s or tfidfModel=%s)" % (dicts, tfidfModel))
        else:
            logger.error("Params error! (tfidfVecs=%s or labels=%s)" % (tfidfVecs, labels))


def splitDataSet(labels, vectorSpace):
    """
        :param labels: [label,]
        :param vectorSpace:
        :return:
    """
    # 划分训练集和测试集
    trainSet = []
    testSet = []
    labelsA = []  # 非电信诈骗类型的索引集合
    labelsB = []  # 电信诈骗类型的索引集合
    for i in range(len(labels)):
        if "电诈案件" != labels[i]:
            labelsA.append(i)
        else:
            labelsB.append(i)
    trainSet.extend(random.sample(labelsA, int(len(labelsA) * 0.9)))
    trainSet.extend(random.sample(labelsB, int(len(labelsB) * 0.9)))
    testSet.extend([index for index in labelsA if index not in trainSet])
    testSet.extend([index for index in labelsB if index not in trainSet])

    trainLabel = [labels[index] for index in trainSet]
    testLabel = [(index, labels[index]) for index in testSet]
    trainSet = [vectorSpace[index] for index in trainSet]
    testSet = [vectorSpace[index] for index in testSet]

    return [trainLabel, testLabel], [trainSet, testSet]


def calcPerformance(testLabels, cateResult):
    """
        :param testLabels: []
        :param cateResult:
        :return:
    """
    total = len(cateResult)
    rate = 0
    resultFile = '../../Out/Models/cateResult.txt'
    lines = ['文本ID\t\t实际类别\t\t预测类别\n']
    errLines = ['文本ID\t\t实际类别\t\t预测类别\n']
    for labelTuple, cateTuple in zip(testLabels, cateResult):
        txtId = labelTuple[0]
        label = labelTuple[1]
        cate = cateTuple[0]
        llh = cateTuple[1]
        if label != cate:
            rate += 1
            print("文本编号: %s >>> 实际类别: %s >>> 错误预测分类:%s" % (txtId, label, cate))
            errLine = '%s\t\t%s\t\t%s(%s)\n' % (txtId, label, cate, str(round(llh, 3)))
            errLines.append(errLine)
        else:
            line = '%s\t\t%s\t\t%s(%s)\n' % (txtId, label, cate, str(round(llh, 3)))
            lines.append(line)
    # 模型精度
    lines.append('\n' + '>>>>>>>>>>' * 5)
    lines.append('\nTotal : %s\t\tError : %s\n' % (total, rate))
    lines.append('error_rate : %s \n' % str(float(rate * 100 / float(total))))
    lines.append('accuracy_rate : %s \n' % str(float(100 - (rate * 100 / float(total)))))
    lines.append('>>>>>>>>>>' * 5 + '\n\n')
    with open(resultFile, 'w', encoding='utf-8') as fw:
        fw.writelines(lines)
        fw.writelines(errLines)


def algorithmTest(labels=None, dataSet=None, cols=0):
    """ 算法测试
        :param labels: []
        :param dataSet: [[],]
        :param cols: len(dict4corpus)
        :return:
    """
    retVal = False
    # 数据集划分 trainSet(90%) testSet(10%)
    if labels is not None and dataSet is not None:
        subLabels, subDataSets = splitDataSet(labels, dataSet)

        trainLabels = subLabels[0]
        testLabels = subLabels[1]

        if 0 != cols:
            trainVecs = TextCateServer.vecs2csrm(subDataSets[0], cols)
            testVecs = TextCateServer.vecs2csrm(subDataSets[1], cols)

            # 模型测试
            bayesTool = MultinomialNB2TextCates()
            bayesTool.buildModel(labels=trainLabels, tdm=trainVecs)

            # 模型评估
            cateResult = bayesTool.modelPredict(tdm=testVecs)
            calcPerformance(testLabels, cateResult)  # 性能计算
            retVal = True
        else:
            logger.error("Params error! (cols=%s)" % cols)
    else:
        logger.error("Params error! (labels=%s or dataSet=%s)" % (labels, dataSet))

    return retVal


def main():
    # 预处理
    lIndex, corpus, dicts, model, tfidfVecs = __baseProcess()
    tfidfVecs = list(tfidfVecs)
    labels = [l for i, l in lIndex]
    # 社会综合：交通、政治、教育、环境、经济、艺术、体育
    # 军事：
    # 医药：
    # 计算机：
    # 电诈案件：
    for i in range(len(labels)):
        soc = ["交通", "政治", "教育", "环境", "经济", "艺术", "体育"]
        if labels[i] in soc:
            labels[i] = "社会综合"
        elif "电信诈骗" == labels[i]:
            labels[i] = "电诈案件"
        pass

    # 模型构建
    ts = TextCateServer()
    ts.buildCateModel(labels=labels, dicts=dicts, tfidfModel=model, vecsSet=tfidfVecs)

    # 性能评估
    cols = len(dicts)
    algorithmTest(labels=labels, dataSet=tfidfVecs, cols=cols)
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
