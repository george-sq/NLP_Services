# -*- coding: utf-8 -*-
"""
    @File   : services_textCate.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/27 9:51
    @Todo   : 
"""

import logging
import services_textProcess as tps
import services_bayes2cate as bayes
import pickle

logger = logging.getLogger(__name__)


def calcPerformance(testLabels, cateResult):
    """
        :param testLabels: []
        :param cateResult:
        :return:
    """
    total = len(cateResult)
    rate = 0
    resultFile = './Out/cateResult.txt'
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


def buildLocalModel(**kwargs):
    """ 构建本地数据模型
        :param kwargs: labels=, vecsSet=, dicts=, tfidfModel=
        :return:
    """
    retVal = False
    # 参数解析
    labels = kwargs.get("labels", None)
    tfidfVecs = kwargs.get("vecsSet", None)
    dicts = kwargs.get("dicts", None)
    tfidfModel = kwargs.get("tfidfModel", None)
    len_labels = len(labels)
    len_tfidfVecs = len(tfidfVecs)

    # 标准化（数字化）
    if tfidfVecs is not None and len_labels == len_tfidfVecs:
        corpusVecs = tps.vecs2csrm(tfidfVecs)

        # 构建本地模型
        bayesTool = bayes.MultinomialNB2TextCates()

        bayesTool.buildModel(labels=labels, tdm=corpusVecs)

        if dicts is not None and tfidfModel is not None:
            bayesTool.dicts = dicts
            bayesTool.tfidfModel = tfidfModel
            # 本地存储
            try:
                with open("./Out/bayesModel.pickle", "wb") as fw:
                    pickle.dump(bayesTool, fw, protocol=4)
            except FileNotFoundError as fne:
                print(fne)
            retVal = True
        else:
            logger.error("ERROR:参数错误 (dicts=%s or tfidfModel=%s)" % (dicts, tfidfModel))
            logger.error("ERROR:参数错误 (dicts=%s or tfidfModel=%s)" % (dicts, tfidfModel))
            logger.error("ERROR:参数错误 (dicts=%s or tfidfModel=%s)" % (dicts, tfidfModel))
    else:
        logger.error("ERROR:参数错误 (tfidfVecs=%s or labels=%s)" % (tfidfVecs, labels))
        logger.error("ERROR:参数错误 (tfidfVecs=%s or labels=%s)" % (tfidfVecs, labels))
        logger.error("ERROR:参数错误 (tfidfVecs=%s or labels=%s)" % (tfidfVecs, labels))

    return retVal


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
        subLabels, subDataSets = tps.splitDataSet(labels, dataSet)

        trainLabels = subLabels[0]
        testLabels = subLabels[1]

        if 0 != cols:
            trainVecs = tps.vecs2csrm(subDataSets[0], cols)
            testVecs = tps.vecs2csrm(subDataSets[1], cols)
            # 模型测试
            bayesTool = bayes.MultinomialNB2TextCates()
            bayesTool.buildModel(labels=trainLabels, tdm=trainVecs)

            # 模型评估
            cateResult = bayesTool.modelPredict(tdm=testVecs)
            calcPerformance(testLabels, cateResult)  # 性能计算
            retVal = True
        else:
            logger.error("ERROR:参数错误 (cols=%s)" % cols)
            logger.error("ERROR:参数错误 (cols=%s)" % cols)
            logger.error("ERROR:参数错误 (cols=%s)" % cols)
    else:
        logger.error("ERROR:参数错误 (labels=%s or dataSet=%s)" % (labels, dataSet))
        logger.error("ERROR:参数错误 (labels=%s or dataSet=%s)" % (labels, dataSet))
        logger.error("ERROR:参数错误 (labels=%s or dataSet=%s)" % (labels, dataSet))

    return retVal


def main():
    # 预处理
    labels, corpus, dicts, tfidfModel, tfidfVecs, freqFile = tps.baseProcess()
    tfidfVecs = list(tfidfVecs)
    cols = len(dicts)
    del freqFile
    for i in range(len(labels)):
        if "电信诈骗" != labels[i]:
            labels[i] = "非电诈相关"
        else:
            labels[i] = "电诈相关"

    # 构建本地模型
    process_1 = buildLocalModel(labels=labels, dicts=dicts, tfidfModel=tfidfModel, vecsSet=tfidfVecs)

    # 算法测试
    process_2 = algorithmTest(labels=labels, dataSet=tfidfVecs, cols=cols)

    if process_1:
        print("构建本地模型成功！！！")
    else:
        print("构建本地模型失败。。。")
    if process_2:
        print("算法测试成功！！！")
    else:
        print("算法测试失败。。。")


if __name__ == '__main__':
    main()
