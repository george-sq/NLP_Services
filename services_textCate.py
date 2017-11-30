# -*- coding: utf-8 -*-
"""
    @File   : services_textCate.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/27 9:51
    @Todo   : 
"""

import services_textProcess as tps
import services_bayes2cate as bayes
import pickle


def calcPerformance(testLabels, cateResult):
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


def main():
    # 预处理
    labels, corpus, dicts, tfidfModel, tfidfVecs, freqFile = tps.baseProcess()
    cols = len(dicts)
    del freqFile
    del tfidfModel

    # 数据集划分 trainSet(90%) testSet(10%)
    subLabels, subDataSets = tps.splitDataSet(labels, tfidfVecs)
    trainLabels = subLabels[0]
    testLabels = subLabels[1]

    # 标准化（数字化）
    trainVecs = tps.vecs2csrm(subDataSets[0], cols)
    testVecs = tps.vecs2csrm(subDataSets[1], cols)
    corpusVecs = tps.vecs2csrm(tfidfVecs)

    # 模型构建
    bayesTool = bayes.MultinomialNB2TextCates()
    bayesTool.dicts = dicts
    bayesTool.tfidfModel = tfidfModel
    for i in range(len(labels)):
        if "电信诈骗" != labels[i]:
            labels[i] = "非电诈相关"
        else:
            labels[i] = "电诈相关"
    bayesTool.buildModel(labels=labels, tdm=corpusVecs)
    try:
        with open("./Out/bayesModel.pickle", "wb") as fw:
            pickle.dump(bayesTool, fw, protocol=4)
    except FileNotFoundError as fne:
        print(fne)

    # 模型测试
    bayesTool = bayes.MultinomialNB2TextCates()
    bayesTool.buildModel(labels=trainLabels, tdm=trainVecs)

    # 模型评估
    cateResult = bayesTool.modelPredict(tdm=testVecs)
    calcPerformance(testLabels, cateResult)  # 性能计算


if __name__ == '__main__':
    main()
