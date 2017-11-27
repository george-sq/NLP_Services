# -*- coding: utf-8 -*-
"""
    @File   : textCateServices.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/27 9:51
    @Todo   : 
"""

import textProcessServices as tps
import bayesServices as bayes


def calcPerformance(testLabels, cateResult):
    total = len(cateResult)
    rate = 0
    resultFile = './Out/cateResult1.txt'
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
    dicts, labels, tfidfVecs = tps.baseProcess()
    cols = len(dicts)

    # 标准化（数字化）
    csrm_tfidf = tps.vecs2csrm(tfidfVecs)
    print("labels.len :", len(labels))
    print("dicts.len :", len(dicts))
    print("csrm_tfidf.shape :", csrm_tfidf.shape)

    # 数据集划分 trainSet(90%) testSet(10%)
    subLabels, subDataSets = tps.splitDataSet(labels, tfidfVecs)
    trainLabels = subLabels[0]
    testLabels = subLabels[1]

    trainVecs = tps.vecs2csrm(subDataSets[0], cols)
    testVecs = tps.vecs2csrm(subDataSets[1], cols)
    print("trainVecs.shape :", trainVecs.shape)
    print("testVecs.shape :", testVecs.shape)

    # 模型构建
    bayesTool = bayes.MultinomialNB2TextCates()
    bayesTool.buildModel(labels=trainLabels, tdm=trainVecs)

    # 模型评估
    cateResult = bayesTool.modelPredict(tdm=testVecs)
    calcPerformance(testLabels, cateResult)  # 性能计算


if __name__ == '__main__':
    main()
