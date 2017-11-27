# -*- coding: utf-8 -*-
"""
    @File   : textCateServices.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/27 9:51
    @Todo   : 
"""

import textProcessServices as tps
import bayesServices as bayes


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
    likelihoods = cateResult[1]
    results = cateResult[2]
    # retDatas = dataConvertor(testVecs)
    # 性能计算
    total = len(results)
    rate = 0
    # resultFile = './Out/cateResult.txt'
    lines = ['文本ID\t\t实际类别\t\t预测类别\t\t预处理后的文本内容(文本词序列)\n']
    errLines = ['文本ID\t\t实际类别\t\t预测类别\t\t预处理后的文本内容(文本词序列)\n']
    # for label, expct_cate, lls, idAddTxt in zip(testVecs.labels, results, likelihoods, retDatas):
    for label, expct_cate, lls in zip(testLabels, results, likelihoods):
        fz = max(lls)
        fm = lls[0] + lls[1]
        llss = 1 - (fz / fm)
        # txtId = idAddTxt[0]
        # txt = idAddTxt[1]
        if label != expct_cate:
            rate += 1
            # print("文本ID: %s \t\t实际类别: %s --> 错误预测分类:%s \n文本词序列: %s" % (txtId, label, expct_cate, txt))
            print("实际类别: %s --> 错误预测分类:%s" % (label, expct_cate))
            # for index, word, freq in txt:
            # for index, word, freq in txt:
            #     print('\t\t', index, word, freq)
            errLine = '%s\t\t%s(%s)\n' % (label, expct_cate, str(round(llss, 3)))
            errLines.append(errLine)
        else:
            # line = '%s\t\t%s\t\t%s(%s)\t\t%s\n' % (txtId, label, expct_cate, str(round(llss, 3)), txt)
            line = '%s\t\t%s(%s)\n' % (label, expct_cate, str(round(llss, 3)))
            lines.append(line)
    # 模型精度
    lines.append('\n' + '>>>>>>>>>>' * 5)
    lines.append('\nTotal : %s\t\tError : %s\n' % (total, rate))
    lines.append('error_rate : %s \n' % str(float(rate * 100 / float(total))))
    lines.append('accuracy_rate : %s \n' % str(float(100 - (rate * 100 / float(total)))))
    lines.append('>>>>>>>>>>' * 5 + '\n\n')
    # with open(resultFile, 'w', encoding='utf-8') as fw:
    #     fw.writelines(lines)
    #     fw.writelines(errLines)
    pass


if __name__ == '__main__':
    main()
