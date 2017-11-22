# -*- coding: utf-8 -*-
"""
    @File   : pamo_2.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/10/23 16:35
    @Todo   : 
"""
import os
import time
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import multiprocessing
from multiprocessing import Pool
import jieba
import random
from gensim import corpora
import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF向量生成类
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import getData as gd


def timeCost(func):
    def cost(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        useTime = time.time() - startTime
        logging.info('>>>>' * 30)
        logging.info('\t\t\t\t\t\t%s  [ %s ]  cost time  %s s' % (
            func.__class__, func.__name__, useTime))
        logging.info('<<<<' * 30)
        return result

    return cost


def doWordCut(inStr):
    """
        :param inStr:
        :return: list --> [word,]
    """
    words = jieba.cut(inStr)
    return list(words)


def buildWordSeq(argsTuple):
    """
        :param argsTuple: (txtId, txtLabel, text, stopWords -> set)
        :return: words --> (label, words)
    """
    txtId = argsTuple[0]
    label = argsTuple[1]
    text = argsTuple[2]
    stopWords = argsTuple[3]
    wordSeqs = doWordCut(text.strip().replace('\n', ''))
    wordSeqs = [word for word in wordSeqs if word not in stopWords]
    # yield label, words
    return txtId, label, wordSeqs


@timeCost
def splitDataSets(dataSets):
    """
        :param dataSets: [(label,words),]
        :return:
    """
    newSets = [row for row in dataSets]  # 全集
    testSet = []  # 测试集
    dz = [row for row in newSets if row[1] == '电信诈骗']
    fdz = [row for row in newSets if row not in dz]
    # 筛选10%的测试集
    tstNum = int(len(dz) * 0.1)
    indexs = random.sample(range(len(dz)), tstNum)
    for n in indexs:
        testSet.append(dz[n])
    tstNum = int(len(fdz) * 0.1)
    indexs = random.sample(range(len(fdz)), tstNum)
    for n in indexs:
        testSet.append(fdz[n])
    # 构建测试集
    trainSet = [row for row in newSets if row not in testSet]

    return trainSet, testSet


@timeCost
def dataConvertor(bunchObj):
    """
        :param bunchObj: Bunch(txtIds=[], classNames=[], labels=[], tdm=[], vocabulary=[])
        :return: dataList --> list [[(dictIndex, word, tfidf), ], ]
    """
    dataList = []
    vocabulary = {v: k for k, v in bunchObj.vocabulary.items()}
    txtIds = bunchObj.txtIds
    matrix = bunchObj.tdm
    for rowId in range(matrix.shape[0]):
        row = matrix[rowId].toarray()[0]
        txtid = txtIds[rowId]
        wSeqs = []
        for i in range(len(row)):
            vals = row[i]
            if vals != 0:
                wSeqs.append((i, vocabulary[i], str(vals)))

        dataList.append((txtid, wSeqs))

    return dataList


########################################################################################################################
class LoadData(object):
    """
        原始语料加载类
    """

    def __init__(self):
        self.stopWords = None

    @staticmethod
    def loadFile(filePath):
        with open(filePath, 'r', encoding='utf-8') as fr:
            contents = fr.read()
            contents = contents.split('\n')
            return contents

    @staticmethod
    def saveFile(filePath, contents):
        """
            :param filePath:
            :param contents: list[line,]
            :return:
        """
        with open(filePath, 'w', encoding='utf-8') as fw:
            fw.writelines(contents)

    @timeCost
    def getStopWords(self):
        stopWords = []
        stopWords_EN = LoadData.loadFile('./StopWords/stopWords_EN.txt')
        stopWords_CN = LoadData.loadFile('./StopWords/stopWords_CN.txt')
        stopWords.extend(stopWords_EN)
        stopWords.extend(stopWords_CN)
        stopWords.append(' ')
        self.stopWords = set(stopWords)
        lStopwords = list(self.stopWords)
        dictionary = corpora.Dictionary([lStopwords])
        dictionary.save('./Out/Dicts/stopWords_ALL.dict')
        # self.saveFile('./StopWords/stopWords_ALL.txt', '\n'.join(self.stopWords))
        return self

    @timeCost
    def getRawGenerator(self):
        """
            :return: results --> generator object [(label,txt,stopWords),]
        """
        raws = gd.LoadData().getRawCorpus()
        if self.stopWords is not None:
            for raw in raws:
                yield raw[0], raw[2], raw[3], self.stopWords
        else:
            print('self.stopWords is None! 请先获取停用词库。')


########################################################################################################################
"""
    e、统计处理
"""


class LocalMultinomialNB(MultinomialNB):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        super(LocalMultinomialNB, self).__init__(alpha, fit_prior, class_prior)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse='csr')

        return (safe_sparse_dot(X, self.feature_log_prob_.T) +
                self.class_log_prior_)

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X
        """
        jll = self._joint_log_likelihood(X)
        # print(self.classes_, jll, self.classes_[np.argmax(jll, axis=1)])
        # print(np.argmax(jll, axis=1))

        # Returns classes_=['涉嫌电诈' '非电诈相关'], jll=[ M x N ], predictLabels=[]
        return self.classes_, jll, self.classes_[np.argmax(jll, axis=1)]


class TextCates(object):
    """
        # 模型构建
        # 类型预测
    """

    def __init__(self):
        self.clf = None

    # 5、模型构建
    @timeCost
    def buildModel(self, tdm=None, labels=None):
        """
            :param tdm:
            :param labels:
            :return:
        """
        if tdm is not None and labels is not None:
            self.clf = LocalMultinomialNB(alpha=0.001).fit(tdm, labels)
            return self
        else:
            print('样本的向量空间（tdm） 和 样本的类型集合（labels） 不能为None 。')

    # 6、类型预测
    @timeCost
    def modelPredict(self, tdm):
        """
            :param tdm:
            :return: Tuple => (classes:[], 各个类型的概率:[], 预测类型:str)
        """
        return self.clf.predict(tdm)


class PamoWordBag(object):
    """
        词包构建类
    """

    def __init__(self):
        self.classNames = ['非电诈相关', '涉嫌电诈']

    @timeCost
    def list2WordBag(self, dataSets=None):
        """
            :param dataSets: [(lab,txt -> [word,]),]
            :return: bow = Bunch(txtIds, classNames=['非电诈相关', '涉嫌电诈'], labels=[], contents=[wordSeqs])
        """
        bow = Bunch(txtIds=[], classNames=self.classNames, labels=[], contents=[])
        if dataSets is not None:
            for row in dataSets:
                txtid = row[0]
                lab = row[1]
                txt = row[2]
                if lab != '电信诈骗':
                    lab = bow.classNames[0]
                else:
                    lab = bow.classNames[1]

                bow.txtIds.append(txtid)
                bow.labels.append(lab)
                bow.contents.append(' '.join(txt))
            return bow
        else:
            print('dataSets is None！方法需要输入参数dataSets。')
            return None


class PamoTfidf(object):
    """
        Tfidf构建类
    """

    def __init__(self):
        self.TFIDF_Train = Bunch(txtIds=[], classNames=[], labels=[], tdm=[], vocabulary=[])
        self.TFIDF_Test = Bunch(txtIds=[], classNames=[], labels=[], tdm=[], vocabulary=[])

    @timeCost
    def buildTrainTfidf(self, bowObj=None):
        """
            :param bowObj: Bunch(txtIds, classNames=['非电诈相关', '涉嫌电诈'], labels=[], contents=[wordSeqs,])
            :return: self.TFIDF_Train = Bunch(txtIds=[], classNames=[], labels=[], tdm=[], vocabulary=[])
        """
        if bowObj is not None:
            vocabulary = corpora.Dictionary.load('./Out/Dicts/pamo_dicts.dict')
            self.TFIDF_Train.txtIds.extend(bowObj.txtIds)
            self.TFIDF_Train.vocabulary = vocabulary.token2id
            self.TFIDF_Train.classNames.extend(bowObj.classNames)
            self.TFIDF_Train.labels.extend(bowObj.labels)
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                         vocabulary=self.TFIDF_Train.vocabulary)  # 将测试集文本映射到训练集词典中
            self.TFIDF_Train.tdm = vectorizer.fit_transform(bowObj.contents)
            return self.TFIDF_Train
        else:
            print('bowObj is None！方法需要输入参数bowObj(词袋对象)。')
            return None

    @timeCost
    def buildTestTfidf(self, bowObj=None, trainVecs=None):
        """
            :param bowObj: Bunch(txtIds, classNames=['非电诈相关', '涉嫌电诈'], labels=[], contents=[])
            :param trainVecs: Bunch(txtIds, classNames=['非电诈相关', '涉嫌电诈'], labels=[], tdm=[], vocabulary=[])
            :return: self.TFIDF_Test = Bunch(txtIds=[], classNames=[], labels=[], tdm=[], vocabulary=[])
        """
        if bowObj is not None and trainVecs is not None:
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                         vocabulary=trainVecs.vocabulary)  # 将测试集文本映射到训练集词典中
            self.TFIDF_Test.txtIds.extend(bowObj.txtIds)
            self.TFIDF_Test.classNames.extend(bowObj.classNames)
            self.TFIDF_Test.labels.extend(bowObj.labels)
            self.TFIDF_Test.vocabulary = trainVecs.vocabulary  # 将测试集文本映射到训练集词典中
            self.TFIDF_Test.tdm = vectorizer.fit_transform(bowObj.contents)
            return self.TFIDF_Test
        else:
            print('bowObj or trainVecs is None！方法需要输入参数 bowObj(词袋对象) 和 trainVecs(训练集向量空间)。')
            return None


########################################################################################################################
"""
    <<主函数>>
"""


@timeCost
def modeTest(Corpus):
    # 2、划分训练集和测试集
    dataSets = splitDataSets(Corpus)

    # 3、转换词包模型
    bowTool = PamoWordBag()
    trainBow = bowTool.list2WordBag(dataSets[0])
    testBow = bowTool.list2WordBag(dataSets[1])

    # 4、构建TFIDF向量空间
    vecsTool = PamoTfidf()
    trainVecs = vecsTool.buildTrainTfidf(trainBow)
    testVecs = vecsTool.buildTestTfidf(testBow, trainVecs)

    # 5、构建贝叶斯模型
    nbModel = TextCates()
    nbModel.buildModel(trainVecs.tdm, trainVecs.labels)

    # 6、模型评估
    predicted = nbModel.modelPredict(testVecs.tdm)
    likelihoods = predicted[1]
    results = predicted[2]
    retDatas = dataConvertor(testVecs)
    # 性能计算
    total = len(results)
    rate = 0
    resultFile = './Out/cateResult.txt'
    lines = ['文本ID\t\t实际类别\t\t预测类别\t\t预处理后的文本内容(文本词序列)\n']
    errLines = ['文本ID\t\t实际类别\t\t预测类别\t\t预处理后的文本内容(文本词序列)\n']
    for label, expct_cate, lls, idAddTxt in zip(testVecs.labels, results, likelihoods, retDatas):
        fz = max(lls)
        fm = lls[0] + lls[1]
        llss = 1 - (fz / fm)
        txtId = idAddTxt[0]
        txt = idAddTxt[1]
        if label != expct_cate:
            rate += 1
            print("文本ID: %s \t\t实际类别: %s --> 错误预测分类:%s \n文本词序列: %s" % (txtId, label, expct_cate, txt))
            for index, word, freq in txt:
                print('\t\t', index, word, freq)
            errLine = '%s\t\t%s\t\t%s(%s)\t\t%s\n' % (txtId, label, expct_cate, str(round(llss, 3)), txt)
            errLines.append(errLine)
        else:
            line = '%s\t\t%s\t\t%s(%s)\t\t%s\n' % (txtId, label, expct_cate, str(round(llss, 3)), txt)
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


@timeCost
def storeData(Corpus):
    # 3、转换词包模型
    bowTool = PamoWordBag()
    allBow = bowTool.list2WordBag(Corpus)

    # 4、构建TFIDF向量空间
    vecsTool = PamoTfidf()
    allVecs = vecsTool.buildTrainTfidf(allBow)

    # 5、将样本向量空间数据存储到本地磁盘
    path = "./Out/Dats"
    if not os.path.exists(path):
        os.mkdir(path)
    gd.saveVectorSpaces(path + '/pamo_nlp_DataVectorSpaces.dat', allVecs)

    # 6、样本数据分析
    classNames = allBow.classNames
    print('classNames.type =', type(classNames), '\t\t' * 3, classNames)
    labels = allBow.labels
    print('labels.type =', type(labels), '\t\t' * 3, 'labels.len =', len(labels))
    wordSeqs_docs = allBow.contents
    print('docs.type =', type(wordSeqs_docs), '\t\t' * 3, 'docs.len =', len(wordSeqs_docs))
    vocabularyDict = allVecs.vocabulary
    print('vocabularyDict.type =', type(vocabularyDict), '\t\t' * 3, 'vocabularyDict.len =', len(vocabularyDict))
    tdm = allVecs.tdm
    print('tdm.type =', type(tdm), '\t\t' * 3, 'tdm.shape =', np.shape(tdm))


def main():
    # 0、加载语料
    ld = LoadData()
    ld.getStopWords()
    gRawCorpus = ld.getRawGenerator()

    # 1、分词
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>创建进程池。')
    pool = Pool(multiprocessing.cpu_count())
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>初始化进程池完成。')
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>分配子进程任务。')
    gTxtCorpus = pool.imap(buildWordSeq, gRawCorpus)
    pool.close()
    pool.join()
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>子进程任务完成。')

    ####################################################################################################################
    corpus = list(gTxtCorpus)
    # 模型测试
    modeTest(corpus)
    # 将样本的向量空间数据存储到本地磁盘
    storeData(corpus)


if __name__ == '__main__':
    main()
