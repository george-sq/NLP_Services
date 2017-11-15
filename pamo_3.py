# -*- coding: utf-8 -*-
"""
    @File   : pamo_3.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/10/31 14:49
    @Todo   : 
"""

import getopt
import sys
import time
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import pickle
import jieba
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


def timeCost(func):
    def cost(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        useTime = time.time() - startTime
        logging.info('>>>>' * 30)
        logging.info('\t\t\t\t\t\t%s  [ %s ]  cost time  %s s' % (func.__class__, func.__name__, useTime))
        logging.info('<<<<' * 30)
        return result

    return cost


@timeCost
def doWordCut(inStr, stpwdPath):
    """
        :param inStr:
        :param stpwdPath:
        :return: list --> [word,]
    """
    wordSeqs = jieba.cut(inStr)
    stopWords = list(corpora.Dictionary.load(stpwdPath).token2id)
    # return list(words)
    return [word for word in wordSeqs if word not in stopWords]


@timeCost
def dataConvertor(bunchObj):
    """
        :param bunchObj: Bunch(txtIds=[], classNames=[], labels=[], tdm=[], vocabulary=[])
        :return: dataList --> list [[(dictIndex, word, tfidf), ], ]
    """
    dataList = []
    vocabulary = dict(zip(bunchObj.vocabulary.values(), bunchObj.vocabulary.keys()))
    matrix = bunchObj.tdm
    for rowId in range(matrix.shape[0]):
        row = matrix[rowId].toarray()[0]
        wSeqs = []
        for i in range(len(row)):
            vals = row[i]
            if vals != 0:
                wSeqs.append((str(i), vocabulary[i], str(vals)))

        dataList.append(wSeqs)

    return dataList


@timeCost
def loadDatFile(objPath):
    try:
        with open(objPath, 'rb') as pkb:
            return pickle.load(pkb)
    except FileNotFoundError:
        print('FileNotFoundError: 错误的文件路径(%s)，没有找到指定文件！！！' % objPath)
        exit(1)


########################################################################################################################
# class LoadData(object):
#     """
#         原始语料加载类
#     """
#
#     def __init__(self):
#         # self.stopWordsDicts = corpora.Dictionary.load('./Out/Dicts/stopWords_ALL.dict').token2id
#         # self.stopWords = dict(zip(self.stopWordsDicts.values(), self.stopWordsDicts.keys()))
#         pass
#
#     @staticmethod
#     def loadDatFile(objPath):
#         try:
#             with open(objPath, 'rb') as pkb:
#                 return pickle.load(pkb)
#         except FileNotFoundError:
#             print('错误的文件路径，没有找到指定文件！！！')
#             exit(1)


########################################################################################################################
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

        # Returns classes_=['涉嫌电诈' '非电诈相关'], jll=[ M x N ], predictLabels=[]
        return self.classes_, jll, self.classes_[np.argmax(jll, axis=1)]


########################################################################################################################
class PamoWordBag(object):
    """
        词包构建类
    """

    def __init__(self):
        self.classNames = ['非电诈相关', '涉嫌电诈']

    @timeCost
    def wordSeqs2WordBag(self, wordSeqs=None):
        """
            :param wordSeqs: [word,]
            :return: bow = Bunch(txtIds, classNames=['非电诈相关', '涉嫌电诈'], labels=[], contents=[wordSeqs])
        """
        bow = Bunch(classNames=self.classNames, labels=[], contents=[])
        if wordSeqs is not None:
            bow.contents.append(' '.join(wordSeqs))
            return bow
        else:
            print('wordSeqs is None！方法需要输入参数wordSeqs。')
            return None


########################################################################################################################
class PamoTfidf(object):
    """
        Tfidf构建类
    """

    def __init__(self):
        self.TFIDF_Test = Bunch(classNames=[], labels=[], tdm=[], vocabulary=[])

    @timeCost
    def buildTestTfidf(self, bowObj=None, trainVecs=None):
        """
            :param bowObj: Bunch(classNames=['非电诈相关', '涉嫌电诈'], labels=[], contents=[])
            :param trainVecs: Bunch(classNames=['非电诈相关', '涉嫌电诈'], labels=[], tdm=[], vocabulary=[])
            :return: self.TFIDF_Test = Bunch(classNames=[], labels=[], tdm=[], vocabulary=[])
        """
        if bowObj is not None and trainVecs is not None:
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                         vocabulary=trainVecs.vocabulary)
            self.TFIDF_Test.classNames.extend(bowObj.classNames)
            self.TFIDF_Test.vocabulary = trainVecs.vocabulary
            self.TFIDF_Test.tdm = vectorizer.fit_transform(bowObj.contents)
            return self.TFIDF_Test
        else:
            print('bowObj or trainVecs is None！方法需要输入参数 bowObj(词袋对象) 和 trainVecs(训练集向量空间)。')
            return None


########################################################################################################################
class TextCates(object):
    """
        # 模型构建
        # 类型预测
    """

    def __init__(self):
        self.clf = None

    # 5、模型构建
    @timeCost
    def buildModel(self, modelVecs=None):
        """
        :param modelVecs: Bunch(txtIds=[], classNames=[], labels=[], tdm=[], vocabulary=[])
        :return:
        """
        tdm = modelVecs.tdm
        labels = modelVecs.labels
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
            :return: Tuple => (预测类型:str, 预测类型的概率)
        """
        # return classes_ = ['涉嫌电诈' '非电诈相关'], jll = [M x N], predictLabels = []
        clas, likelihoods, resLabel = self.clf.predict(tdm)
        likelihoods = likelihoods[0]
        # print(type(likelihoods))
        # print(likelihoods.shape)
        # print(len(likelihoods))
        fz = max(likelihoods)
        fm = likelihoods[0] + likelihoods[1]
        llh = 1 - (fz / fm)
        return list(resLabel)[0] + "(" + str(round(llh * 100, 3)) + ")"


########################################################################################################################
@timeCost
def py4java(docContent, p_stpwd, p_model):
    """
        :param docContent:
        :param p_stpwd:
        :param p_model:
        :return:
    """
    """
        :param tstDoc:
        :return:
    """
    # 加载数据
    # ld = LoadData()
    # modelVecs = ld.loadDatFile('./Out/Dats/pamo_nlp_DataVectorSpaces.dat')
    # modelVecs = ld.loadDatFile(p_model)
    modelVecs = loadDatFile(p_model)

    # 分词
    # wordSeqs = doWordCut(tstDoc)
    wordSeqs = doWordCut(docContent, p_stpwd)
    # 词袋模型
    bowTool = PamoWordBag()
    testBow = bowTool.wordSeqs2WordBag(wordSeqs)
    # TFIDF向量空间
    vecsTool = PamoTfidf()
    testVecs = vecsTool.buildTestTfidf(testBow, modelVecs)
    # 多项式贝叶斯模型
    nbModel = TextCates()
    nbModel.buildModel(modelVecs)
    # 预测
    predicted = nbModel.modelPredict(testVecs.tdm)
    result = "Result>>" + predicted
    cvtDatas = dataConvertor(testVecs)[0]
    keyContent = "KeyContent>>"
    for iwf in cvtDatas:
        strOut = "\t".join(iwf) + ","
        keyContent += strOut
    return result, keyContent


def usage():
    """
        The output  configuration file contents.

        Usage: xxx.py -c|--content arg0 -s|--stpwd arg1 -m|--model arg2

        Description
                    -c,--content  Configure Text content information.
                    -s,--stpwd    Configure StopWords Path information.
                    -m,--model    Configure NLP_Model Path information.
        For Example:
            python xxx.py -c|--content textContent -s|--stpwd stpwd_path -m|--model model_path
    """


if __name__ == '__main__':
    # try:
    #     options, argss = getopt.getopt(sys.argv[1:], "c:s:m:", ["content=", "stpwd=", "model="])
    #     content = ""
    #     stpwd_path = ""
    #     model_path = ""
    #     # print(argss)
    #     if len(options) == 0:
    #         print("paramError: 传入参数的数量不正确！！！")
    #         print(usage.__doc__)
    #         exit(1)
    #     else:
    #         for k, v in options:
    #             if k in ['--content', '-c']:
    #                 content = v
    #             if k in ['--stpwd', '-s']:
    #                 stpwd_path = v
    #             if k in ['--model', '-m']:
    #                 model_path = v
    # except getopt.GetoptError as GER:
    #     print('GetoptError:', GER)
    #     print(usage.__doc__)
    #     sys.exit(1)
    #
    # print("Content>>" + content)
    # print("Stpwd>>" + stpwd_path)
    # print("Model>>" + model_path)
    # res = py4java(content, stpwd_path, model_path)
    # for line in res:
    #     print(line)
#######################################################################################################################
    # strDoc = '刑法修正案十草案:侮辱国歌情节严重者最高可判3年'
    strDoc = "8月初，陆某在网上找兼职刷单工作,通过QQ与对方联系后,按对方的指引下单选定商品，将未付款的订单截图发给对方。" \
             "对方要求其通过支付宝扫描发来的二维码进行付款，刷完第一单小额订单后对方按照约定返还。陆某开始相信对方，随后开始刷第二单，" \
             "扫描支付完后对方又称第二单有三个任务且每个任务内含多个商品，要求完成三个任务后方可返还本金和佣金。"
    res = py4java(strDoc, './Out/Dicts/stopWords_ALL.dict', './Out/Dats/pamo_nlp_DataVectorSpaces.dat')
    print()
    for line in res:
        print(line)
