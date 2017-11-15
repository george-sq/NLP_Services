# -*- coding: utf-8 -*-
"""
    @File   : pmnlp.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/8 16:08
    @Todo   : 
"""

import os
import logging
import time
import datetime

from wordcloud import WordCloud
import pickle
import pymysql
import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF向量生成类
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
import jieba
from jieba import posseg as posg

jieba.setLogLevel(log_level=logging.INFO)
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora
from gensim import models


########################################################################################################################
class UseMysql(object):
    """连接Mysql数据库的相关操作"""

    def __init__(self):
        """初始化"""
        self.host = '10.0.0.247'
        self.port = 3306
        self.user = 'pamodata'
        self.passwd = 'pamodata'
        self.db = 'db_pamodata'
        self.charset = 'utf8mb4'

    def __getConnect(self):
        """连接Mysql数据库"""
        # 创建连接
        return pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db=self.db,
                               charset=self.charset)

    def executeSql(self, *args):
        """
            :param args:[sql,[sql args,]]
            :return: results = [受影响的行数, (行内容,)]
        """
        """执行SQL"""
        # 获取数据库连接
        con = self.__getConnect()
        # 使用cursor()方法获取操作游标
        cursor = con.cursor()
        results = []
        try:
            if len(args) == 1:
                results.append(cursor.execute(args[0]))
                results.extend(cursor.fetchall())
            elif len(args) == 2:
                results.append(cursor.executemany(args[0], args[1]))
                results.append(cursor.fetchall())
            else:
                print('输入参数错误！！！')
                raise ValueError
            con.commit()
        except Exception as e:
            # 输出异常信息
            print(args[0], "异常信息 : ", e)
            print(args[0], "异常信息 : ", e)
            print(args[0], "异常信息 : ", e)
            # 数据库回滚
            con.rollback()
        finally:
            cursor.close()
            # 关闭数据库连接
            con.close()
        return results


########################################################################################################################
class LocalMultinomialNB(MultinomialNB):
    """
        多项式贝叶斯算法类
    """

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
        self.stopWords = None

    @staticmethod
    def doWordCut(content, stopWords):
        """
            :param content:
            :param stopWords:
            :return: list --> [(tid, wordSeqs),]
        """
        wordSeqs = jieba.cut(content.replace('\r\n', '').replace('\n', ''))
        return [word for word in wordSeqs if word not in stopWords]

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
        文本分类工具类
            模型构建
            文本类型预测
    """

    def __init__(self):
        self.clf = None

    # 模型构建
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

    # 类型预测
    def modelPredict(self, tdm):
        """
            :param tdm:
            :return: Tuple => (预测类型:str, 预测类型的概率)
        """
        # return classes_ = ['涉嫌电诈' '非电诈相关'], jll = [M x N], predictLabels = []
        clas, likelihoods, resLabel = self.clf.predict(tdm)
        likelihoods = likelihoods[0]
        fz = max(likelihoods)
        fm = likelihoods[0] + likelihoods[1]
        llh = 1 - (fz / fm)
        return list(resLabel)[0] + "(" + str(round(llh * 100, 3)) + ")"


########################################################################################################################
class TxtInfos(object):
    """
        文本处理的工具类
    """

    def __init__(self):
        self.dbs = UseMysql()
        self.localDict = corpora.Dictionary.load('./Out/Dicts/pamo_dicts.dict')
        self.localCorpus = corpora.MmCorpus('./Out/Corpus/pamo_gaCorpus.mm')
        self.tfidfModel = models.TfidfModel(self.localCorpus)

    @staticmethod
    def loadDatFile(objPath):
        try:
            with open(objPath, 'rb') as pkb:
                return pickle.load(pkb)
        except FileNotFoundError:
            print('FileNotFoundError: 错误的文件路径(%s)，没有找到指定文件！！！' % objPath)
            exit(1)

    def getNewTxtStruct(self):
        sql = "SELECT * FROM tb_tinfo WHERE tstatus = 0 AND tid NOT IN (SELECT tid FROM tb_structinfo)"
        result = self.dbs.executeSql(sql)
        return result

    def getNewCateTxt(self):
        """
            :return:
        """
        sqlById = "SELECT * FROM tb_tinfo WHERE tstatus = 1 AND tid NOT IN (SELECT tid FROM tb_tcate)"
        result = self.dbs.executeSql(sqlById)
        return result

    @staticmethod
    def dataConvertor(dicts, kwseqs):
        """
            :param kwseqs:
            :param dicts:
            :return:
        """
        wfSeq = []
        vocabulary = dict(zip(dicts.token2id.values(), dicts.token2id.keys()))
        for ind, fre in kwseqs:
            tt = (vocabulary[ind], fre)
            wfSeq.append(tt)

        return wfSeq

    @staticmethod
    def buildWordCloudWithFreq(freData, imgId):
        """
            :param freData: 频率数据
            :param imgId: 用以命名的文本ID=tid
            :return: imgPath=词云图的本地路径
        """
        freData = {k: round(float(v) * 1000) for k, v in freData.items()}
        wordcloud = WordCloud(max_words=2000, width=1300, height=600, background_color="white",
                              font_path='C:/Windows/Fonts/STSONG.TTF').generate_from_frequencies(freData)

        nowTime = str(time.time()).split('.')[0]
        imgRoot = 'E:/WordCloudImgs/'
        if not os.path.exists(imgRoot):
            os.mkdir(imgRoot)
        imgPath = imgRoot + 't' + imgId + '_' + nowTime + '.png'
        wordcloud.to_file(imgPath)
        return imgPath

    def modStructInfos(self, rec):
        """
            :param rec: 数据库中未处理的文本记录record
            :return: (tid, strWps, strKiseqs, imgPath)
        """
        params = []
        tid = rec[0]
        txt = rec[1]
        # 文本预处理 tb_structinfo
        #     分词和词性标注
        #     抽取关键词集合
        #         加载词典
        #         TFIDF权值
        # 词性序列
        pos = posg.cut(txt)
        wps = ['|'.join(wp) for wp in pos]
        strWps = '\t'.join(wps)
        # 关键词序列
        wseqs = [wp.split('|')[0] for wp in wps]
        txtBow = self.localDict.doc2bow(wseqs)
        kwseqs = self.dataConvertor(self.localDict, self.tfidfModel[txtBow])
        kwseqs = sorted(kwseqs, key=lambda tt: tt[1], reverse=True)
        kwseqs = [(k, str(w)) for k, w in kwseqs]
        strKiseqs = '\t'.join(['|'.join(kw) for kw in kwseqs])
        imgPath = self.buildWordCloudWithFreq(dict(kwseqs), str(tid))
        # 拼接Insert SQL语句的插入参数
        params.append((strWps, strKiseqs, tid, imgPath, datetime.datetime.now()))

        records = self.getNewTxtStruct()
        ids = [rec[0] for rec in records[1:]]  # 未处理文本ID集合
        if tid in ids:  # 新文本数据处理
            result = self.dbs.executeSql(
                "INSERT INTO tb_structinfo(pos, kwseqs, tid, imgpath, modtime) VALUES (%s, %s, %s, %s, %s)", params)
            # 更新tb_tinfo表中tid对应记录的status字段为1
            if len(result) > 0 and 1 == result[0]:
                result = self.dbs.executeSql("UPDATE tb_tinfo SET tstatus=1 WHERE tid=%d" % tid)

        else:  # 已有文本数据处理
            result = self.dbs.executeSql("SELECT * FROM tb_structinfo WHERE tid=%s" % tid)
            if len(result) > 0 and 1 == result[0]:
                result = self.dbs.executeSql(
                    "UPDATE tb_structinfo SET pos='%s',kwseqs='%s',imgpath='%s',modtime='%s' WHERE tid='%s'" % (
                        strWps, strKiseqs, imgPath, datetime.datetime.now(), tid))
        if len(result) > 0:
            result = (tid, strWps, strKiseqs, imgPath)
        else:
            result = None
        return result

    def modCateInfos(self, inParams):
        """
            :param inParams: [args, stpwds, lmodel,]
                    args: (tid, content,)
                    stpwds: 停用词
                    lmodel: 本地模型向量
            :return:
        """
        # 参数初始化
        rows = inParams[0]
        tid = rows[0]
        stpwds = inParams[1]
        lmodel = inParams[2]
        bowTool = PamoWordBag()

        # 分词
        wordSeqs = bowTool.doWordCut(rows[1], stpwds)
        # 构建词袋模型
        testBow = bowTool.wordSeqs2WordBag(wordSeqs)
        # 构建TFIDF向量空间
        vecsTool = PamoTfidf()
        testVecs = vecsTool.buildTestTfidf(testBow, lmodel)
        # 构建多项式贝叶斯模型
        nbModel = TextCates()
        nbModel.buildModel(lmodel)
        # 预测文本类型
        tlabel = nbModel.modelPredict(testVecs.tdm)

        # 文本数据库情况分析
        records = self.getNewCateTxt()
        ids = [rec[0] for rec in records[1:]]
        if tid in ids:  # 新文本数据分类
            # 更新tb_tcate表数据
            insertSql = "INSERT INTO tb_tcate(tlabel, tid, modtime) VALUES ('%s', '%s', '%s')" % (
                tlabel, tid, datetime.datetime.now())
            result = self.dbs.executeSql(insertSql)
        else:  # 已有文本分类类型更新
            updateSql = "UPDATE tb_tcate SET tlabel='%s', modtime='%s' WHERE tid=%s" % (
                tlabel, datetime.datetime.now(), tid)
            result = self.dbs.executeSql(updateSql)

        if len(result) > 0 and 1 == result[0]:
            result = self.dbs.executeSql("SELECT * FROM tb_tcate WHERE tid = %d" % tid)
        else:
            result = None
        return result


########################################################################################################################
def doTxtCate(tObj, args):
    """
        :param tObj: TxtInfos类对象
        :param args: (tid, content,)
        :return:
    """
    # 加载预处理数据
    modelVecs = tObj.loadDatFile('./Out/Dats/pamo_nlp_DataVectorSpaces.dat')
    stopWords = list(corpora.Dictionary.load('./Out/Dicts/stopWords_ALL.dict').token2id)
    # 参数验证
    reStr = ""
    ci = tObj.modCateInfos([args, stopWords, modelVecs])
    if len(ci) > 0 and ci is not None:
        reStr += "\ntLabel : " + ci[1][1]
    return reStr


def doKeyInfosExtraction(tObj, args):
    """
        :param tObj: TxtInfos类对象
        :param args: (tid, content,)
        :return:
    """
    # 参数验证
    si = tObj.modStructInfos(args)
    reStr = "posSeqs : " + si[1] + "\nkiSeqs : " + si[2] + "\nwciPath : " + si[3]
    return reStr


def main(inStr):
    """
        :param inStr: 客户端请求数据
        :return:
    """
    # 客户端请求数据解析
    paramsDict = {}
    if len(inStr) > 0:
        lines = inStr.split('\n')
        for ln in lines:
            k, v = ln.split(' : ')
            paramsDict.setdefault(k, v)
    tid = int(paramsDict.get("tid", -1))
    content = paramsDict.get("content", "")
    responseMsg = ""
    if -1 == tid or "" == content:
        pass
    else:
        # 初始化
        txtTool = TxtInfos()
        # 关键信息抽取
        strctInfos = doKeyInfosExtraction(txtTool, (tid, content))
        # 文本分类
        cateInfos = doTxtCate(txtTool, (tid, content))
        # 数据处理结果判断
        if strctInfos is not None and cateInfos is not None:
            responseMsg += strctInfos + cateInfos + '\n'
    print(responseMsg)
    return responseMsg


if __name__ == '__main__':
    # content = "8月初，陆某在网上找兼职刷单工作,通过QQ与对方联系后,按对方的指引下单选定商品，将未付款的订单截图发给对方。" \
    #          "对方要求其通过支付宝扫描发来的二维码进行付款，刷完第一单小额订单后对方按照约定返还。陆某开始相信对方，随后开始刷第二单，" \
    #          "扫描支付完后对方又称第二单有三个任务且每个任务内含多个商品，要求完成三个任务后方可返还本金和佣金。"
    tstStr = 'tid : 2\ncontent : 8月初，陆某在网上找兼职刷单工作,通过QQ与对方联系后,按对方的指引下单选定商品，将未付款的订单截图发给对方。'
    main(tstStr)
