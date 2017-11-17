# -*- coding: utf-8 -*-
"""
    @File   : pmnlp.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/8 16:08
    @Todo   : 
"""

import os
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import time
import jieba
from jieba import posseg as posg
from gensim import corpora
from gensim import models
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import pickle
import pymysql
import datetime
import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF向量生成类
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot


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
        wordSeqs = jieba.cut(content)
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
        sql = "SELECT * FROM tb_tinfo WHERE tstatus = 0 OR tid NOT IN (SELECT tid FROM tb_structinfo)"
        result = self.dbs.executeSql(sql)
        return result

    def getNewCateTxt(self):
        """
            :return:
        """
        sqlById = "SELECT * FROM tb_tinfo WHERE tstatus = 0 OR tid NOT IN (SELECT tid FROM tb_tcate)"
        result = self.dbs.executeSql(sqlById)
        return result

    def getTxtInfo(self, txtid=None):
        """
            :param txtid:
            :return:
        """
        if txtid is not None:
            return self.dbs.executeSql("SELECT * FROM tb_tinfo WHERE tid ='%d'" % txtid)[0]

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

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        nowTime = str(time.time()).split('.')[0]
        imgRoot = 'E:/WordCloudImgs/'
        if not os.path.exists(imgRoot):
            os.mkdir(imgRoot)
        imgPath = imgRoot + 't' + imgId + '_' + nowTime + '.png'
        wordcloud.to_file(imgPath)
        return imgPath

    def insertStructinfo(self, raws):
        """
            :param raws: 数据库中未处理的文本记录
            :return: 非负数为成功
        """
        params = []
        for row in raws:
            tid = row[0]
            # 词性序列
            pos = posg.cut(row[1])
            wps = ['|'.join(wp) for wp in pos]
            strWps = '\t'.join(wps)
            # 关键词序列
            wseqs = [wp.split('|')[0] for wp in wps]
            txtBow = self.localDict.doc2bow(wseqs)
            kwseqs = self.dataConvertor(self.localDict, self.tfidfModel[txtBow])
            kwseqs = sorted(kwseqs, key=lambda tt: tt[1], reverse=True)
            kwseqs = [(k, str(w)) for k, w in kwseqs]
            strKwseqs = '\t'.join(['|'.join(kw) for kw in kwseqs])
            imgPath = self.buildWordCloudWithFreq(dict(kwseqs), str(tid))
            # 拼接Insert SQL语句的插入参数
            params.append((strWps, strKwseqs, tid, imgPath, datetime.datetime.now()))

        result = self.dbs.executeSql(
            "INSERT INTO tb_structinfo(pos, kwseqs, tid, imgpath, modtime) VALUES (%s, %s, %s, %s, %s)", params)
        return result

    @staticmethod
    def insertCateInfos(inParams):
        """
        :param inParams: [(rows, stpwds, lmodel),]
                rows: [(tid,content,tsrc,fid,modtime),]
                stpwds: 停用词
                lmodel: 本地模型向量
        :return:
        """
        rows = inParams[0]
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
        predicted = nbModel.modelPredict(testVecs.tdm)
        tlabel = predicted
        insertSql = "INSERT INTO tb_tcate(tlabel, tid, modtime) VALUES ('%s', '%s', '%s')" % (
            tlabel, rows[0], datetime.datetime.now())
        rr = UseMysql().executeSql(insertSql)
        print(rr)
        return rr


########################################################################################################################
def doTxtCate(p_stpwd, p_model):
    txtTool = TxtInfos()
    # 1、加载预处理数据
    modelVecs = txtTool.loadDatFile(p_model)
    stopWords = list(corpora.Dictionary.load(p_stpwd).token2id)

    # 2、从tb_tinfo获取文本内容
    txtInfos = txtTool.getNewCateTxt()
    if txtInfos[0] > 0:
        pass

    # 3、构建参数列表 params
    params = []
    for infos in txtInfos:
        params.append([infos, stopWords, modelVecs])

    # 4、更新tb_tcate表数据
    for p in params:
        txtTool.insertCateInfos(p)

    pool = Pool(multiprocessing.cpu_count())
    pool.imap(txtTool.insertCateInfos, params)
    # gTxtCorpus = pool.imap(insertCateInfos, params)
    pool.close()
    pool.join()
    print('end')


def doKeyInfosExtraction():
    docTool = TxtInfos()
    raws = docTool.getNewTxtStruct()
    # 2、文本预处理 tb_structinfo
    #     2.1 分词和词性标注
    #     2.2 抽取关键词集合
    #         2.2.1 加载词典
    #         2.2.3 TFIDF权值
    docTool.insertStructinfo(raws)


def main():
    doKeyInfosExtraction()
    # doTxtCate('./Out/Dicts/stopWords_ALL.dict', './Out/Dats/pamo_nlp_DataVectorSpaces.dat')


if __name__ == '__main__':
    main()
