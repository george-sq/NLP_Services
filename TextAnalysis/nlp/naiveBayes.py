# -*- coding: utf-8 -*-
"""
    @File   : naiveBayes.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/2/6 10:21
    @Todo   : 
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LocalMultinomialNB(MultinomialNB):
    """
        多项式贝叶斯算法类
    """

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        super(LocalMultinomialNB, self).__init__(alpha, fit_prior, class_prior)

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

        # Returns classes_=['class1', 'class2',], jll=[ M x N ], predictLabels=[]
        return self.classes_, jll, self.classes_[np.argmax(jll, axis=1)]


class MultinomialNB2TextCates(object):
    """
        文本分类工具类
            模型构建
            文本类型预测
    """

    def __init__(self):
        self.clf = None

    # 模型构建
    def buildModel(self, labels=None, tdm=None):
        """
            :param labels: [label,]
            :param tdm: csr_matrix
            :return:
        """
        if tdm is not None and labels is not None:
            self.clf = LocalMultinomialNB(alpha=0.001).fit(tdm, labels)
            return True
        else:
            logger.error("Params type error! tdm=%s and labels=%s" % (tdm, labels))

    # 类型预测
    def modelPredict(self, tdm=None):
        """
            :param tdm:
            :return: Tuple => (预测类型:str, 预测类型的概率)
        """
        result = False
        if isinstance(self.clf, LocalMultinomialNB):
            if tdm is not None:
                # return classes_ = ['涉嫌电诈' '非电诈相关'], jll = [M x N], predictLabels = []
                clas, likelihoods, resLabel = self.clf.predict(tdm)
                result = []
                for llh, label in zip(likelihoods, resLabel):
                    fz = max(llh)
                    fm = llh[0] + llh[1]
                    llh = 1 - (fz / fm)
                    result.append((label, llh))
            else:
                logger.error("Params type error! tdm=%s" % tdm)
        else:
            logger.error("Params type error! clf=%s" % self.clf)
        return result
