# -*- coding: utf-8 -*-
"""
    @File   : services_bayes.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/20 14:43
    @Todo   : 提供本地的贝叶斯算法服务
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np


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

        # Returns classes_=['class1', 'class2',], jll=[ M x N ], predictLabels=[]
        return self.classes_, jll, self.classes_[np.argmax(jll, axis=1)]


class MultinomialNB2TextCates(object):
    """
        文本分类工具类
            模型构建
            文本类型预测
    """

    def __init__(self, dicts=None, tfidfVecs=None):
        self.clf = None
        self.dicts = dicts
        self.tfidfVecs = tfidfVecs

    # 模型构建
    def buildModel(self, labels=None, tdm=None):
        """
            :param labels: [label,]
            :param tdm: csr_matrix
            :return:
        """
        # tdm = modelVecs.tdm
        # labels = modelVecs.labels
        if tdm is not None and labels is not None:
            self.clf = LocalMultinomialNB(alpha=0.001).fit(tdm, labels)
            return self
        else:
            print('样本的向量空间（tdm） 和 样本的类型集合（labels） 不能为None 。')

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
                print("tdm = None , 参数错误。")
        else:
            print("Model = None , 请先通过模型训练构建模型")
        return result


def main():
    pass


if __name__ == '__main__':
    main()
