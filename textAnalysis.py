# coding = utf-8
"""
    @File   : textAnalysis.py
    @Author : NLP_QingShen (275171387@qq.com)
    @Time   : 2017/9/18 15:32
    @Todo   : 
"""
import jieba
from sklearn.datasets.base import Bunch
import getData as bd
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot


class LocalMultinomialNB(MultinomialNB):
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
        return self.classes_, jll, self.classes_[np.argmax(jll, axis=1)]


class TextCates:
    """
        1、获取原始数据集
        2、划分训练集和测试集
        3、分词和词袋模型构建
        4、构建向量空间
        5、模型构建
        6、类型预测
    """

    def __init__(self):
        self.clf = None

    # 5、模型构建
    def buildModel(self, tdm, lab):
        """
            :param tdm:
            :param lab:
            :return:
        """
        self.clf = LocalMultinomialNB(alpha=0.001).fit(tdm, lab)
        return self

    # 6、类型预测
    def modelPredict(self, tdm):
        """
            :param tdm:
            :return: Tuple => (classes:[], 各个类型的概率:[], 预测类型:str)
        """
        return self.clf.predict(tdm)


def modelTest():
    """
        1、获取原始数据集
        2、划分训练集和测试集
        3、分词和词袋模型构建
        4、构建向量空间
        5、模型构建
        6、类型预测
    """
    # 1、获取原始数据集、划分训练集和测试集、分词和词袋模型构建
    dataSets = bd.LoadData().buildSets()
    # 4、构建向量空间
    vcs = bd.VectorSpaces()
    vcs.buildWordBag(trainSets=bd.VectorSpaces.buildBunch(dataSets[0]),
                     testSets=bd.VectorSpaces.buildBunch(dataSets[1]))

    train = bd.VectorSpaces.buildTrainVectorSpaces(vcs.wgTrain)  # 训练集
    test = bd.VectorSpaces.buildTestVectorSpaces(train, vcs.wgTest)  # 测试集

    # 5、模型构建
    model = TextCates()
    model.buildModel(train.tdm, train.label)
    print('train.shape =', np.shape(train.tdm))
    print('test.shape =', np.shape(test.tdm))

    # 6、类型预测
    predicted = model.modelPredict(test.tdm)[2]

    # 预测性能评估
    total = len(predicted)
    rate = 0

    for label, fileId, expct_cate in zip(test.label, test.fileId, predicted):
        if label != expct_cate:
            rate += 1
            print(fileId, " 实际类别： ", label, " --> 错误预测分类： ", expct_cate)
            # else:
            #     print(fileId, " 实际类别： ", label, " ==> 预测分类： ", expct_cate)

    # 模型精度
    print("error : ", rate, "  total : ", total)
    print("error_rate : ", str(float(rate * 100 / float(total))))


def buildDat():
    # 1、获取原始数据集、划分训练集和测试集、分词和词袋模型构建
    dataSets = bd.LoadData().buildSets(ALL=True)
    # 4、构建向量空间
    vcs = bd.VectorSpaces()
    vcs.buildWordBag(trainSets=bd.VectorSpaces.buildBunch(dataSets))

    train = bd.VectorSpaces.buildTrainVectorSpaces(vcs.wgTrain)  # 训练集

    bd.saveVectorSpaces('./Out/DataVectorSpaces.dat', train)


# main入口
if __name__ == '__main__':
    # 模型测试
    modelTest()

    # 数据构建
    # buildDat()

    print('完成文本分类测试。')

    options, args = getopt.getopt(sys.argv[1:], "", ["content=","stpwd=","train="])
    content = ""
    stpwd_path = ""
    train_path = ""
    if len(options)==0 :
        print("paramError: 传入参数的数量不正确！！！")
        exit(1)
    else:
        for k,v in options:
            if k=='--content':
                content=v
            elif k=='--stpwd':
                stpwd_path=v
            elif k=='--train':
                train_path=v
    print("Content:"+content,)
    print("Stpwd:"+stpwd_path)
    print("Train:"+train_path)
    print("Result:"+TextCategorized().py4java(content,stpwd_path,train_path))
