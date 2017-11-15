# coding = utf-8
"""
    @File   : getData.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/9/19 10:31
    @Todo   :
"""

import dataBases as udb
import random
import jieba
from sklearn.datasets.base import Bunch
import pickle
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF向量生成类


def saveVectorSpaces(path, someBunch):
    """
        :param path: 本地文件系统的存储位置
        :param someBunch: 持久化到本地文件系统的对象
        @todo : 将数据集的向量空间持久化存储到本地文件系统
    """
    try:
        with open(path, "wb")as file:
            pickle.dump(someBunch, file)
    except FileNotFoundError:
        print('错误的文件路径，没有找到指定文件！！！')
        exit(1)


def loadVectorSpaces(objPath):
    """
        :param objPath: 持久化对象在本地文件系统中的位置路径
        :return: 持久化到本地文件系统的对象
        @todo : 加载持久化存储在本地文件系统中的数据集向量空间
    """
    try:
        with open(objPath, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print('错误的文件路径，没有找到指定文件！！！')
        exit(1)


def segWord(contents):
    """
        :param contents: 待分词内容
        :return: 分词后内容
        @todo : 文本分词处理
    """
    lines = contents.strip().split('\n')
    for i in range(len(lines)):
        lines[i] = lines[i].strip().split(' ')
        for s in range(len(lines[i])):
            lines[i][s] = lines[i][s].strip()
        lines[i] = "".join(lines[i])
    seg = jieba.cut("".join(lines).replace('\'', '\"'))
    return " ".join(seg)


def updateSegWd():
    rows = LoadData().getRawCorpus()[1]
    db = udb.useMysql()
    i = 0
    for row in rows:
        content = row[3]
        segwd = segWord(content)
        updateSql = "UPDATE tb_txtcate SET txtSegWd = '%s' WHERE txtId = %d " % (segwd, row[0])
        db.executeSql(updateSql)
        i += 1
        print('<<<<<<<<<<<<<<<<<<')
        print('      ', i)
        print('>>>>>>>>>>>>>>>>>>')
    print(i)


################################################################################################
class LoadData:
    """
        1、获取原始数据集
        2、划分训练集和测试集
    """

    def __init__(self):
        """初始化"""
        self.host = '10.0.0.247'
        self.port = 3306
        self.user = 'pamo'
        self.passwd = 'pamo'
        self.db = 'textcorpus'
        self.charset = 'utf8mb4'

    # 1、获取原始数据集
    def getRawCorpus(self):
        """
            :return: results = [rowNum,rows:[(),]]
            @todo : 从数据库中加载所有原始语料数据
        """
        db = udb.useMysql()
        db.setConnect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db=self.db,
                      charset=self.charset)
        return db.executeSql("SELECT * FROM tb_txtcate ORDER BY txtId")

    def getStpwd(self):
        """
            :return: results = [wordId:str,stopWord:str]
            @todo : 从数据库中加载停用词表
        """
        db = udb.useMysql()
        db.setConnect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db=self.db,
                      charset=self.charset)
        result = db.executeSql("SELECT * FROM tb_stopwords ORDER BY wordId")
        return [tp[1] for tp in result[1]]

    # 2、划分训练集和测试集
    @staticmethod
    def __dataConverter(paramsTuple):
        """
            :param paramsTuple: paramsTuple = rows:[(),]
            :return: reData = [fileID:str,fileContents:str,fileLabel:str]
            @todo : 语料数据结果预处理，fileContents为分词后的文本，对应数据库中的txtSegWd字段
        """
        reData = [paramsTuple[0], paramsTuple[4].strip()]
        if paramsTuple[2] == '电信诈骗':
            reData.append(paramsTuple[2])
        else:
            reData.append('非电信诈骗')
        return reData

    def buildSets(self, **kwargs):
        """
            :return: Tuple => (trainSets:[],testSets:[])
            @todo : 将原始语料数据集按9:1的比例，分别划分为测试集和训练集
        """
        corpus = self.getRawCorpus()[1]
        dataSets = []
        if len(kwargs) == 0:
            nums = len(corpus)
            testSize = int(nums * 0.1)
            testIndex = random.sample(range(nums), testSize)
            trainSets = []
            testSets = []
            for n in range(nums):
                if n in testIndex:
                    testSets.append(self.__dataConverter(corpus[n]))
                else:
                    trainSets.append(self.__dataConverter(corpus[n]))
            return trainSets, testSets
        elif kwargs.get('ALL'):
            for row in corpus:
                dataSets.append(self.__dataConverter(row))
            return dataSets
        else:
            print('参数错误！！！')
            exit(1)


class VectorSpaces:
    """
        2、划分训练集和测试集
        3、分词和词袋模型构建
        4、构建向量空间
    """

    def __init__(self):
        self.wgTrain = None
        self.wgTest = None

    @staticmethod
    def buildBunch(dataList):
        """
            :param dataList: list => [fileID:str,fileContents:str,fileLabel:str]
            :return: Tuple => (fileId:[],contents:[],label:[],className:[])
            @todo : 筛选需要信息，并分词处理
        """
        fileId = []
        contents = []
        label = []
        className = []
        for row in dataList:
            fileId.append(row[0])
            contents.append(row[1])
            label.append(row[2])
            if row[2] not in className:
                className.append(row[2])
        return fileId, contents, label, className

    def buildWordBag(self, **kwargs):
        """
            :param kwargs: trainSets OR testSets : Tuple => (fileId:[],contents:[],label:[],className:[])
            :return: Bunch(fileId=[], label=[], contents=[],className=[])
            @todo : 构建词袋模型
        """
        if kwargs.get('trainSets'):
            trainSets = kwargs.get('trainSets')
            self.wgTrain = Bunch(fileId=trainSets[0], contents=trainSets[1], label=trainSets[2], className=trainSets[3])
        if kwargs.get('testSets'):
            testSets = kwargs.get('testSets')
            self.wgTest = Bunch(fileId=testSets[0], contents=testSets[1], label=testSets[2], className=testSets[3])
        return self

    @staticmethod
    def buildTrainVectorSpaces(someBunch):
        """
            :param someBunch: Bunch(fileId=[], label=[], contents=[],className=[])
            :return: TFIDF_VectorSpace => Bunch(fileId=[], label=[], className=[], tdm=[], vocabulary=[])
            @todo : 构建训练数据集样本的向量空间
        """
        TFIDF_VectorSpace = Bunch(fileId=[], label=[], className=[], tdm=[], vocabulary=[])
        stpwd_list = LoadData().getStpwd()[1]
        vectorizer = TfidfVectorizer(stop_words=stpwd_list, sublinear_tf=True, max_df=0.5)
        TFIDF_VectorSpace.fileId.extend(someBunch.fileId)
        TFIDF_VectorSpace.label.extend(someBunch.label)
        TFIDF_VectorSpace.className.extend(someBunch.className)
        TFIDF_VectorSpace.tdm = vectorizer.fit_transform(someBunch.contents)
        TFIDF_VectorSpace.vocabulary = vectorizer.vocabulary_
        return TFIDF_VectorSpace

    @staticmethod
    def buildTestVectorSpaces(vec, someBunch):
        """
            :param vec: Bunch(fileId=[], label=[], className=[], tdm=[], vocabulary=[])
            :param someBunch: Bunch(fileId=[], label=[], contents=[],className=[])
            :return: TFIDF_VectorSpace => Bunch(fileId=[], label=[], className=[], tdm=[], vocabulary=[])
            @todo : 构建测试数据集样本的向量空间
        """
        TFIDF_VectorSpace = Bunch(fileId=[], label=[], className=[], tdm=[], vocabulary=[])
        stpwd_list = LoadData().getStpwd()[1]
        vectorizer = TfidfVectorizer(stop_words=stpwd_list, sublinear_tf=True, max_df=0.5, vocabulary=vec.vocabulary)
        TFIDF_VectorSpace.fileId.extend(someBunch.fileId)
        TFIDF_VectorSpace.label.extend(someBunch.label)
        TFIDF_VectorSpace.className.extend(someBunch.className)
        TFIDF_VectorSpace.tdm = vectorizer.fit_transform(someBunch.contents)
        TFIDF_VectorSpace.vocabulary = vec.vocabulary
        return TFIDF_VectorSpace


################################################################################################
# main测试入口
if __name__ == '__main__':
    # updateSegWd()

    pass
