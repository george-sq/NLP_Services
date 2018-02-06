# -*- coding: utf-8 -*-
"""
    @File   : textCateServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/2/6 16:02
    @Todo   : 
"""

from bases.fileServer import FileServer
from nlp.basicTextProcessing import BasicTextProcessing
from nlp.textCate import MultinomialNB2TextCates, TextCateServer
import logging
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora
from gensim import models

logger = logging.getLogger(__name__)


def main():
    bayes = None
    # 测试数据预处理
    txt = "1月4日，东四路居民张某，在微信聊天上认识一位自称为香港做慈善行业的好友，对方称自己正在做慈善抽奖活动，因与张某关系好，" \
          "特给其预留了30万中奖名额，先后以交个人所得税、押金为名要求张某以无卡存款的形式向其指定的账户上汇款60100元"
    textHandler = BasicTextProcessing()
    wordSeqs = textHandler.doWordSplit(txt)[0]

    # 去停用词
    fileHandler = FileServer()

    stopWords = fileHandler.loadLocalGensimDict(path="../../Out/Dicts/", fileName="stopWords.dict")
    wordSeqs = [word for word in wordSeqs if word not in stopWords.token2id.keys()]

    # 加载文本分类模型
    nbModel = fileHandler.loadPickledObjFile(path="../../Out/Models/", fileName="nbTextCate-2018.pickle")

    # 解析模型数据
    dicts = nbModel.dicts
    tfidf = nbModel.tfidf
    nbayes = nbModel.nbayes

    # 预处理测试文本
    txtsBow = []
    testVecs = []
    if isinstance(dicts, corpora.Dictionary):
        txtsBow.append(dicts.doc2bow(wordSeqs))
    if isinstance(tfidf, models.TfidfModel):
        testVecs = tfidf[txtsBow]
    # 生成测试文本的稀疏矩阵向量
    testVecs = list(testVecs)
    csrm_testVecs = TextCateServer.vecs2csrm(testVecs, len(dicts))
    # 文本类型预测
    if isinstance(nbayes, MultinomialNB2TextCates):
        cateResult = nbayes.modelPredict(tdm=csrm_testVecs)
        print(cateResult)


if __name__ == '__main__':
    main()
