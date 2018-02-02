# -*- coding: utf-8 -*-
"""
    @File   : basicTextProcessingServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/19 10:05
    @Todo   : 
"""

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora
from gensim import models
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF向量生成类
from collections import defaultdict
from bases.mysqlServer import MysqlServer
import multiprocessing
import re
import jieba
from jieba import posseg
import logging

logger = logging.getLogger(__name__)

__url_regExp = re.compile(
    r"(?:(?:(?:https?|ftp|file)://(?:www\.)?|www\.)[a-z0-9+&@#/%=~_|$?!:,.-]*[a-z0-9+&@#/%=~_|$])|"
    r"(?:[a-z0-9+&@#/%=~_|$?!:,.-]+(?:\.[a-z]+))", re.IGNORECASE)

__money_regExp = re.compile(r"((?:(?:\d(?:\.[0-9]+多?)?|one|two|three|four|five|six|seven|eight|nine|ten|一|二|两|三|四|五|六|"
                            r"七|八|九|十|零|兆|亿|万|千|百|拾|玖|捌|柒|陆|伍|肆|叁|贰|壹)+)(?:[多余])?(?:\s*(?:hundred|thousand|Million|"
                            r"Billion)?\s*)(?:元|人民币|rmb|美元|美金|dollars?|韩元|日元|欧元|英镑))",
                            re.IGNORECASE)

__idcard_regExp = re.compile(r"(?<!\d)((?:(?:[1-9]\d{5})(?:(?:18|19|2\d)\d{2}[0-1]\d[0-3]\d)(?:\d{3})[\dxX])|"
                             r"[1-9]\d{5}\d{2}(?:(?:0[1-9])|(?:10|11|12))(?:(?:[0-2][1-9])|10|20|30|31)\d{2}[0-9Xx])"
                             r"(?!\d)")

__phoneNumber_regExp = re.compile(r"(?<!\d)(?:([(+（]{0,2})?(?: ?[0-9]{2,4} ?)(?:[)-）] ?)?)?(?:1[3-9]\d{9})(?!\d)")

__bankCard_regExp = re.compile(r"((?<![0-9_+=-])(?:[\d]{6})(?:[\d]{6,12})[\d ]?(?!\d))")

__email_regExp = re.compile(r"((?:(?:[a-z0-9+.']+)|(?:\"\w+\\ [a-z0-9']+\"))@"
                            r"(?:(?:[a-z0-9]+|\[)+(?:\.(?!\.+)))+(?:(?:[a-z0-9]+|\])+)?)", re.IGNORECASE)

__time_regExp = re.compile(r"(?:(?:上午|中午|下午|凌晨|早上|晚上|午夜|半夜)?(?:(?:(?:[0-1]\d|2[0-3]|(?<!\d)\d(?!\d))(?:点钟?|时))"
                           r"(?:(?:过?(?:(?:(?<!\d)\d(?!\d))|(?:(?<!\d)[0-5]\d(?!\d)))分)(?:过?(?:(?:(?<!\d)\d(?:\.\d+)?"
                           r"(?!\d))|(?:(?<!\d)[0-5]\d(?:\.\d+)?(?!\d)))秒)?|(?:一刻钟|一刻|半|多))?|"
                           r"(?:(?:(?:[零一二两三四五六七八九十]|(?:十|一十)[一二三四五六七八九]|二十[一二三四])(?:点钟?|时))"
                           r"(?:(?:(?:过?(?:(?:(?:(?:十|一十|二十|三十|四十|五十)[一二三四五六七八九]?)(?![一二两三四五六七八九十])?)"
                           r"|(?:(?<![一二两三四五六七八九十])零?[一二两三四五六七八九](?![一二两三四五六七八九十])))分)(?:过?"
                           r"(?:(?:(?:(?:十|一十|二十|三十|四十|五十)[一二三四五六七八九]?)(?![一二两三四五六七八九十])?)|"
                           r"(?:(?<![一二两三四五六七八九十])零?[一二两三四五六七八九](?![一二两三四五六七八九])))秒)?)|"
                           r"(?:过?(?:一刻钟|一刻|半|多)|(?:过?(?:(?:十|一十|二十|三十|四十|五十)[一二三四五六七八九]?)"
                           r"(?![一二两三四五六七八九十])?)|(?:(?<![一二两三四五六七八九十])过[一二两三四五六七八九]"
                           r"(?![一二两三四五六七八九]))))?)))|(?:(?<!\d)(?:[0-2]?[0-9][：:]+[0-5]?[0-9])[:：]?"
                           r"(?:[0-5]?[0-9]\.?[0-9]+)?(?:\s*(?:am|pm))?)|"
                           r"(?:(?:(?:一刻|半刻|半|零|一|二|两|三|四|五|六|七|八|九|十|百|千|万|百万|千万|亿|兆|\d)+"
                           r"(?:多个|多|个)?\s*(?:世纪|年|月|季度|日子|日|天|小时|分钟|秒钟|秒|毫秒|钟))|(?:[0-9]+\s*"
                           r"(?:years|year|months|month|days|day|hours|hour|hr\.|Minutes|Minute|second|secs\.|sec\.|"
                           r"Millisecond|msec\.|msel|ms)+))", re.IGNORECASE)

__date_regExp = re.compile(
    r"(?:(?<!\d)(?:3[01]|[12][0-9]|0?[1-9])(?!\d)[ /.-])(?:(?<!\d)(?:1[012]|0?[1-9])(?!\d)[ /.-])"
    r"(?:(?<!\d)(?:19|20)[0-9]{2}(?!\d))|(?:(?<!\d)(?:1[012]|0?[1-9])(?!\d)[ /.-])(?:(?<!\d)"
    r"(?:3[01]|[12][0-9]|0?[1-9])(?!\d)[ /.-])(?:(?<!\d)(?:19|20)[0-9]{2}(?!\d))|(?:(?<!\d)"
    r"(?:(?:19|20)[0-9]{2})(?!\d)[ 年/.-])(?:(?:(?<!\d)(?:1[012]|0?[1-9])(?!\d)[ 月/.-])"
    r"(?:(?<!\d)(?:3[01]|[12][0-9]|0?[1-9])(?!\d)[日|号]?))?|(?:(?<!\d)(?:1[012]|0?[1-9])(?!\d)"
    r"月)(?:(?:3[01]|[12][0-9]|0?[1-9])(?!\d)[日|号]?)")

__ip_regExp = re.compile(
    r"(?<!\d)(?:(?:2[0-5]\d)|(?:1\d{2})|(?:[1-9]\d)|(?:\d))\.(?:(?:2[0-5]\d)|(?:1\d{2})|(?:[1-9]\d)|"
    r"(?:\d))\.(?:(?:2[0-5]\d)|(?:1\d{2})|(?:[1-9]\d)|(?:\d))\.(?:(?:2[0-5]\d)|(?:1\d{2})|"
    r"(?:[1-9]\d)|(?:\d))(?!\d)")

__regExpSets = {"url": __url_regExp, "email": __email_regExp, "money": __money_regExp, "ip": __ip_regExp,
                "date": __date_regExp, "idcard": __idcard_regExp, "phnum": __phoneNumber_regExp,
                "bkcard": __bankCard_regExp, "time": __time_regExp}


def __cut(contents, regExpK=None, pos=False):
    """ 切分识别结果
    :param contents: [[txt],]
    :param regExpK: 正则表达式规则字典集合的关键字索引
    :param pos: 是否进行词性标注
    :return: [[word, label],]
    """
    retVal = []
    regExp = __regExpSets.get(regExpK, None)
    for i in range(len(contents)):  # 遍历输入List
        sub = contents[i]
        results = []
        if 1 == len(sub):  # 判断是否需要进行处理, 逻辑表达式结果为True则表示需要处理
            content = sub[0].strip()
            if 0 != len(content):  # 判断内容是否为空格符、占位符或回车换行符
                if isinstance(regExp, type(re.compile(""))):  # 判断是否需要正则匹配，True为进行正则匹配切分，False为进行Jieba切分
                    resultSet = regExp.findall(content)
                    # 根据正则表达式的匹配结果处理输入inStr
                    if len(resultSet) > 0:
                        post = content
                        for res in resultSet:
                            idx = post.find(res)
                            if idx is not None:
                                pre = post[:idx].strip()
                                if len(pre) > 0:
                                    results.append([pre])
                                if pos:
                                    results.append([res, regExpK])
                                else:
                                    results.append([res, "pos"])  # 不需要词性标注时，用“pos”占位
                                idx += len(res)
                                post = post[idx:].strip()
                        endPart = post.strip()
                        if len(endPart) > 0:
                            results.append([endPart])
                else:
                    # 分词处理
                    if pos:
                        results.extend([[item, pos] for item, pos in posseg.lcut(content)])
                    else:
                        results.extend([[item, "pos"] for item in jieba.lcut(content)])  # 不需要词性标注时，用“pos”占位
            else:
                if pos:
                    results.append([sub[0], "x"])
                else:
                    results.append([sub[0], "pos"])  # 不需要词性标注时，用“pos”占位
        if len(results) > 0:  # result > 0 表示有切分结果
            retVal.extend(results)
        elif sub:  # 遍历项不为空
            retVal.append(sub)

    return retVal


def __match(content, pos=False):
    """ 匹配识别
    :param content: [txt]
    :param pos: 是否进行词性标注
    :return: [(item, label?),]
    """

    # url处理
    step1 = __cut([[content]], regExpK="url", pos=pos)

    # email处理
    step2 = __cut(step1, regExpK="email", pos=pos)

    # money处理
    step3 = __cut(step2, regExpK="money", pos=pos)

    # idcard处理
    step4 = __cut(step3, regExpK="idcard", pos=pos)

    # bankcard处理
    step5 = __cut(step4, regExpK="bkcard", pos=pos)

    # date处理
    step6 = __cut(step5, regExpK="date", pos=pos)

    # time处理
    step7 = __cut(step6, regExpK="time", pos=pos)

    # phone处理
    step8 = __cut(step7, regExpK="phnum", pos=pos)

    # IpAddress处理
    step9 = __cut(step8, regExpK="ip", pos=pos)

    # 未标注内容的分词处理
    step10 = __cut(step9, pos=pos)

    # 修改时间词汇标记
    if pos:
        for i in range(len(step10)):
            if "t" == step10[i][-1] or "tg" == step10[i][-1]:
                step10[i][-1] = "time"
            elif "eng" == step10[i][-1]:
                if step10[i][0].isdigit():
                    step10[i][-1] = "m"
        retVal = step10
    else:
        retVal = [item[0] for item in step10]

    return retVal


match = __match


def buildTaggedTxtCorpus():
    # 数据库连接
    mysql = MysqlServer(host="10.0.0.247", db="db_pamodata", user="pamo", passwd="pamo")
    # 查询结果
    sql = "SELECT * FROM corpus_rawtxts WHERE txtLabel<>'电信诈骗' ORDER BY txtId LIMIT 100"
    # sql = "SELECT * FROM corpus_rawtxts ORDER BY txtId"
    queryResult = mysql.executeSql(sql=sql)

    # 切分标注
    queryResult = [(record[0], record[2], record[3].replace(" ", "")) for record in queryResult[1:]]
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    params = [[tid, txt] for tid, l, txt in queryResult]
    retVal = pool.map(match, params)
    pool.close()
    pool.join()
    raw_root = "../../Out/文本分类语料库/"
    print(raw_root)
    csv_header = ["word", "pos", "ner"]
    csv_data = []
    for i in retVal:
        for a in i:
            if isinstance(a, list):
                for s in a:
                    print(s)
            else:
                print(a)
        print("##########" * 15)
    pass


class BasicTextProcessing(object):
    """文本预处理类"""

    def __init__(self, **kwargs):
        self.results = None
        self.dict_gensim = None
        self.bow_bunch = None
        self.bow_gensim = None
        self.freqDictObj = defaultdict(int)
        jieba.setLogLevel(logging.INFO)
        userDict = kwargs.get("dict", None)
        newWords = kwargs.get("words", None)
        try:
            if userDict:
                jieba.set_dictionary(userDict)
                logger.info("Add custom dictionary successed.")
            if newWords:
                with open(newWords, "r", encoding="utf-8") as nw:
                    wordsSet = nw.readlines()
                    for line in wordsSet:
                        w, t = line.split()
                        jieba.add_word(word=w, tag=t.strip())
                        logger.info("Add word=%s(%s) freq : %s" % (w, t.strip(), str(jieba.suggest_freq(w))))
                    logger.info("Add new custom words finished.")

        except Exception as e:
            logger.error("Error:%s" % e)
            logger.warning("Use custom dictionary failed, use default dictionary.")

    def doWordSplit(self, content="", contents=(), pos=False):
        """
        :param content: [txt]
        :param contents: [txt,]
        :param pos: 是否进行词性标注 True=标注
        :return: self.results = [[(item, label?),],]
        """
        retVal = []
        if content:
            retVal.append(match(content, pos=pos))
        elif contents:
            for li in contents:
                retVal.append(match(li, pos=pos))
        else:
            logger.warning("None content for splitting word")
        self.results = retVal
        return self.results

    @staticmethod
    def batchWordSplit(contentList, pos=False):
        retVal = []
        if contentList:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            params = [(li, pos) for li in contentList]
            retVal = pool.starmap(match, params)
            pool.close()
            pool.join()
        else:
            logger.warning("None content for splitting word")
        for r in retVal:
            yield r

    def buildWordFrequencyDict(self, dataSets):
        """ 生成数据集中最小单元的频率字典
            :param dataSets: 输入数据集 --> [[column0,column1,],]
            :return: self.freqDictObj
        """
        for record in dataSets:
            for column in record:
                self.freqDictObj[column] += 1
        return self.freqDictObj

    def buildGensimDict(self, dataSets):
        """ 生成数据集的字典
            :param dataSets: 输入数据集 --> [[column0,column1,],]
            :return: self.dict_gensim
        """
        self.dict_gensim = corpora.Dictionary(dataSets)
        return self.dict_gensim

    def buildBow2Bunch(self, wordSeqs=None):
        """ 生成BOW的Bunch对象
            :param wordSeqs: 词序列集合 --> [[column,],]
            :return: self.bow_bunch
        """
        if wordSeqs is not None:
            self.bow_bunch = Bunch(txtIds=[], classNames=[], labels=[], contents=[])
            for record in wordSeqs:
                if isinstance(record, list):
                    self.bow_bunch.contents.append(" ".join(record))
                else:
                    logger.warning("wordSeqs 的内容结构错误！wordSeqs --> [[column,],]")
        else:
            logger.warning("wordSeqs is None！请输入正确的参数 wordSeqs --> [[column,],]")
        return self.bow_bunch

    def buildGensimCorpusByCorporaDicts(self, dataSets=None, dictObj=None):
        """ 生成语料库文件
            :param dataSets: 输入数据集 --> [[column0,column1,],]
            :param dictObj: Gensim字典对象 --> corpora.Dictionary
            :return: corpus --> [[(wordIndex,wordFreq),],]
        """
        corpus = None
        if isinstance(dictObj, corpora.Dictionary):
            corpus = [dictObj.doc2bow(record) for record in dataSets]
        elif isinstance(self.dict_gensim, corpora.Dictionary):
            corpus = [self.dict_gensim.doc2bow(record) for record in dataSets]
        else:
            logger.warning("非法的dictObj对象 (%s)，需要有效的 corpora.Dictionary对象！！！" % dictObj)
        return corpus


class TfidfVecSpace(object):
    """TFIDF向量空间的生成类"""

    def __init__(self):
        self.TFIDF_Train_Vecs = None
        self.TFIDF_Test_Vecs = None
        self.TFIDF_Model = None
        self.TFIDF_Vecs = None

    def buildVecs4Train(self, bowObj=None, dictObj=None):
        """ 生成训练集的TFIDF向量空间（Bunch对象）
            :param bowObj: Bunch(txtIds=[], classNames=[], labels=[], contents=[[],])
            :param dictObj: Gensim字典对象 --> corpora.Dictionary
            :return: self.TFIDF_Train_Vecs
        """
        if isinstance(bowObj, Bunch):
            self.TFIDF_Train_Vecs = Bunch(txtIds=[], classNames=[], labels=[], tdm=[], vocabulary=[])
            self.TFIDF_Train_Vecs.txtIds.extend(bowObj.txtIds)
            self.TFIDF_Train_Vecs.classNames.extend(bowObj.classNames)
            self.TFIDF_Train_Vecs.labels.extend(bowObj.labels)
            if isinstance(dictObj, corpora.Dictionary):
                self.TFIDF_Train_Vecs.vocabulary = dictObj.token2id
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                         vocabulary=self.TFIDF_Train_Vecs.vocabulary)  # 将测试集文本映射到训练集词典中
            self.TFIDF_Train_Vecs.tdm = vectorizer.fit_transform(bowObj.contents)
        else:
            logger.warning("参数bowObj 类型错误 (%s)！请输入正确的bowObj参数。" % bowObj)
        return self.TFIDF_Train_Vecs

    def buildVecs4Test(self, bowObj=None, trainTfidfObj=None):
        """ 生成测试集的TFIDF向量空间（Bunch对象）
            :param bowObj: Bunch(txtIds=[], classNames=[], labels=[], contents=[])
            :param trainTfidfObj: Bunch(txtIds=[], classNames=[], labels=[], tdm=[], vocabulary=[])
            :return: self.TFIDF_Test_Vecs
        """
        if isinstance(bowObj, Bunch):
            if isinstance(trainTfidfObj, Bunch):
                self.TFIDF_Test_Vecs = Bunch(txtIds=[], classNames=[], labels=[], tdm=[], vocabulary=[])
                self.TFIDF_Test_Vecs.txtIds.extend(bowObj.txtIds)
                self.TFIDF_Test_Vecs.classNames.extend(bowObj.classNames)
                self.TFIDF_Test_Vecs.labels.extend(bowObj.labels)

                self.TFIDF_Test_Vecs.vocabulary = trainTfidfObj.vocabulary
                vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                             vocabulary=trainTfidfObj.vocabulary)  # 将测试集文本映射到训练集词典中
                self.TFIDF_Test_Vecs.tdm = vectorizer.fit_transform(bowObj.contents)
        else:
            logger.warning("参数bowObj 类型错误 (%s)！请输入正确的bowObj参数。" % bowObj)
        return self.TFIDF_Test_Vecs

    def buildVecsByGensim(self, **kwargs):
        """ 生成数据集的TFIDF向量空间
            :param kwargs: 初始化TFIDF向量工具模型的数据 initCorpus --> [[doc2bow的处理结果(wordIndex,wordFreq),],]
            :param kwargs: record --> [] or corpus --> [[],]
            :return: self
        """
        initCorpus = kwargs.get("initCorpus", None)
        record = kwargs.get("record", None)
        corpus = kwargs.get("corpus", None)
        if initCorpus is not None:
            self.TFIDF_Model = models.TfidfModel(initCorpus)
            logger.info("Build TFIDF Model Successed")
            if isinstance(record, list):
                self.TFIDF_Vecs = self.TFIDF_Model[record]
            elif isinstance(corpus, list):
                self.TFIDF_Vecs = self.TFIDF_Model[corpus]
            else:
                logger.warning("Build TFIDF Vector Spaces Failed (record=%s, corpus=%s)" % (record, corpus))
        else:
            logger.warning("参数initCorpus 类型错误 (%s)！请输入正确的initCorpus参数。" % initCorpus)

        return self


def tst(btp, txts):  # 功能测试
    btp.doWordSplit(content=txts[0])
    print("*********" * 15)
    for l in btp.results:
        print(l)
    btp.doWordSplit(contents=txts)
    print()
    print("*********" * 15)
    for l in btp.results:
        print(l)
    r = btp.batchWordSplit(contentList=txts)
    print()
    print("*********" * 15)
    for l in list(r):
        print(l)

    btp.doWordSplit(content=txts[0], pos=True)
    print()
    print(">>>>>>>>>" * 15)
    for l in btp.results:
        print(l)
    btp.doWordSplit(contents=txts, pos=True)
    print()
    print(">>>>>>>>>" * 15)
    for l in btp.results:
        print(l)
    r = btp.batchWordSplit(contentList=txts, pos=True)
    print()
    print(">>>>>>>>>" * 15)
    for l in list(r):
        print(l)


def main():
    # 数据库连接
    mysql = MysqlServer(host="10.0.0.247", db="db_pamodata", user="pamo", passwd="pamo")

    # 查询结果
    # sql = "SELECT * FROM corpus_rawtxts WHERE txtLabel<>'电信诈骗' ORDER BY txtId LIMIT 100"
    sql = "SELECT * FROM corpus_rawtxts ORDER BY txtId LIMIT 3"
    queryResult = mysql.executeSql(sql=sql)
    queryResult = [(record[0], record[2], record[3].replace("\r\n", "").replace("\n", ""))
                   for record in queryResult[1:]]
    ids = [r[0] for r in queryResult]
    labels = [r[1] for r in queryResult]
    txts = [r[2] for r in queryResult]

    # 切分标注
    userDict = "../../Dicts/dict_jieba_check.txt"
    newWords = "../../Dicts/newWords.txt"

    # 功能类测试
    btp = BasicTextProcessing(dict=userDict, words=newWords)
    tst(btp, txts)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
