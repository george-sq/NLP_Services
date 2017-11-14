# -*- coding: utf-8 -*-
"""
    @File   : pamo_1.py
    @Author : NLP_QingShen (275171387@qq.com)
    @Time   : 2017/10/23 9:56
    @Todo   : 
"""

import time
from collections import defaultdict
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import jieba
import jieba.posseg as psg
from gensim import corpora
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import dataBases as db
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


########################################################################################################################
"""
    加载原始数据
"""


def loadFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fr:
        contents = fr.read()
        contents = contents.split('\n')
        return contents


@timeCost
def getStopWords():
    stopWords = []
    stopWords_EN = loadFile('./StopWords/stopWords_EN.txt')
    stopWords_CN = loadFile('./StopWords/stopWords_CN.txt')
    stopWords.extend(stopWords_EN)
    stopWords.extend(stopWords_CN)
    stopWords.append(' ')
    # stopWords.append('　')
    return set(stopWords)


@timeCost
def getRawGenerator(stopWords):
    """
        :return: results --> generator object [rowNum,
            rows:[(txtid,txtname,txtlabel,txtcontent,txtsegword,inserttime,modtime),]]
    """
    raws = gd.LoadData().getRawCorpus()
    # retList = []
    for raw in raws:
        yield raw[2], raw[3], stopWords
        #     retList.append((raw[2], raw[3]))
        # return retList


# @timeCost
def getRawByLabelGenerator(sLabel):
    """
        :return: results --> generator object [rowNum,
            rows:[(txtid,txtname,txtlabel,txtcontent,txtsegword,inserttime,modtime),]]
    """
    raws = db.useMysql().executeSql("SELECT * FROM tb_txtcate WHERE txtLabel='%s' ORDER BY txtId" % sLabel)
    retList = []
    for raw in raws:
        retList.append((raw[2], raw[3]))
    return retList


########################################################################################################################
"""
    a、脏数据清洗
        1)、长度非法数据清洗
        2)、纯英文数据清洗
        3)、中英混合数据清洗
"""


def dataClean():
    pass


########################################################################################################################
"""
    b、分词和词性标注
        1)、中文处理
        2)、英文处理
        2)、构建词典
"""


def doWordCut(inStr):
    """
        :param inStr:
        :return: list --> [word,]
    """
    words = jieba.cut(inStr)
    return list(words)


def doPosCut(inStr):
    """
        :param inStr:
        :return: list --> [(word + '\t' + pos),]
    """
    wps = psg.cut(inStr)
    ret = []
    for w, p in wps:
        ret.append(w + '\t' + p)
    return ret


########################################################################################################################
"""
    c、构建数据集
"""


def buildWordSeq(argsTuple):
    """
        :param argsTuple: (fileInfo:(label,text),stopWords:set)
        :return: words --> (label,words)
    """
    label = argsTuple[0]
    text = argsTuple[1]
    # stopWords = argsTuple[2]
    words = doWordCut(text.strip().replace('\n', ''))
    words = [word for word in words]
    # yield label, words
    return label, words


########################################################################################################################
"""
    d、词频统计
"""


class StatData(object):
    def __init__(self):
        self.dictionary = None
        self.wordFreqs = None
        self.labels = None
        self.txts = None

    @timeCost
    def statFreq(self, _txtCorpus):
        """
            :param _txtCorpus: IMapIterator object --> (label, words)
            :return:
        """
        self.dictionary = corpora.Dictionary()
        self.wordFreqs = defaultdict(int)
        self.labels = []
        self.txts = []
        for lab, words in _txtCorpus:
            self.labels.append(lab)
            for token in words:
                self.wordFreqs[token] += 1
            self.dictionary.add_documents([words])
            self.txts.append(words)
        return self

    @timeCost
    def storeAllStatData(self):
        """
            :return:
        """
        if self.dictionary is not None and self.wordFreqs is not None:
            lines = []
            for word in self.dictionary.token2id:
                wid = self.dictionary.token2id[word]
                dfss = self.dictionary.dfs[wid]
                freq = self.wordFreqs[word]
                line = 'id:%s\tword:%s\tdfs:%s\twordfreq:%s\n' % (wid, word, dfss, freq)
                lines.append(line)

            filePath = './Out/StatFiles/statData_' + str(time.time()).split('.')[0] + '.txt'
            with open(filePath, 'w', encoding='utf-8') as fw:
                fw.writelines(lines)
        else:
            print('未统计数据，请先行统计数据。')

    @timeCost
    def storeWordFreq(self):
        """
            :return:
        """
        if self.dictionary is not None and self.wordFreqs is not None:
            outWordFreq = []
            wf = sorted(self.wordFreqs.items(), key=lambda twf: twf[1], reverse=True)
            for w, f in wf:
                outWordFreq.append(str(w) + '\t' + str(f) + '\n')
            with open('./Out/StatFiles/wordFrequencys_' + str(time.time()).split('.')[0] + '.txt', 'w',
                      encoding='utf-8') as fw:
                fw.writelines(outWordFreq)
        else:
            print('未统计数据，请先行统计数据。')


########################################################################################################################
@timeCost
def doMultiProcess(func, argsItr):
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>创建进程池。')
    pool = Pool(multiprocessing.cpu_count())
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>初始化进程池完成。')
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>分配子进程任务。')
    results = pool.imap(func, argsItr)
    pool.close()
    pool.join()
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>子进程任务完成。')
    return results


@timeCost
def buildWordCloudWithFreq(dicts, imgName):
    """
        :param dicts: dict from string to float. {'word':freq,}
        :param imgName:
        :return:
    """
    # Generate a word cloud image
    wordcloud = WordCloud(max_words=2000, width=1300, height=600, background_color="white",
                          font_path='C:/Windows/Fonts/STSONG.TTF').generate_from_frequencies(
        dicts)

    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # nowTime = time.time()
    nowTime = str(time.time()).split('.')[0]
    # print(nowTime)
    wordcloud.to_file('./Out/WordCloudImgs/' + imgName + '_' + nowTime + '.png')
    # plt.show()


########################################################################################################################
"""
    <<主函数>>
"""


@timeCost
def main():
    """
        1、数据预处理        a、脏数据清洗    b、分词和词性标注    c、去停用词    d、数据统计
        2、数据分析          a、标准化输入输出    b、数据向量化    c、构建任务算法    d、算法性能评估
        3、抽象模块          a、封装输入输出模块    b、封装算法模型
        4、工具化            a、模块集成    b、流程工具化
    """

    # 0、加载语料
    stopWords = getStopWords()
    gRawCorpus = getRawGenerator(stopWords)

    # 1、分词和词性标注
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>创建进程池。')
    pool = Pool(multiprocessing.cpu_count())
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>初始化进程池完成。')
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>分配子进程任务。')
    gTxtCorpus = pool.imap(buildWordSeq, gRawCorpus)
    pool.close()
    pool.join()
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>子进程任务完成。')

    # 2、数据统计
    statTool = StatData()
    statTool.statFreq(gTxtCorpus)  # dictionary, frequency, labels, txts
    # dicts = statTool.dictionary

    statTool.storeAllStatData()
    statTool.storeWordFreq()

    # 3、画词云图片
    buildWordCloudWithFreq(statTool.wordFreqs, 'pamo_wordcloud')

    # 4、构建词典
    statTool.dictionary.save('./Out/Dicts/pamo_dicts.dict')

    # 5、构建词包空间
    corpus = [statTool.dictionary.doc2bow(text) for text in statTool.txts]
    labels = statTool.labels
    classSets = {'非电诈相关': 0, '涉嫌电诈': 1}
    docs = []
    if len(corpus) == len(labels):
        len_corpus = len(corpus)
        for i in range(len_corpus):
            # lab = labels[i]
            # txt = corpus[i]
            if labels[i] != '电信诈骗':
                docs.append((classSets['非电诈相关'], corpus[i]))
            else:
                docs.append((classSets['涉嫌电诈'], corpus[i]))
            pass
    else:
        print('文档数目与文档类型数目不一致。')

    # 6、构建语料库
    corpora.MmCorpus.serialize('./Out/Corpus/pamo_gaCorpus.mm', corpus=corpus)


if __name__ == '__main__':
    main()
