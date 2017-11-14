# -*- coding: utf-8 -*-
"""
    @File   : 1.py
    @Author : NLP_QingShen (275171387@qq.com)
    @Time   : 2017/9/30 10:38
    @Todo   :
                1、数据预处理
                    a、脏数据清洗
                        1)、长度非法数据清洗
                        2)、纯英文数据清洗
                        3)、中英混合数据清洗
                    b、分词和词性标注
                        1)、中文处理
                        2)、英文处理
                        2)、构建词典
                    c、去停用词
                        1)、中文停用词处理
                        2)、英文停用词处理
                    d、数据统计
                        1)、全数据集统计
                        2)、子类数据集统计
"""

import os
import time
from collections import Iterable, Iterator
from collections import defaultdict
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import jieba
import jieba.posseg as psg
from gensim import corpora
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Process
from multiprocessing import Pool
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import dataBases as db
import getData as gd


class timeLog(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        # print( "Entering", self.func.__name__)
        # self.func()
        # print("Exited", self.func.__name__)
        startTime = time.time()
        result = self.func(*args, **kwargs)
        useTime = time.time() - startTime
        print('>>>>>' * 30)
        print('     ' * 10, '[ %s %s ] >>> cost time %s s' % (self.func.__name__, self.func.__class__, useTime))
        print('<<<<<' * 30)
        return result


def timeCost(func):
    def cost(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        useTime = time.time() - startTime
        logging.info('>>>>' * 30)
        logging.info('\t\t\t\t\t\t\t\t%s  [ %s ]  cost time  %s s' % (
            func.__class__, func.__name__, useTime))
        logging.info('<<<<' * 30)
        # print('>>>>>' * 30)
        # print('      ' * 6, ' %s [ %s ]  cost time  %s s:' % (func.__class__, func.__name__, useTime))
        # print('<<<<<' * 30)
        # print()
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
def getRawGenerator():
    """
        :return: results --> generator object [rowNum,rows:[(txtid,txtname,txtlabel,txtcontent,txtsegword,inserttime,modtime),]]
    """
    raws = gd.LoadData().getRawCorpus()
    stopWords = getStopWords()
    for raw in raws:
        yield (raw[3], stopWords)


# @timeCost
def getRawByLabelGenerator(sLabel):
    """
        :return: results --> generator object [rowNum,rows:[(txtid,txtname,txtlabel,txtcontent,txtsegword,inserttime,modtime),]]
    """
    # sql = "SELECT * FROM tb_txtcate WHERE txtLabel='%s' ORDER BY txtId"%sLabel
    raws = db.useMysql().executeSql("SELECT * FROM tb_txtcate WHERE txtLabel='%s' ORDER BY txtId" % sLabel)
    stopWords = getStopWords()
    # alist = []
    for raw in raws:
        # alist.append((raw[3], stopWords))
        yield (raw[3], stopWords)
        # return alist


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
        1)、中文停用词处理
        2)、英文停用词处理
"""


def buildWordSeq(argsTuple):
    """
        :param argsTuple: (text,stopWords)
        :return: words --> [word,]
    """
    text = argsTuple[0]
    stopWords = argsTuple[1]
    # fileId = str(dataRow[0])
    # fileLabel = dataRow[2]
    words = doWordCut(text.strip().replace('\n', ''))
    return [word for word in words if word not in stopWords]


def buildPosSeq(dataRow):
    """
        :param dataRow:
        :return: outfile --> fileId, fileLabel, fileContents
    """
    fileId = str(dataRow[0])
    fileLabel = dataRow[2]
    txtName = fileId + '-' + fileLabel + '.txt'
    fileContents = doPosCut(dataRow[3].strip())
    with open('./Out/CorpusPos/' + txtName, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fileContents))


########################################################################################################################
"""
    d、词频统计
"""


@timeCost
def statFreq(_txtCorpus):
    txts = list(_txtCorpus)
    frequency = defaultdict(int)
    dictionary = corpora.Dictionary(txts)
    for text in txts:
        # dictionary.add_documents([text])
        for token in text:
            frequency[token] += 1
    return dictionary, frequency, txts


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


def buildWordCloud(text):
    """
        :param text: [空格符分割的词序列]
        :return:
    """
    # Generate a word cloud image
    wordcloud = WordCloud(background_color="white", font_path='C:/Windows/Fonts/STSONG.TTF').generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


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

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    wordcloud.to_file('./Out/' + imgName)
    # plt.show()


def storeAllStatData(statData):
    dictionary = statData[0]
    wordFreq = statData[1]
    lines = []

    for wd in dictionary.token2id:
        wid = dictionary.token2id[wd]
        word = wd
        dfss = dictionary.dfs[wid]
        freq = wordFreq[word]
        line = 'id:%s\tword:%s\tdfs:%s\twordfreq:%s\n' % (wid, word, dfss, freq)
        lines.append(line)

    with open('./Out/statData.txt', 'w', encoding='utf-8') as fw:
        fw.writelines(lines)


def storeWordFreq(statData, fileName):
    wordFreq = statData[1]
    outWordFreq = []
    wordFreq = sorted(wordFreq.items(), key=lambda twf: twf[1], reverse=True)
    for w, f in wordFreq:
        outWordFreq.append(str(w) + '\t' + str(f) + '\n')
    with open('./Out/wordFrequencys_' + fileName + '.txt', 'w', encoding='utf-8') as fw:
        fw.writelines(outWordFreq)


########################################################################################################################
"""
    <<主函数>>
"""


def main():
    """
        1、数据预处理        a、脏数据清洗    b、分词和词性标注    c、去停用词    d、数据统计
        2、数据分析          a、标准化输入输出    b、数据向量化    c、构建任务算法    d、算法性能评估
        3、抽象模块          a、封装输入输出模块    b、封装算法模型
        4、工具化            a、模块集成    b、流程工具化
    """

    # 0、加载语料
    gRawCorpus = getRawGenerator()
    gLabelCorpus = getRawByLabelGenerator('电信诈骗')

    # 1、分词和词性标注
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>创建进程池。')
    pool = Pool(multiprocessing.cpu_count())
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>初始化进程池完成。')
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>分配子进程任务。')
    gTxtCorpus = pool.imap(buildWordSeq, gRawCorpus)
    gLabelTxtCorpus = pool.imap(buildWordSeq, gLabelCorpus)
    pool.close()
    pool.join()
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>子进程任务完成。')
    # gTxtCorpus = doMultiProcess(buildWordSeq, gRawCorpus)
    # gLabelTxtCorpus = doMultiProcess(buildWordSeq, gLabelCorpus)

    # 2、词频统计
    statResult = statFreq(gTxtCorpus)
    dictionary = statResult[0]
    labelStatResult = statFreq(gLabelTxtCorpus)

    storeAllStatData(statResult)
    storeWordFreq(statResult, 'all')
    storeWordFreq(labelStatResult, '电信诈骗')

    # 画词云图片
    subProcess = Process(target=buildWordCloudWithFreq, args=(statResult[1], 'pamo_wordcloud.png'))
    subProcess.start()
    subProcess1 = Process(target=buildWordCloudWithFreq, args=(labelStatResult[1], 'pamo_wordcloud_电信诈骗.png'))
    subProcess1.start()

    # 3、构建词典
    dictionary.save('./Out/pamo_nlp.dict')

    # 4、构建语料库
    corpus = [dictionary.doc2bow(text) for text in statResult[2]]
    corpora.MmCorpus.serialize('./Out/pamo_nlp_gaCorpus.mm', corpus)


if __name__ == '__main__':
    main()
