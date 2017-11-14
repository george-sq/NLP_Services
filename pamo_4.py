# -*- coding: utf-8 -*-
"""
    @File   : pamo_4.py
    @Author : NLP_QingShen (275171387@qq.com)
    @Time   : 2017/11/2 16:10
    @Todo   : 将tb_tinfo表中未预处理的文本进行分词和关键词处理等预处理，并将预处理结果插入到tb_tstructinfo表中
"""

import os
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import datetime
import time
import pymysql
from jieba import posseg as posg
from gensim import corpora
from gensim import models
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def tstData():
    """准备测试数据"""
    dbs = UseMysql()
    doc1 = "今年第26次！中国海警船今日在钓鱼岛领海内巡航！海外网11月2日电2日，中国海警2308、2502、2113、2302舰船编队在我钓鱼岛领海内巡航。" \
           "日媒称，当地时间上午10点10分左右，4艘中国海警船相继驶入钓鱼岛海域，中国海警船上次在钓鱼岛领海内巡航是10月5日，今年进入领海内的次数达26天。"
    doc2 = "8月初，陆某在网上找兼职刷单工作,通过QQ与对方联系后,按对方的指引下单选定商品，将未付款的订单截图发给对方。" \
           "对方要求其通过支付宝扫描发来的二维码进行付款，刷完第一单小额订单后对方按照约定返还。陆某开始相信对方，随后开始刷第二单，" \
           "扫描支付完后对方又称第二单有三个任务且每个任务内含多个商品，要求完成三个任务后方可返还本金和佣金。"
    docs = [doc1, doc2]
    params = []
    for d in docs:
        now = datetime.datetime.now()
        param = [d, 2, now]
        params.append(param)
    insertSql = 'INSERT INTO tb_tinfo(content, tsrc, modtime)VALUES(%s, %s, %s)'
    dbs.executeSql(insertSql, params)


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
            :return: results = [受影响的行数, (行内容,),]
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

class DocInfos(object):
    def __init__(self):
        self.dbs = UseMysql()
        self.localDict = corpora.Dictionary.load('./Out/Dicts/pamo_dicts.dict')
        self.localCorpus = corpora.MmCorpus('./Out/Corpus/pamo_gaCorpus.mm')
        self.tfidfModel = models.TfidfModel(self.localCorpus)

    def getNewTxtStruct(self):
        sql = "SELECT * FROM tb_tinfo WHERE tid NOT IN (SELECT tid FROM tb_structinfo)"
        result = self.dbs.executeSql(sql)
        return result

    def getNewTxtCate(self):
        sql = "SELECT * FROM tb_tinfo WHERE tid NOT IN (SELECT tid FROM tb_tcate)"
        result = self.dbs.executeSql(sql)
        return result

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
            kwseqs = dataConvertor(self.localDict, self.tfidfModel[txtBow])
            kwseqs = sorted(kwseqs, key=lambda tt: tt[1], reverse=True)
            kwseqs = [(k, str(w)) for k, w in kwseqs]
            strKwseqs = '\t'.join(['|'.join(kw) for kw in kwseqs])
            imgPath = buildWordCloudWithFreq(dict(kwseqs), str(tid))
            # 拼接Insert SQL语句的插入参数
            params.append((strWps, strKwseqs, tid, imgPath, datetime.datetime.now()))

        result = self.dbs.executeSql(
            "INSERT INTO tb_structinfo(pos, kwseqs, tid, imgpath, modtime) VALUES (%s, %s, %s, %s, %s)", params)
        return result


########################################################################################################################
def main():
    # tstData()
    # 1、获取未处理的文本 tb_tinfo
    docTool = DocInfos()
    raws = docTool.getNewTxtStruct()

    # 2、文本预处理 tb_structinfo
    #     2.1 分词和词性标注
    #     2.2 抽取关键词集合
    #         2.2.1 加载词典
    #         2.2.3 TFIDF权值
    docTool.insertStructinfo(raws[1:])
    # 3、文本分类 tb_tcate


if __name__ == '__main__':
    main()
    pass
