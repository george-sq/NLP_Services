# -*- coding: utf-8 -*-
"""
    @File   : fileServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/26 15:56
    @Todo   : 
"""

import os
import pickle
from wordcloud import WordCloud
import logging
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim

logger = logging.getLogger(__name__)


class FileServer(object):
    """关于文件操作的服务类"""

    def __init__(self):
        self.filecode = "utf-8"
        self.pick = pickle

    @staticmethod
    def __checkPathArgType(path):
        """ 检验路径参数的类型
            :param path: (str)文件目录路径
            :return: boolean
        """
        try:
            if not isinstance(path, str):
                raise TypeError
        except TypeError:
            logger.error("TypeError: 文件路径参数的类型错误 (%s) ！！！" % path)
            return False
        else:
            return True

    def __checkPath(self, path):
        """ 校验文件路径参数的有效性
            :param path: (str)文件目录路径
            :return: True or "MK"
        """
        # 路径校验
        if os.path.exists(path):
            return True
        else:
            dn, bn = os.path.split(path)
            r = self.__checkPath(dn)
            if r or "MK" == r:
                mkPath = os.path.join(dn, bn)
                if not os.path.exists(mkPath):
                    os.mkdir(mkPath)
                return "MK"

    def loadTextByUTF8(self, path, fileName):
        """ 加载本地Text文件
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :return: fileContent or None
        """
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            try:
                # 加载文件
                with open(fullName, 'r', encoding=self.filecode) as txtr:
                    logger.debug("Loaded TXT File(UTF-8) Successed")
                    return txtr.read()
            except FileNotFoundError:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
                return None
        else:
            return None

    def loadPickledObjFile(self, path, fileName):
        """ Read a pickled object representation from the open file
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :return: fileContent or None
        """
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            try:
                # 加载文件
                with open(fullName, 'rb') as ldf:
                    logger.debug("Loaded Pickled File Successed")
                    return self.pick.load(ldf)
            except FileNotFoundError:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
                return None
        else:
            return None

    def loadLocalMmCorpus(self, path, fileName):
        """ 加载gensim   模块生成的本地语料库文件
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :return: fileContent or None
        """
        retVal = None
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            # 文件目录路径校验
            if os.path.exists(path) and os.path.isfile(fullName):
                # 加载文件
                retVal = gensim.corpora.MmCorpus(fullName)
                if isinstance(retVal, gensim.corpora.MmCorpus):
                    logger.debug("Loaded gensim.corpora.MmCorpus File Successed")
                else:
                    retVal = None
                    logger.warning("Loaded gensim.corpora.MmCorpus File Failed")
            else:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
        return retVal

    def loadLocalGensimDict(self, path, fileName):
        """ 加载gensim模块生成的本地字典文件
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :return: fileContent or None
        """
        retVal = None
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            # 文件目录路径校验
            if os.path.exists(path) and os.path.isfile(fullName):
                # 加载文件
                retVal = gensim.corpora.Dictionary.load(fullName)
                if isinstance(retVal, gensim.corpora.Dictionary):
                    logger.debug("Loaded Gensim.Dictionary File Successed")
                else:
                    retVal = None
                    logger.warning("Loaded Gensim.Dictionary File Failed")
            else:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
        return retVal

    def loadWord2VectorModel(self, path, fileName):
        retVal = None
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            # 文件目录路径校验
            if os.path.exists(path) and os.path.isfile(fullName):
                # 加载文件
                retVal = gensim.models.Word2Vec.load(fullName)
                if isinstance(retVal, gensim.models.Word2Vec):
                    logger.debug("Loaded Gensim.Word2VectorModel File Successed")
                else:
                    retVal = None
                    logger.warning("Loaded 本地Gensim.Word2VectorModel File Failed")
            else:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
        return retVal

    def loadIndex4tfidfSimilarity(self, path, fileName):
        retVal = None
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            # 文件目录路径校验
            if os.path.exists(path) and os.path.isfile(fullName):
                # 加载文件
                retVal = gensim.similarities.SparseMatrixSimilarity.load(fullName)
                if isinstance(retVal, gensim.similarities.SparseMatrixSimilarity):
                    logger.debug("Loaded Similarity Index of TFIDF Successed")
                else:
                    retVal = None
                    logger.warning("Loaded Similarity Index of TFIDF Failed")
            else:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
        return retVal

    def loadGensimTfidfModel(self, path, fileName):
        retVal = None
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            # 文件目录路径校验
            if os.path.exists(path) and os.path.isfile(fullName):
                # 加载文件
                retVal = gensim.models.TfidfModel.load(fullName)
                if isinstance(retVal, gensim.models.TfidfModel):
                    logger.debug("Loaded TFIDF Model Successed")
                else:
                    retVal = None
                    logger.warning("Loaded TFIDF Model Failed")
            else:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
        return retVal

    def loadIndex4topicSimilarity(self, path, fileName):
        retVal = None
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            # 文件目录路径校验
            if os.path.exists(path) and os.path.isfile(fullName):
                # 加载文件
                retVal = gensim.similarities.MatrixSimilarity.load(fullName)
                if isinstance(retVal, gensim.similarities.MatrixSimilarity):
                    logger.debug("Loaded Similarity Index of Topics Model Successed")
                else:
                    retVal = None
                    logger.warning("Loaded Similarity Index of Topics Model Failed")
            else:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
        return retVal

    def loadGensimLsiModel(self, path, fileName):
        retVal = None
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            # 文件目录路径校验
            if os.path.exists(path) and os.path.isfile(fullName):
                # 加载文件
                retVal = gensim.models.LsiModel.load(fullName)
                if isinstance(retVal, gensim.models.LsiModel):
                    logger.debug("Loaded Lsi Topics Model Successed")
                else:
                    retVal = None
                    logger.warning("Loaded Similarity Index of Topics Model Failed")
            else:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
        return retVal

    def loadGensimLdaModel(self, path, fileName):
        retVal = None
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            # 文件目录路径校验
            if os.path.exists(path) and os.path.isfile(fullName):
                # 加载文件
                retVal = gensim.models.LsiModel.load(fullName)
                if isinstance(retVal, gensim.models.LdaModel):
                    logger.debug("Loaded Lda Topics Model Successed")
                else:
                    retVal = None
                    logger.warning("Loaded Similarity Index of Topics Model Failed")
            else:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
        return retVal

    def saveText2UTF8(self, path, fileName, **kwargs):
        """
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :param kwargs: 写入内容 --> content or lines
            :return: boolean
        """
        retVal = False
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            # 文件目录路径校验
            roc = self.__checkPath(path)
            if "MK" == roc:
                logger.info("新建文件路径 (%s)" % path)
            fullName = os.path.join(path, fileName)
            content = kwargs.get("content", None)
            lines = kwargs.get("lines", [])
            try:
                # 内容写入
                with open(fullName, "w", encoding=self.filecode) as txtw:
                    # 写入内容校验
                    if content is not None:
                        txtw.write(content)
                    elif len(lines) > 0:
                        txtw.writelines(lines)
                    else:
                        raise ValueError
                    logger.debug("Saved TXT File(UTF-8) Successed")
                    retVal = True
            except FileNotFoundError:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
            except ValueError:
                logger.error("**kwargs参数错误(content=%s, lines=%s), 需要给定参数content 或者 参数lines" % (content, lines))
        return retVal

    def savePickledObjFile(self, path, fileName, writeContentObj=None):
        """
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :param writeContentObj: 写入内容对象 eg. Bunch()
            :return: boolean
        """
        retVal = False
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            # 文件目录路径校验
            roc = self.__checkPath(path)
            if "MK" == roc:
                logger.info("新建文件路径 (%s)" % path)
            fullName = os.path.join(path, fileName)
            try:
                # 写入内容校验
                if writeContentObj is not None:
                    # 内容写入
                    with open(fullName, "wb")as pobj:
                        self.pick.dump(writeContentObj, pobj)
                        logger.debug("Saved Pickled File Successed(%s)" % pobj)
                        retVal = True
            except FileNotFoundError:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
        return retVal

    def saveIndex4tfidfSimilarity(self, path, fileName, index=None):
        retVal = False
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            # 文件目录路径校验
            roc = self.__checkPath(path)
            if "MK" == roc:
                logger.info("新建文件路径 (%s)" % path)
            fullName = os.path.join(path, fileName)
            try:
                # 写入内容校验
                if isinstance(index, gensim.similarities.SparseMatrixSimilarity):
                    # 内容写入
                    index.save(fullName)
                    logger.debug("Saved Similarity Index of TFIDF Successed(%s)" % index)
                    retVal = True
            except FileNotFoundError:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
        return retVal

    def saveIndex4topicSimilarity(self, path, fileName, index=None):
        retVal = False
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            # 文件目录路径校验
            roc = self.__checkPath(path)
            if "MK" == roc:
                logger.info("新建文件路径 (%s)" % path)
            fullName = os.path.join(path, fileName)
            try:
                # 写入内容校验
                if isinstance(index, gensim.similarities.MatrixSimilarity):
                    # 内容写入
                    index.save(fullName)
                    logger.debug("Saved Similarity Index of Topics Model Successed(%s)" % index)
                    retVal = True
            except FileNotFoundError:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
        return retVal

    def saveTopicModel(self, path, fileName, tmodel=None):
        retVal = False
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            # 文件目录路径校验
            roc = self.__checkPath(path)
            if "MK" == roc:
                logger.info("新建文件路径 (%s)" % path)
            fullName = os.path.join(path, fileName)
            try:
                # 写入内容校验
                if isinstance(tmodel, gensim.models.LdaModel) or isinstance(tmodel, gensim.models.LsiModel):
                    # 内容写入
                    tmodel.save(fullName)
                    logger.debug("Saved Topics Model Successed(%s)" % tmodel)
                    retVal = True
            except FileNotFoundError:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)

        return retVal

    def saveGensimTfidfModel(self, path, fileName, tfidf=None):
        retVal = False
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            # 文件目录路径校验
            roc = self.__checkPath(path)
            if "MK" == roc:
                logger.info("新建文件路径 (%s)" % path)
            fullName = os.path.join(path, fileName)
            try:
                # 写入内容校验
                if isinstance(tfidf, gensim.models.TfidfModel):
                    # 内容写入
                    tfidf.save(fullName)
                    logger.debug("Saved TFIDF Model Successed(%s)" % tfidf)
                    retVal = True
            except FileNotFoundError:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)

        return retVal

    def saveWord2VectorModel(self, path, fileName, wvmodel=None):
        retVal = False
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            # 文件目录路径校验
            roc = self.__checkPath(path)
            if "MK" == roc:
                logger.info("新建文件路径 (%s)" % path)
            fullName = os.path.join(path, fileName)
            try:
                # 写入内容校验
                if isinstance(wvmodel, gensim.models.Word2Vec):
                    # 内容写入
                    wvmodel.save(fullName)
                    logger.debug("Saved gensim.models.Word2Vec Model File Successed(%s)" % wvmodel)
                    retVal = True
            except FileNotFoundError:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)

        return retVal

    def saveGensimCourpus2MM(self, path, fileName, **kwargs):
        """
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :param kwargs: inCorpus写入语料库内容对象 --> [[(,),],]
            :return: boolean
        """
        retVal = False
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            # 文件目录路径校验
            roc = self.__checkPath(path)
            if "MK" == roc:
                logger.info("新建文件路径 (%s)" % path)
            fullName = os.path.join(path, fileName)
            inCorpus = kwargs.get("inCorpus", None)
            try:
                # 写入内容参数校验
                if inCorpus is not None:
                    # 内容写入
                    gensim.corpora.MmCorpus.serialize(fullName, corpus=inCorpus)
                    logger.debug("Saved GensimCourpus2MM File Successed")
                    retVal = True
                else:
                    raise ValueError
            except FileNotFoundError:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
            except ValueError:
                logger.error("**kwargs参数错误, 缺少写入内容, inCorpus = %s" % inCorpus)
        return retVal

    def saveGensimDict(self, path, fileName, dicts=None):
        """
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :param dicts: 写入字典对象 --> gensim.corpora.Dictionary()
            :return: boolean
        """
        retVal = False
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            # 文件目录路径校验
            roc = self.__checkPath(path)
            if "MK" == roc:
                logger.info("新建文件路径 (%s)" % path)
            fullName = os.path.join(path, fileName)
            try:
                # 写入内容参数校验
                if dicts is not None:
                    if isinstance(dicts, gensim.corpora.Dictionary):
                        # 内容写入
                        dicts.save(fullName)
                        logger.debug("Saved GensimDict File Successed(%s)" % dicts)
                        retVal = True
                    else:
                        raise TypeError
            except FileNotFoundError:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
            except TypeError:
                logger.error("TypeError: 错误的字典类型 (dicts = %s)！！！" % dicts)
        return retVal

    def buildWordCloudWithFreq(self, path, fileName, dicts=None):
        """
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :param dicts: {'word':freq,}
            :return: boolean
        """
        retVal = False
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            # 文件目录路径校验
            roc = self.__checkPath(path)
            if "MK" == roc:
                logger.info("新建文件路径 (%s)" % path)
            fullName = os.path.join(path, fileName)
            try:
                # 写入内容参数校验
                if dicts is not None:
                    # Generate a word cloud image
                    wordcloud = WordCloud(max_words=2000, width=1300, height=600, background_color="white",
                                          font_path='C:/Windows/Fonts/STSONG.TTF').generate_from_frequencies(dicts)
                    wordcloud.to_file(fullName)
                    logger.debug("Saved WordCloudImg File Successed")
                    retVal = True
                else:
                    raise TypeError
            except FileNotFoundError:
                logger.error("FileNotFoundError: 文件目录的路径错误 (%s) ！！！" % path)
            except TypeError:
                logger.error("TypeError: 错误的字典类型 (dicts = %s)！！！" % dicts)
        return retVal


def main():
    fs = FileServer()
    stwd = fs.loadLocalGensimDict(path="../../Out/Dicts/", fileName="stopWords.dict")
    if stwd:
        print("Successed")
    else:
        print("Failed")

        # print("当前工作目录:%s" % os.getcwd())
        # print("当前工作目录:%s" % os.path.abspath('.'))
        # print("当前工作目录:%s" % os.path.abspath(os.curdir))
        # print("当前工作目录的父目录:%s" % os.path.abspath('..'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
