# -*- coding: utf-8 -*-
"""
    @File   : services_fileIO.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/15 11:18
    @Todo   : 提供关于操作本地文件的服务
"""

import os
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import pickle
from wordcloud import WordCloud
from gensim import corpora


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
            print('TypeError: 文件路径参数的类型错误 (%s) ！！！' % path)
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

    def loadLocalTextByUTF8(self, path, fileName):
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
                    return txtr.read()
            except FileNotFoundError:
                print('FileNotFoundError: 文件目录的路径错误 (%s) ！！！' % path)
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
                    return self.pick.load(ldf)
            except FileNotFoundError:
                print('FileNotFoundError: 文件目录的路径错误 (%s) ！！！' % path)
                return None
        else:
            return None

    def loadLocalCorpus(self, path, fileName):
        """ 加载gensim   模块生成的本地语料库文件
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :return: fileContent or None
        """
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            # 文件目录路径校验
            if os.path.exists(path) and os.path.isfile(fullName):
                # 加载文件
                return corpora.MmCorpus(fullName)
            else:
                print('FileNotFoundError: 文件目录的路径错误 (%s) ！！！' % path)
                return None
        else:
            return None

    def loadLocalDict(self, path, fileName):
        """ 加载gensim模块生成的本地字典文件
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :return: fileContent or None
        """
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            # 文件目录路径校验
            if os.path.exists(path) and os.path.isfile(fullName):
                # 加载文件
                return corpora.Dictionary.load(fullName)
            else:
                print('FileNotFoundError: 文件目录的路径错误 (%s) ！！！' % path)
                return None
        else:
            return None

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
            if roc:
                print("存在文件路径 (%s)" % path)
            elif "MK" == roc:
                print("新建文件路径 (%s)" % path)
            fullName = os.path.join(path, fileName)
            try:
                # 内容写入
                with open(fullName, "w", encoding=self.filecode) as txtw:
                    # 写入内容校验
                    content = kwargs.get("content", None)
                    lines = kwargs.get("lines", [])
                    if content is not None:
                        txtw.write(content)
                    elif len(lines) > 0:
                        txtw.writelines(lines)
                        retVal = True
                    else:
                        raise ValueError
            except FileNotFoundError:
                print('FileNotFoundError: 文件目录的路径错误 (%s) ！！！' % path)
            except ValueError as ver:
                print(ver)
                print("**kwargs参数错误, 需要给定参数content 或者 参数lines")
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
            if roc:
                print("存在文件路径 (%s)" % path)
            elif "MK" == roc:
                print("新建文件路径 (%s)" % path)
            fullName = os.path.join(path, fileName)
            try:
                # 写入内容校验
                if writeContentObj is not None:
                    # 内容写入
                    with open(fullName, "wb")as pobj:
                        self.pick.dump(writeContentObj, pobj)
                        retVal = True
            except FileNotFoundError:
                print('FileNotFoundError: 错误的文件路径(%s)！！！' % path)
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
            if roc:
                print("存在文件路径 (%s)" % path)
            elif "MK" == roc:
                print("新建文件路径 (%s)" % path)
            fullName = os.path.join(path, fileName)
            try:
                # 写入内容参数校验
                inCorpus = kwargs.get("inCorpus", None)
                if inCorpus is not None:
                    # 内容写入
                    corpora.MmCorpus.serialize(fullName, corpus=inCorpus)
                    retVal = True
                else:
                    print("缺少写入内容, inCorpus = None")
            except FileNotFoundError:
                print('FileNotFoundError: 错误的文件路径(%s)！！！' % path)
        return retVal

    def saveGensimDict(self, path, fileName, dicts=None):
        """
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :param dicts: 写入字典对象 --> corpora.Dictionary()
            :return: boolean
        """
        retVal = False
        # 检验路径参数的类型
        if self.__checkPathArgType(path):
            # 文件目录路径校验
            roc = self.__checkPath(path)
            if roc:
                print("存在文件路径 (%s)" % path)
            elif "MK" == roc:
                print("新建文件路径 (%s)" % path)
            fullName = os.path.join(path, fileName)
            try:
                # 写入内容参数校验
                if dicts is not None:
                    if isinstance(dicts, corpora.Dictionary):
                        # 内容写入
                        dicts.save(fullName)
                        retVal = True
                    else:
                        raise TypeError
            except FileNotFoundError:
                print('FileNotFoundError: 错误的文件路径(%s)！！！' % path)
            except TypeError:
                print('TypeError: 错误的字典类型 (%s)！！！' % dicts)
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
            if roc:
                print("存在文件路径 (%s)" % path)
            elif "MK" == roc:
                print("新建文件路径 (%s)" % path)
            else:
                pass
            fullName = os.path.join(path, fileName)
            try:
                # 写入内容参数校验
                if dicts is not None:
                    # Generate a word cloud image
                    wordcloud = WordCloud(max_words=2000, width=1300, height=600, background_color="white",
                                          font_path='C:/Windows/Fonts/STSONG.TTF').generate_from_frequencies(dicts)
                    wordcloud.to_file(fullName)
                else:
                    raise TypeError
            except FileNotFoundError:
                print('FileNotFoundError: 错误的文件路径(%s)！！！' % path)
            except TypeError:
                print('TypeError: 错误的字典类型 (%s)！！！' % dicts)
        return retVal
