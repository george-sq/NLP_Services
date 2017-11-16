# -*- coding: utf-8 -*-
"""
    @File   : fileServices.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/15 11:18
    @Todo   : 提供关于操作本地文件的服务
"""

import os
import pickle
from gensim import corpora


class FileServer(object):
    """关于文件操作的服务类"""

    def __init__(self):
        self.filecode = "utf-8"
        self.pick = pickle

    @staticmethod
    def __checkPathArgType(path):
        """ 路径参数类型校验
            :param path: (str)文件目录路径
            :return: boolean
        """
        try:
            if not isinstance(path, str):
                raise TypeError
        except TypeError:
            print('TypeError: 错误的文件路径参数类型(%s)！！！' % path)
            return False
        else:
            return True

    def __checkPath(self, path):
        """ 校验文件路径有效性
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
        # 路径参数类型校验
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            try:
                # 加载文件
                with open(fullName, 'r', encoding=self.filecode) as txtr:
                    return txtr.read()
            except FileNotFoundError:
                print('FileNotFoundError: 错误的文件路径(%s)！！！' % path)
                return None
        else:
            return None

    def loadPickledObjFile(self, path, fileName):
        """ Read a pickled object representation from the open file
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :return: fileContent or None
        """
        # 路径参数类型校验
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            try:
                # 加载文件
                with open(fullName, 'rb') as ldf:
                    return self.pick.load(ldf)
            except FileNotFoundError:
                print('FileNotFoundError: 错误的文件路径(%s)！！！' % path)
                return None
        else:
            return None

    def loadLocalCorpus(self, path, fileName):
        """ 加载gensim   模块生成的本地语料库文件
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :return: fileContent or None
        """
        # 路径参数类型校验
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            # 文件目录路径校验
            if os.path.exists(path) and os.path.isfile(fullName):
                # 加载文件
                return corpora.MmCorpus(fullName)
            else:
                print('FileNotFoundError: 错误的文件路径(%s)！！！' % path)
                return None
        else:
            return None

    def loadLocalDict(self, path, fileName):
        """ 加载gensim模块生成的本地字典文件
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :return: fileContent or None
        """
        # 路径参数类型校验
        if self.__checkPathArgType(path):
            fullName = os.path.join(path, fileName)
            # 文件目录路径校验
            if os.path.exists(path) and os.path.isfile(fullName):
                # 加载文件
                return corpora.Dictionary.load(fullName)
            else:
                print('FileNotFoundError: 错误的文件路径(%s)！！！' % path)
                return None
        else:
            return None


            ########################################################################################################################

    def saveText2UTF8(self, path, fileName, **kwargs):
        """
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :param kwargs: 写入内容
            :return: boolean --> content or lines
        """
        retVal = False
        # 路径参数类型校验
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
                with open(fullName, "wb", encoding=self.filecode) as txtw:
                    # 写入内容校验
                    content = kwargs.get("content", None)
                    lines = kwargs.get("lines", [])
                    if content is not None:
                        txtw.write(content)
                    if len(lines) > 0:
                        txtw.writelines(lines)
                        retVal = True
            except FileNotFoundError:
                print('FileNotFoundError: 错误的文件路径(%s)！！！' % path)
        return retVal

    def savePickledObjFile(self, path, fileName, writeContentObj=None):
        """
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :param writeContentObj: 写入内容对象 eg. Bunch()
            :return: boolean
        """
        retVal = False
        # 路径参数类型校验
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

    def saveGensimCourpus2MM(self, path, fileName, inCorpus=None):
        """
            :param path: (str)文件所在的目录路径
            :param fileName: 文件名
            :param inCorpus: 写入语料库内容对象 --> [[(,),],]
            :return: boolean
        """
        retVal = False
        # 路径参数类型校验
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
                if inCorpus is not None:
                    # 内容写入
                    corpora.MmCorpus.serialize(fullName, corpus=inCorpus)
                    retVal = True
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
        # 路径参数类型校验
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
