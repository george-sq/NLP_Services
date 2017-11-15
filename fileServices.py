# -*- coding: utf-8 -*-
"""
    @File   : fileServices.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/15 11:18
    @Todo   : 提供关于操作本地文件的服务
"""

import os
import pickle


class FileServer(object):
    """关于文件操作的服务类"""

    def __init__(self):
        self.filecode = "utf-8"
        self.pick = pickle

    def loadPickledObjFile(self, path, fileName):
        """Read a pickled object representation from the open file
            :param path: (str)文件所在的目录
            :param fileName: 文件名
            :return:
        """
        if not os.path.exists(path):
            os.mkdir(path)
        filePath = path
        if isinstance(path, str):
            if "/" == path[-1]:
                filePath += fileName
            else:
                filePath += "/" + fileName
        else:
            raise FileNotFoundError
        try:
            with open(filePath, 'rb') as ldf:
                return self.pick.load(ldf)
        except FileNotFoundError:
            print('FileNotFoundError: 错误的文件路径(%s)！！！' % filePath)
            exit(1)

    def loadLocalText(self):

        pass
