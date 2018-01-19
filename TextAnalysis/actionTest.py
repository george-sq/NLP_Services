# -*- coding: utf-8 -*-
"""
    @File   : actionTest.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/18 16:44
    @Todo   : 
"""

import logging

logger = logging.getLogger(__name__)


class Test(object):
    @staticmethod
    def app(*args):
        logger.warning("App方法获得的参数 : args=%s" % args)
        return "test action"
        pass
