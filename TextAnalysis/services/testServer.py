# -*- coding: utf-8 -*-
"""
    @File   : testServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/18 16:44
    @Todo   : 
"""

import logging

logger = logging.getLogger(__name__)


class Test(object):
    def __call__(self, *args, **kwargs):
        logger.info("App方法获得的参数 : args=%s" % args)
        logger.info("App方法获得的参数 : kwargs=%s" % kwargs)
        return "test action"


app = Test()
