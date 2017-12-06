# -*- coding: utf-8 -*-
"""
    @File   : services_similarity4lsi.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/12/6 11:26
    @Todo   : 
"""

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
from gensim import models
from gensim import similarities
import services_textProcess as tps
import jieba

jieba.setLogLevel(log_level=logging.INFO)


def main():
    pass


if __name__ == '__main__':
    main()
