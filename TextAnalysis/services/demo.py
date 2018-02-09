# -*- coding: utf-8 -*-
"""
    @File   : demo.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/2/9 9:37
    @Todo   : 
"""

import json
import logging

from bases.mysqlServer import MysqlServer
from nlp.basicTextProcessing import BasicTextProcessing
from services import textCateServer as tc

logger = logging.getLogger(__name__)


class Demo(object):
    def __call__(self, *args, **kwargs):
        rsp_json = None  # 参数错误返回None, 过程出错返回False, 成功返回tid
        # 解析请求数据,获取新文本
        body = json.loads(args[0])
        tid = body.get("tid", None)
        txt = body.get("txt", None)
        if txt:
            # 分词&词性标注
            txtHandler = BasicTextProcessing()
            wordSeqs = txtHandler.doWordSplit(content=txt)
            posSeqs = txtHandler.doWordSplit(content=txt, pos=True)
            if not wordSeqs and not posSeqs:
                logger.error("Text segmentation failed")
            # 词频统计
            wordFreqs = txtHandler.buildWordFrequencyDict(wordSeqs)
            wordFreqs = sorted(wordFreqs.items(), key=lambda x: x[1], reverse=True)
            if not wordFreqs:
                logger.error("Word frequency statistical failed")
            # 分类
            label = tc.app(json.dumps({"txt": txt}))
            if not label:
                logger.error("Classifying text failed")
            if not wordSeqs or not posSeqs or not wordFreqs or not label:
                rsp_json = False
            # 入库？
            dnHandler = MysqlServer(host="10.0.0.247", db="db_pamodata", user="pamo", passwd="pamo")
            #     文本入库
            txt_sql = ""
            #     分词序列入库
            ws_sql = ""
            #     词性标注序列入库
            pos_sql = ""
            #     分类结果入库
            tc_sql = ""
        # 返回响应
        print(rsp_json)
        return rsp_json


app = Demo()


def main():
    txt = "1月4日，东四路居民张某，在微信聊天上认识一位自称为香港做慈善行业的好友，对方称自己正在做慈善抽奖活动，因与张某关系好，" \
          "特给其预留了30万中奖名额，先后以交个人所得税、押金为名要求张某以无卡存款的形式向其指定的账户上汇款60100元"
    body = json.dumps({"tid": 0, "txt": txt})
    rsp = app(body)
    print(rsp)
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
