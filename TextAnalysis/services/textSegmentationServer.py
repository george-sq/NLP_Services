# -*- coding: utf-8 -*-
"""
    @File   : textSegmentationServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/2/8 9:58
    @Todo   : 
"""

import json
import logging

from nlp.basicTextProcessing import BasicTextProcessing

logger = logging.getLogger(__name__)


class TextSegmentation(object):
    def __call__(self, *args, **kwargs):
        """
        :param args: args[0]={"tag": False, "txt": "textContent"}
        :param kwargs:
        :return: segmentation sequence
        """
        rsp_json = None
        request_data = json.loads(args[0])
        if isinstance(request_data, dict):
            # 获取待处理数据
            txt = request_data.get("txt", None)
            if txt:
                tag = request_data.get("tag", False)
                # 生成文本预处理器
                textHandler = BasicTextProcessing()
                if tag:  # 需要词性标注
                    logger.info("Splitting word and tagging word for requesting text")
                    wordSeqs = textHandler.doWordSplit(content=txt, pos=True)[0]
                    for i in range(len(wordSeqs)):
                        wordSeqs[i] = " | ".join(wordSeqs[i])  # " | " 连接词和词性标注作为一个切分单元项
                else:  # 仅需要分词
                    logger.info("Splitting word for requesting text")
                    wordSeqs = textHandler.doWordSplit(content=txt, )[0]
                rsp_json = "\t|\t".join(wordSeqs)  # "\t|\t" 连接每个切分单元
                # print(rsp_json)
                if not rsp_json:
                    rsp_json = False

        if rsp_json:
            logger.info("Text segmentation successed")
        else:
            logger.error("Text segmentation failed")
        return rsp_json


app = TextSegmentation()


def main():
    request_json = {"tag": True, "txt": "1月4日，东四路居民张某，在微信聊天上认识一位自称为香港做慈善行业的好友，对方称自己正在做慈善抽奖活动"}
    rsp = app(json.dumps(request_json))
    print(rsp)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
