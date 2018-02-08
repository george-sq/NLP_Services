# -*- coding: utf-8 -*-
"""
    @File   : textSegmentationServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/2/8 9:58
    @Todo   : 
"""

import logging
from nlp.basicTextProcessing import BasicTextProcessing

logger = logging.getLogger(__name__)


class TextSegmentation(object):
    def __call__(self, *args, **kwargs):
        # args[0]={"tag": False, "txt": "textContent"}
        rsp_json = None
        if isinstance(args[0], dict):
            # 获取待处理数据
            txt = args[0].get("txt", None)
            if txt:
                tag = args[0].get("tag", False)
                # 生成文本预处理器
                textHandler = BasicTextProcessing()
                if tag:  # 需要词性标注
                    logger.info("Splitting word and tagging word for requesting text")
                    wordSeqs = textHandler.doWordSplit(content=txt, pos=True)[0]
                else:  # 仅需要分词
                    logger.info("Splitting word for requesting text")
                    wordSeqs = textHandler.doWordSplit(content=txt, )[0]
                if wordSeqs:
                    rsp_json = wordSeqs
                else:
                    rsp_json = False
        return rsp_json


app = TextSegmentation()


def main():
    request_json = {"tag": True, "txt": "1月4日，东四路居民张某，在微信聊天上认识一位自称为香港做慈善行业的好友，对方称自己正在做慈善抽奖活动"}
    rsp = app(request_json)
    print(rsp)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
