# -*- coding: utf-8 -*-
"""
    @File   : textAnalysisServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/16 13:47
    @Todo   : 
"""

import time
import json

RESULT_CODES = {"f": 0, "s": 1}


def show_ctime():
    """测试0"""
    result = {"Root Response": {"RESULT_CODES": 1, "RESULT": None}}
    rsp = result.get("Root Response")
    rsp["RESULT"] = str(time.ctime())
    result = json.dumps(result, ensure_ascii=False)

    return result


class Application(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        request_data = args[0]
        getResponseHeader = args[1]
        print(request_data)
        print(getResponseHeader)
        getResponseHeader(200)
        return show_ctime()


action_dict = {"/": show_ctime}
app = Application()


def main():
    pass


if __name__ == '__main__':
    main()
