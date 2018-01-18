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


def buildResponseHeader(status, headerInfos=("Content-Type", "application/json; charset=UTF-8")):
    """ 构造HTTP响应报文的头部信息
    :param status:
    :param headerInfos:
    :return:
    """
    # HTTP响应状态字典
    STATUS_Dicts = {200: "HTTP/1.1 200 OK\r\n", 404: "HTTP/1.1 404 NO_ACTION\r\n"}
    response_header = STATUS_Dicts[status]
    response_header += "%s: %s\r\n\r\n" % headerInfos


class Application(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        request_data = args[0]
        getResponseHeader = args[1]
        # print(request_data)
        # print(getResponseHeader)
        getResponseHeader(200)
        return show_ctime()


action_dict = {"/": show_ctime}
app = Application()


def main():
    request_data = {'url': '/', 'body': '', 'Host': '10.0.0.230:8899', 'Connection': 'keep-alive',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate', 'Accept-Language': 'zh-CN,zh;q=0.9'}
    rsp_body = app(request_data, buildResponseHeader)


if __name__ == '__main__':
    main()
