# -*- coding: utf-8 -*-
"""
    @File   : textAnalysisServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/16 13:47
    @Todo   : 
"""

import json
import logging
import time

from bases import mysqlServer as db
from services import testServer as act
from services import textCateServer as tc
from services import textSegmentationServer as tseg

logger = logging.getLogger(__name__)

RESULT_CODES = {"f": 0, "s": 1}


class ShowTime(object):
    """默认响应测试"""

    @staticmethod
    def __name__():
        return "ShowTime"

    @staticmethod
    def app(*args):
        """测试0"""
        logger.warning("App方法获得的参数 : args=%s" % args)
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
    STATUS_Dicts = {200: "HTTP/1.1 200 OK\r\n", 404: "HTTP/1.1 404 NO_ACTION\r\n",
                    500: "HTTP/1.1 500 Server_Error\r\n"}
    logger.info("生成HTTP响应报文的Header信息...")
    response_header = STATUS_Dicts[status]
    response_header += "%s: %s\r\n\r\n" % headerInfos
    logger.info("完成HTTP响应报文的Header信息生成")


class Application(object):
    def __init__(self, action_modules):
        self.action_modules = action_modules

    def __call__(self, *args, **kwargs):
        if len(args) == 2:  # 参数验证
            # 获取参数
            request_data = args[0]
            getResponseHeader = args[1]
            # 解析请求url
            # url = request_data.get("url", "/")
            url = request_data.get("url", None)
            if url is None:  # 默认的容错响应
                logger.error("缺少url参数, 进行默认响应")
                getResponseHeader(404)
            else:
                logger.info("解析URL信息...")
                # 根据url选择响应功能模块
                mdl = self.action_modules.get(url, None)
                if not mdl:
                    logger.error("不存在请求功能模块(%s),进行默认响应" % url)
                    getResponseHeader(404)
                else:
                    # 生成功能响应
                    logger.info("获取请求功能模块 : %s" % mdl)
                    logger.info("生成HTTP响应报文的Body信息...")
                    result = mdl.app(request_data.get("body", None))
                    logger.info("完成HTTP响应报文的Body信息 : %s" % result)
                    # 根据响应结果生成HTTP响应报文的头部信息
                    if result:  # 成功响应
                        getResponseHeader(200)
                    elif result is False:  # 响应出错, 响应功能内部错误
                        getResponseHeader(500)
                    else:  # 无法响应请求, 请求数据异常
                        getResponseHeader(400)
                    return result
            return self.action_modules.get("/").app()

        else:
            logger.error("调用应用服务框架的参数错误")
            return None


action_dicts = {
    "/": ShowTime(),  # {'url': '/', "body": ""}
    "/db": db,  # {'url': '/db', "body": "test"}
    "/test": act,  # {'url': '/test', "body": "test"}
    "/txtCate": tc,  # {'url': '/txtCate', "body": "textContent"}
    "/txtSeg": tseg  # {'url': 'txtSeg', "body": {"tag": False, "txt": "textContent"}}
}
app = Application(action_dicts)


def main():
    # request_data = {'url': '/test', 'Host': '10.0.0.230:8899', 'Connection': 'keep-alive',
    #                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    #                 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    #                 'Accept-Encoding': 'gzip, deflate', 'Accept-Language': 'zh-CN,zh;q=0.9',
    #                 "body": "test"}
    txt = "1月4日，东四路居民张某，在微信聊天上认识一位自称为香港做慈善行业的好友，对方称自己正在做慈善抽奖活动，因与张某关系好，" \
          "特给其预留了30万中奖名额，先后以交个人所得税、押金为名要求张某以无卡存款的形式向其指定的账户上汇款60100元"
    request_data = {'url': '/txtCate', 'Host': '10.0.0.230:8899', 'Connection': 'keep-alive',
                    "body": "%s" % txt}
    rsp_body = app(request_data, buildResponseHeader)
    print(len(rsp_body))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
