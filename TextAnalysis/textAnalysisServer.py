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
from services import demoServer as demo

logger = logging.getLogger(__name__)


class ShowTime(object):
    """默认响应测试"""

    @staticmethod
    def __name__():
        return "ShowTime"

    @staticmethod
    def app(*args):
        """测试0"""
        logger.warning("App方法获得的参数 : args=%s" % args)
        result = {"RESULT_CODES": 1, "RESULT": str(time.ctime())}
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
                    logger.info("完成HTTP响应报文的Body信息生成 : %s" % result)
                    # 根据响应结果生成HTTP响应报文的头部信息
                    if result:  # 成功响应
                        getResponseHeader(200)
                        result = {"RESULT_CODES": 1, "RESULT": result}
                        result = json.dumps(result)
                    elif result is False:  # 响应出错, 响应功能内部错误
                        getResponseHeader(500)
                        result = json.dumps({"RESULT_CODES": 0})
                    else:  # 无法响应请求, 请求数据异常, 如返回值为None
                        getResponseHeader(400)
                        result = json.dumps({"RESULT_CODES": 0})
                    return result
            return self.action_modules.get("/").app()

        else:
            logger.error("调用应用服务框架的参数错误")
            return None


"""  模块功能映射字典说明 RESULT_CODES = {"Failed": 0, "Successed": 1}

    0. HTTP服务框架测试功能
        "/": ShowTime()
            请求参数格式: {"url": '/', "body": ""}
            响应结果内容: {None: 请求数据异常, False: 响应异常, True:成功响应}
            响应数据格式: {"RESULT_CODES": 0 or 1, "RESULT": result}
        "/db": db
            请求参数格式: {"url": '/db', "body": "test"}
            响应结果内容: {None: 请求数据异常, False: 响应异常, True:成功响应}
            响应数据格式: {"RESULT_CODES": 0 or 1, "RESULT": result}
        "/test": act
            请求参数格式: {"url": '/test', "body": "test"}
            响应结果内容: {None: 请求数据异常, False: 响应异常, True:成功响应}
            响应数据格式: {"RESULT_CODES": 0 or 1, "RESULT": result}
            
    1. 文本分析模块功能
        "/txtSeg": tseg
            请求功能描述: 文本切分
            请求参数格式: {"url": '/txtSeg', "body": {"tag": True or False, "txt": "textContent"}}
            响应结果内容: {None: 请求数据异常, False: 响应异常, True:成功响应}
            响应数据格式: {"RESULT_CODES": 0 or 1, "RESULT": segmentation sequence}
        "/txtCate": tc
            请求功能描述: 文本分类
            请求参数格式: {"url": '/txtCate', "body": {"txt": "textContent"}}
            响应结果内容: {None: 请求数据异常, False: 响应异常, True:成功响应}
            响应数据格式: {"RESULT_CODES": 0 or 1, "RESULT": {"label": oneLabel, "prob": "%.5f" % probability}}
        "/demo": demo
            请求功能描述: 文本分析功能演示平台接口
            请求参数格式: {"url": '/demo', "body": {"tid": tid or None, "txt": "textContent"}}
            响应结果内容: {None: 请求数据异常, False: 响应异常, True:成功响应}
            响应数据格式: {"RESULT_CODES": 0 or 1, "RESULT": None, False or tid}
"""
action_dicts = {
    "/": ShowTime(),
    "/db": db,
    "/test": act,
    "/txtCate": tc,
    "/txtSeg": tseg,
    "/demo": demo
}
app = Application(action_dicts)


def main():
    # request_data = {"url": '/test', 'Host': '10.0.0.230:8899', 'Connection': 'keep-alive',
    #                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    #                 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    #                 'Accept-Encoding': 'gzip, deflate', 'Accept-Language': 'zh-CN,zh;q=0.9',
    #                 "body": "test"}

    # txt = "1月4日，东四路居民张某，在微信聊天上认识一位自称为香港做慈善行业的好友，对方称自己正在做慈善抽奖活动，因与张某关系好，" \
    #       "特给其预留了30万中奖名额，先后以交个人所得税、押金为名要求张某以无卡存款的形式向其指定的账户上汇款60100元"
    # body = json.dumps({"txt": txt})
    # request_data = {"url": '/txtCate', 'Host': '10.0.0.230:8899', 'Connection': 'keep-alive',
    #                 "body": body}

    txt = "1月4日，东四路居民张某，在微信聊天上认识一位自称为香港做慈善行业的好友，对方称自己正在做慈善抽奖活动，因与张某关系好，" \
          "特给其预留了30万中奖名额，先后以交个人所得税、押金为名要求张某以无卡存款的形式向其指定的账户上汇款60100元"
    body = json.dumps({"tag": False, "txt": txt})
    request_data = {"url": '/txtSeg', 'Host': '10.0.0.230:8899', 'Connection': 'keep-alive',
                    "body": body}
    rsp_body = app(request_data, buildResponseHeader)
    print(len(rsp_body))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
