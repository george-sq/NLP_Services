#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @File   : httpServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/16 13:49
    @Todo   : 
"""

import logging
import re
import socket
from multiprocessing import Process
from textAnalysisServer import app

logger = logging.getLogger(__name__)

# 创建一个handler，用于写入日志文件
logfile = "/home/pamo/Codes/Logs/log_textAnalysis.log"
fileLogger = logging.FileHandler(filename=logfile, encoding="utf-8")
fileLogger.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
stdoutLogger = logging.StreamHandler()
stdoutLogger.setLevel(logging.DEBUG)  # 输出到console的log等级的开关

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                    datefmt="%Y-%m-%d(%A) %H:%M:%S", handlers=[fileLogger, stdoutLogger])

url_regexp = re.compile(r"(?:[/])(\S*)(?=\s)", re.IGNORECASE)


class HTTPServer(object):
    """"""

    def __init__(self, application=None):
        """构造函数"""
        self.serSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.request_data = None
        self.response_header = None
        self.response_body = None
        self.response_data = None
        self.app = application

    def bind(self, addr):
        """
        :param addr:
        :return:
        """
        if isinstance(addr, tuple):
            self.serSocket.bind(addr)

    def getResponseHeader(self, status=None, headerInfos=("Content-Type", "application/json; charset=UTF-8")):
        """ 构造HTTP响应报文的头部信息
        :param status:
        :param headerInfos:
        :return:
        """
        # HTTP响应状态字典
        STATUS_Dicts = {200: "HTTP/1.1 200 OK\r\n", 404: "HTTP/1.1 404 NO_ACTION\r\n",
                        500: "HTTP/1.1 500 Server_Error\r\n"}
        if status is not None:
            logger.info("生成HTTP响应报文的Header信息...")
            self.response_header = STATUS_Dicts[status]
            self.response_header += "%s: %s\r\n\r\n" % headerInfos
            logger.info("生成HTTP响应报文的Header信息 : %s" % repr(self.response_header))
        else:
            logger.info("缺少构造HTTP响应报文的状态码 : status=%s" % status)

    def parseData(self, reqData):
        """ 解析http request data
        :param reqData:
        """
        request_dict = {}
        requestLines = reqData.splitlines()
        startLine = requestLines[0]
        url = str(url_regexp.search(startLine).group())
        request_dict.setdefault("url", url)
        if len(requestLines[-1]) > 0:
            request_dict.setdefault("body", requestLines[-1])
        for line in requestLines[1:-1]:
            if len(line) > 0:
                k, v = line.split(": ")
                request_dict.setdefault(k, v)

        self.request_data = request_dict

    def clientHandler(self, client_socket, destAddr):
        """ 处理客户端请求
        :param client_socket:
        :param destAddr:
        """
        logger.info("开启 客户端%s 服务子进程" % str(destAddr))
        request = ""
        client_socket.settimeout(0.5)  # 防止请求数据长度为2048造成卡死
        try:
            # 1.获取客户端请求数据
            while True:
                recvData = client_socket.recv(2048)
                logger.info("接收 客户端%s 请求数据..." % str(destAddr))
                if 2048 == len(recvData):
                    request += recvData.decode("utf-8")
                elif 0 < (2048 - len(recvData)):
                    request += recvData.decode("utf-8")
                    break

        except socket.timeout:
            pass
        finally:
            # 2.解析请求数据
            logger.info("完成 客户端%s 数据接收" % str(destAddr))
            if len(request) > 0:
                logger.info("解析 客户端%s 请求数据..." % str(destAddr))
                self.parseData(request)
                logger.info("完成 客户端%s 请求数据解析 : %s" % (str(destAddr), repr(self.request_data)))
            if self.request_data is not None:
                logger.info("生成 服务器 响应数据...")
                self.response_body = self.app(self.request_data, self.getResponseHeader)
                # 3.生成响应数据
                if self.response_header is not None:
                    self.response_data = self.response_header
                    self.response_data += str(self.response_body)
                else:
                    self.getResponseHeader(500)
                    self.response_data = self.response_header
                # 4.向客户端返回响应数据
                logger.info("返回 服务器 响应数据 : %s" %
                            repr([line for line in self.response_data.splitlines() if 0 < len(line)]))

                client_socket.send(self.response_data.encode("utf-8"))
            else:
                logger.warning("客户端%s 请求报文为空" % str(destAddr))

            # 5.关闭客户端连接
            logger.info("关闭 客户端%s 连接" % str(destAddr))
            client_socket.close()

    def start(self):
        """ 启动httpServer """
        self.serSocket.listen(128)
        logger.info("[ HTTP服务器 ] 服务器启动成功")
        while True:
            try:
                logger.info("[ HTTP服务器 ] 等待新客户端请求...")
                newSocket, destAddr = self.serSocket.accept()
                logger.info("[ HTTP服务器 ] 收到客户端%s请求，创建客户端服务子进程" % str(destAddr))
                client_process = Process(target=self.clientHandler, args=(newSocket, destAddr))
                logger.info("[ HTTP服务器 ] 启动客户端处理子进程%s" % str(destAddr))
                client_process.start()
                newSocket.close()
            except Exception as er:
                logger.error("[ HTTP服务器 ] HTTP服务器出错 : %s" % er)
                self.serSocket.close()
                logger.warning("[ HTTP服务器 ] 重启 HTTP服务器...")
                self.start()


def main():
    logger.info("[ HTTP服务器 ] 服务器初始化")
    http_server = HTTPServer(app)
    localAddr = ("", 8899)
    http_server.bind(localAddr)
    http_server.start()


if __name__ == "__main__":
    main()
