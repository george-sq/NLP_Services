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

import textAnalysisServer as tas

logger = logging.getLogger(__name__)

url_regexp = re.compile(r"(?:[/])(\S*)(?=\s)", re.IGNORECASE)

# 创建一个handler，用于写入日志文件
logfile = "/home/pamo/Codes/Logs/log_textAnalysis.log"
fileLogger = logging.FileHandler(filename=logfile, encoding="utf-8")
fileLogger.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
stdoutLogger = logging.StreamHandler()
stdoutLogger.setLevel(logging.DEBUG)  # 输出到console的log等级的开关

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d(%A) %H:%M:%S", handlers=[fileLogger, stdoutLogger])


class HTTPServer(object):
    """"""

    def __init__(self):
        """构造函数"""
        self.serSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.request_data = None
        self.response_header = None
        self.response_body = None
        self.response_data = None

    def bind(self, addr):
        """
        :param addr:
        :return:
        """
        if isinstance(addr, tuple):
            self.serSocket.bind(addr)

    def getResponseHeader(self, status, headerInfos=("Content-Type", "application/json; charset=UTF-8")):
        """
        :param status:
        :param headerInfos:
        :return:
        """
        # HTTP响应状态字典
        STATUS_Dicts = {200: "HTTP/1.1 200 OK\r\n", 404: "HTTP/1.1 404 NO_ACTION\r\n"}
        self.response_header = STATUS_Dicts[status]
        self.response_header += "%s: %s\r\n\r\n" % headerInfos

    def parseData(self, reqData):
        """ 解析http request data
        :param reqData:
        """
        request_dict = {}
        requestLines = reqData.splitlines()
        startLine = requestLines[0]
        url = str(url_regexp.search(startLine).group())
        request_dict.setdefault("url", url)
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
        logger.info("[ 服务子进程 ] 开启 客户端%s 服务子进程" % str(destAddr))
        request_data = ""
        client_socket.settimeout(0.5)  # 防止请求数据长度为2048造成卡死
        try:
            # 1.获取客户端请求数据
            while True:
                recvData = client_socket.recv(2048)
                logger.info("[ 服务子进程 ] 接收 客户端%s 请求数据......" % str(destAddr))
                if 2048 == len(recvData):
                    request_data += recvData.decode("utf-8")
                elif 0 < (2048 - len(recvData)):
                    request_data += recvData.decode("utf-8")
                    break

        except socket.timeout:
            pass
        finally:
            # 2.解析请求数据
            logger.info("[ 服务子进程 ] 完成 客户端%s 数据接收" % str(destAddr))
            logger.info("[ 服务子进程 ] 解析 客户端%s 请求数据......" % str(destAddr))
            self.parseData(request_data)
            logger.info(eval(repr("[ 服务子进程 ] 完成 客户端%s 请求数据解析 : %s" % (str(destAddr), str(self.request_data)))))
            logger.info("[ 服务子进程 ] 生成 服务器 响应数据......")
            if self.request_data is not None:
                self.response_body = tas.app(self.request_data, self.getResponseHeader)
            # 3.生成响应数据
            self.response_data = self.response_header
            self.response_data += self.response_body
            logger.info(r"[ 服务子进程 ] 返回 服务器 响应数据 : %s" % self.response_data)
            # 4.向客户端返回响应数据
            client_socket.send(self.response_data.encode("utf-8"))

            # 5.关闭客户端连接
            logger.info("[ 服务子进程 ] 关闭 客户端%s 连接" % str(destAddr))
            client_socket.close()

    def start(self):
        """ 启动httpServer """
        self.serSocket.listen(128)
        logger.info("[ HTTP服务器 ] 服务器启动成功")
        while True:
            try:
                logger.info("[ HTTP服务器 ] 等待客户端请求......")
                newSocket, destAddr = self.serSocket.accept()
                logger.info("[ HTTP服务器 ] 收到客户端%s请求，创建客户端服务子进程" % str(destAddr))
                client_process = Process(target=self.clientHandler, args=(newSocket, destAddr))
                logger.info("[ HTTP服务器 ] 启动客户端处理子进程%s" % str(destAddr))
                client_process.start()
                newSocket.close()
            except Exception as er:
                logger.error("[ HTTP服务器 ] HTTP服务器出错 : %s" % er)
                self.serSocket.close()
                logger.warning("[ HTTP服务器 ] 重启 HTTP服务器......")
                self.start()


def main():
    logger.info("[ HTTP服务器 ] 服务器初始化")
    http_server = HTTPServer()
    localAddr = ("", 8899)
    http_server.bind(localAddr)
    http_server.start()


if __name__ == "__main__":
    main()
