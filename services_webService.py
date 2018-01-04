# -*- coding: utf-8 -*-
"""
    @File   : services_webService.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/12/29 9:52
    @Todo   : 
"""

from multiprocessing import Process
import time
import socket
import datetime
import json


def show_ctime():
    """测试"""
    return json.dumps({"Default Response": {"current time": str(time.ctime())}})


ACTION_DICTS = {"/": show_ctime, 1: "", 2: ""}
STATUS_Dicts = {2: "HTTP/1.1 200 OK\r\n", 4: "HTTP/1.1 404 NO ACTION\r\n"}


class HTTPServer(object):
    """"""

    def __init__(self):
        """构造函数"""
        self.serSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.response_header = None
        self.response_body = None

    def bind(self, addr):
        if isinstance(addr, tuple):
            self.serSocket.bind(addr)

    def getResponseHeader(self, status):
        self.response_header = STATUS_Dicts[status] + "%s: %s\r\n" % ("Content-Type", "text/html; charset=UTF-8")

    def getResponseBody(self, action, request_data):
        print("action :", action)
        print("request_data :", request_data)
        self.response_body = action()

    def getResposeInfos(self, request_data, destAddr):
        # 2.1 解析客户端请求数据
        request_start_line = []
        request_params = []
        request_body = ""
        request_resource = ""
        print(">> %s 客户端服务子进程: 完成客户端%s 请求数据接收." % (datetime.datetime.now(), str(destAddr)))
        if len(request_data) > 0:
            print(">> %s 客户端服务子进程: 开始客户端%s 数据处理服务......" % (datetime.datetime.now(), str(destAddr)))
            request_lines = request_data.splitlines()
            request_start_line.extend(request_lines[0].split())
            request_body = request_lines[-1]
            m, s, p = request_start_line  # m=Http请求方法, s=请求资源标识, p=Http协议
            print("<<<<<<<<<<" * 20)
            request_resource = s.split("?")[0]
            if 2 == len(s.split("?")):
                request_params.extend(s.split("?")[-1].split("&"))
            print("request_lines len :", len(request_lines))
            print("request data :")
            for i, line in enumerate(request_lines):
                print(i, line)
            print("<<<<<<<<<<" * 20)
        else:
            print(">> %s 客户端服务子进程: 客户端%s 请求数据长度异常. len=%d" %
                  (datetime.datetime.now(), str(destAddr), len(request_data)))

        # 2.2 生成响应数据
        response = "缺省响应信息"
        action = ACTION_DICTS.get(request_resource, show_ctime)
        if action is not None:
            self.getResponseBody(action, (request_params, request_body))
        if self.response_body is not None:
            self.getResponseHeader(2)
        if self.response_header and self.response_body:
            response = self.response_header + "\r\n" + self.response_body
            print(response)

        print()
        print(">>>>>>>>>>" * 20)
        print("响应数据 :")
        print(response)
        print(">>>>>>>>>>" * 20)
        return response.encode("utf-8")

    def clientHandler(self, client_socket, destAddr):
        """处理客户端请求"""

        # 1.获取客户端请求数据
        print(">> %s 客户端服务子进程: 开启客户端%s 服务子进程!!!" % (datetime.datetime.now(), str(destAddr)))
        request_data = ""
        client_socket.settimeout(0.5)
        try:
            while True:
                recvData = client_socket.recv(2048)
                print(">> %s 客户端服务子进程: 接收客户端%s 请求数据......" % (datetime.datetime.now(), str(destAddr)))
                if 2048 == len(recvData):
                    request_data += recvData.decode("utf-8")
                elif 0 < (2048 - len(recvData)):
                    request_data += recvData.decode("utf-8")
                    break
        except socket.timeout as e:
            print(e)
        finally:
            # 2.生成响应数据
            response = self.getResposeInfos(request_data, destAddr)
            # 3.向客户端返回响应数据
            client_socket.send(response)

            # 4.关闭客户端连接
            client_socket.close()

    def start(self):
        self.serSocket.listen(128)
        print(">> %s [ HTTP服务器: 启动成功!!! ]" % datetime.datetime.now())
        while True:
            try:
                print(">> %s [ HTTP服务器: 主进程等待客户端请求...... ]" % datetime.datetime.now())
                newSocket, destAddr = self.serSocket.accept()
                print(">> %s [ HTTP服务器: 收到客户端%s请求，创建客户端服务子进程. ]" % (datetime.datetime.now(), str(destAddr)))
                client_process = Process(target=self.clientHandler, args=(newSocket, destAddr))
                print(">> %s [ HTTP服务器: 主进程启动客户端处理子进程%s ]" % (datetime.datetime.now(), str(destAddr)))
                client_process.start()
                print(">> %s [ HTTP服务器: 主进程关闭客户端套接字%s ]" % (datetime.datetime.now(), str(destAddr)))
                newSocket.close()
            except Exception as er:
                print(">> %s [ HTTP服务器: 服务器出错 ：%s ]" % (datetime.datetime.now(), er))
                self.serSocket.close()
                print(">> %s [ HTTP服务器: 重启TCP服务器...... ]" % datetime.datetime.now())
                self.start()


def main():
    http_server = HTTPServer()
    localAddr = ("", 8899)
    http_server.bind(localAddr)
    http_server.start()


if __name__ == "__main__":
    main()
