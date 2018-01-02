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


def show_ctime():
    """测试"""
    return time.ctime()


ACTION_DICTS = {"/": show_ctime, 1: "", 2: ""}
STATUS_Dicts = {2: "HTTP/1.1 200 OK\r\n", 4: "HTTP/1.1 404 NO ACTION\r\n"}


class HTTPServer(object):
    """"""

    def __init__(self):
        """构造函数"""
        self.serSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.response_headers = ""
        self.response_body = ""

    def bind(self, addr):
        if isinstance(addr, tuple):
            self.serSocket.bind(addr)

    def start(self):
        self.serSocket.listen(128)
        print('>> %s [ HTTP服务器: 启动成功!!! ]' % datetime.datetime.now())
        while True:
            try:
                print('>> %s [ HTTP服务器: 主进程等待客户端请求...... ]' % datetime.datetime.now())
                newSocket, destAddr = self.serSocket.accept()
                print('>> %s [ HTTP服务器: 收到客户端%s请求，创建客户端服务子进程. ]' % (datetime.datetime.now(), str(destAddr)))
                client = Process(target=self.clientHandler, args=(newSocket,))
                client.start()
                print('>> %s [ HTTP服务器: 关闭客户端%s ]' % (datetime.datetime.now(), str(destAddr)))
                newSocket.close()
            except Exception as er:
                print('>> %s [ HTTP服务器: 服务器出错 ：%s ]' % (datetime.datetime.now(), er))
                self.serSocket.close()
                print('>> %s [ HTTP服务器: 重启TCP服务器...... ]' % datetime.datetime.now())
                self.start()
            finally:
                # 当为所有的客户端服务完之后再进行关闭，表示不再接收新的客户端的链接
                self.serSocket.close()

    def getResponseHeaders(self, status):
        response_headers = STATUS_Dicts[status]
        response_headers += "%s: %s\r\n" % ("Content-Type", "text/html; charset=UTF-8")
        self.response_headers = response_headers

    def clientHandler(self, client_socket):
        """处理客户端请求"""
        # 获取客户端请求数据
        request_data = ""
        while True:
            recvData = client_socket.recv(1024)
            if len(recvData) > 0:
                request_data += recvData.decode("utf-8")
            else:
                break

        print("request data :")
        request_lines = request_data.splitlines()
        for line in request_lines:
            print(line)

        # 解析请求报文
        # 'GET / HTTP/1.1'
        request_start_line = request_lines[0]
        print("*" * 10)
        print("request_start_line :", request_start_line)
        print("*" * 10)

        # 生成响应数据
        actKey = request_start_line.split()[1]
        self.response_body = ACTION_DICTS[actKey]()
        if self.response_body is not None:
            self.getResponseHeaders(2)
        response = self.response_headers + "\r\n" + self.response_body

        # 向客户端返回响应数据
        client_socket.send(response.encode("utf-8"))

        # 关闭客户端连接
        client_socket.close()


def main():
    http_server = HTTPServer()
    localAddr = ('', 8899)
    http_server.bind(localAddr)
    http_server.start()


if __name__ == "__main__":
    main()
