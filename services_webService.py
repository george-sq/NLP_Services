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

# 设置静态文件根目录
HTML_ROOT_DIR = "./html"


def show_ctime(start_response):
    status = "200 OK"
    headers = [
        ("Content-Type", "text/html; charset=UTF-8")
    ]
    start_response(status, headers)
    return time.ctime()


class HTTPServer(object):
    """"""

    def __init__(self):
        """构造函数， application指的是框架的app"""
        self.serSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.response_headers = ""

    def bind(self, port):
        self.serSocket.bind(("", port))

    def start(self):
        self.serSocket.listen(128)
        print('>> %s [ HTTP服务器: 启动成功!!! ]' % datetime.datetime.now())
        while True:
            try:
                print('>> %s [ HTTP服务器: 主进程等待客户端请求...... ]' % datetime.datetime.now())
                newSocket, destAddr = self.serSocket.accept()
                print('>> %s [ HTTP服务器: 收到客户端%s请求，创建客户端服务子进程. ]' % (datetime.datetime.now(), str(destAddr)))
                client = Process(target=self.clientHandler, args=(newSocket, destAddr))
                client.start()
                print('>> %s [ HTTP服务器: 关闭客户端%s ]' % (datetime.datetime.now(), str(destAddr)))
                newSocket.close()
            except Exception as er:
                print('>> %s [ HTTP服务器: 服务器出错 ：%s ]' % (datetime.datetime.now(), er))
                self.serSocket.close()
                print('>> %s [ HTTP服务器: 重启TCP服务器...... ]' % datetime.datetime.now())
                main()
            finally:
                # 当为所有的客户端服务完之后再进行关闭，表示不再接收新的客户端的链接
                self.serSocket.close()

    def start_response(self, status, headers):
        """
            status = "200 OK"
            headers = [
                ("Content-Type", "text/plain")
            ]
        """
        response_headers = "HTTP/1.1 " + status + "\r\n"
        for header in headers:
            response_headers += "%s: %s\r\n" % header
        self.response_headers = response_headers

    def clientHandler(self, client_socket):
        """处理客户端请求"""
        # 获取客户端请求数据
        request_data = client_socket.recv(1024)
        print("request data:")
        request_lines = request_data.splitlines()
        for line in request_lines:
            print(line)

        # 解析请求报文
        # 'GET / HTTP/1.1'
        request_start_line = request_lines[0]
        print("*" * 10)
        print("request_start_line :", request_start_line.decode("utf-8"))
        print("*" * 10)

        response_body = show_ctime(self.start_response)
        response = self.response_headers + "\r\n" + response_body

        # 向客户端返回响应数据
        client_socket.send(bytes(response, "utf-8"))

        # 关闭客户端连接
        client_socket.close()


def main():
    http_server = HTTPServer()
    http_server.bind(7788)
    http_server.start()


if __name__ == "__main__":
    main()
