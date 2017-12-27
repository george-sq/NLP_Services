# -*- coding: utf-8 -*-
"""
    @File   : services_tcp.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/8 9:52
    @Todo   : 
"""

import datetime
from socket import *
from multiprocessing import *
import services_online
import services_similarity4tfidf as tfidfss
import services_ner as ner


def usage():
    """
        The output  configuration file contents.

        Usage: xxx.py -c|--content arg0 -s|--stpwd arg1 -m|--model arg2

        Description
                    -c,--content  Configure Text content information.
                    -s,--stpwd    Configure StopWords Path information.
                    -m,--model    Configure NLP_Model Path information.
        For Example:
            python xxx.py -c|--content textContent -s|--stpwd stpwd_path -m|--model model_path
    """


servicesDicts = {"base": services_online.main, "sim": tfidfss.tfidfSimilartyProcess, "ner": ner}


# 处理客户端的请求并为其服务
def dealClient(newSocket, destAddr):
    print('>> %s 服务子进程: 客户端%s服务子进程开启!!!' % (datetime.datetime.now(), str(destAddr)))
    recvContent = ""
    while True:
        recvData = newSocket.recv(1024)
        if len(recvData) > 0:
            print('>> %s 服务子进程: 开始接收客户端%s请求数据!!!' % (datetime.datetime.now(), str(destAddr)))
            recvContent += recvData.decode("utf-8")
        else:
            print('>> %s 服务子进程: 客户端%s请求数据接收完成，开始数据处理服务......' % (datetime.datetime.now(), str(destAddr)))
            print('>> %s 服务子进程: 客户端%s请求数据内容:\n\t' % (datetime.datetime.now(), str(destAddr)), recvContent)
            responseMsg = ""
            if len(recvContent) > 0:
                responseMsg = services_online.main(recvContent)
            print('>> %s 服务子进程: 服务端响应数据:%s\n\t' % (datetime.datetime.now(), responseMsg))
            newSocket.send(responseMsg.encode("utf-8"))
            print('>> %s 服务子进程: 客户端%s请求数据处理完成!!!' % (datetime.datetime.now(), str(destAddr)))
            break

    print('>> %s 服务子进程: 关闭客户端%s' % (datetime.datetime.now(), str(destAddr)))
    newSocket.close()


def main():
    serSocket = socket(AF_INET, SOCK_STREAM)
    serSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    localAddr = ('', 7788)
    serSocket.bind(localAddr)
    serSocket.listen(5)
    print('>> %s [ TCP服务器: 启动成功!!! ]' % datetime.datetime.now())
    while True:
        try:
            print('>> %s [ TCP服务器: 主进程等待客户端请求...... ]' % datetime.datetime.now())
            newSocket, destAddr = serSocket.accept()
            print('>> %s [ TCP服务器: 收到客户端%s请求，创建客户端服务子进程. ]' % (datetime.datetime.now(), str(destAddr)))
            client = Process(target=dealClient, args=(newSocket, destAddr))
            client.start()
            print('>> %s [ TCP服务器: 关闭客户端%s ]' % (datetime.datetime.now(), str(destAddr)))
            newSocket.close()  # 因为已经向子进程中copy了一份（引用），并且父进程中这个套接字也没有用处了，所以关闭
        except Exception as er:
            print('>> %s [ TCP服务器: 服务器出错 ：%s ]' % (datetime.datetime.now(), er))
            serSocket.close()
            print('>> %s [ TCP服务器: 重启TCP服务器...... ]' % datetime.datetime.now())
            main()
        finally:
            # 当为所有的客户端服务完之后再进行关闭，表示不再接收新的客户端的链接
            serSocket.close()


if __name__ == '__main__':
    main()
