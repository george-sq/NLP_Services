# -*- coding: utf-8 -*-
"""
    @File   : databaseServices.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/11/15 11:17
    @Todo   : 提供关于操作数据库的服务
"""
import pymysql


class MysqlServer(object):
    """关于Mysql数据库的操作的服务类"""

    def __init__(self):
        """初始化"""
        self.host = '10.0.0.247'
        self.port = 3306
        self.user = 'pamodata'
        self.passwd = 'pamodata'
        self.db = 'db_pamodata'
        self.charset = 'utf8mb4'

    def __getConnect(self):
        """连接Mysql数据库"""
        # 创建连接
        return pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db=self.db,
                               charset=self.charset)

    def setConnect(self, **kwargs):
        """自定义设置Mysql数据库连接参数"""
        if kwargs.get('host'):
            self.host = kwargs['host']
        if kwargs.get('port'):
            self.port = kwargs['port']
        if kwargs.get('user'):
            self.user = kwargs['user']
        if kwargs.get('passwd'):
            self.passwd = kwargs['passwd']
        if kwargs.get('db'):
            self.db = kwargs['db']
        if kwargs.get('charset'):
            self.charset = kwargs['charset']

    def executeSql(self, *args):
        """
            :param args:[sql,[sql args,]]
            :return: results = [受影响的行数, (行内容,)]
        """
        """执行SQL"""
        # 获取数据库连接
        con = self.__getConnect()
        # 使用cursor()方法获取操作游标
        cursor = con.cursor()
        results = []
        try:
            if len(args) == 1:
                results.append(cursor.execute(args[0]))
                results.extend(cursor.fetchall())
            elif len(args) == 2:  # 批量操作
                results.append(cursor.executemany(args[0], args[1]))
                results.append(cursor.fetchall())
            else:
                print('输入参数错误！！！')
                raise ValueError
            con.commit()
        except Exception as e:
            # 输出异常信息
            print(args[0], "异常信息 : ", e)
            print(args[0], "异常信息 : ", e)
            print(args[0], "异常信息 : ", e)
            # 数据库回滚
            con.rollback()
        finally:
            cursor.close()
            # 关闭数据库连接
            con.close()
        return results
