# coding = utf-8
"""
    @File   : dataBases.py
    @Author : NLP_QingShen (275171387@qq.com)
    @Time   : 2017/9/18 15:34
    @Todo   : 
"""

import pymysql


class useMysql:
    """连接Mysql数据库的相关操作"""

    def __init__(self):
        """初始化"""
        self.host = '10.0.0.247'#'192.168.107.131'
        self.port = 3306
        self.user = 'pamo'#'mybatis'
        self.passwd = 'pamo'#'mybatis'
        self.db = 'textcorpus'
        self.charset = 'utf8mb4'

    def setConnect(self, **kwargs):
        """设置Mysql数据库连接参数"""
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

    def __getConnect(self):
        """连接Mysql数据库"""
        # 创建连接
        return pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db=self.db,
                               charset=self.charset)

    def tableCotrol(self, sql):
        """表控制:创建表和删除表"""
        # 获取数据库连接
        con = self.__getConnect()
        # 使用cursor()方法获取操作游标
        cursor = con.cursor()
        results = []
        try:
            results.append(cursor.execute(sql))
        except Exception as e:
            # 输出异常信息
            print(sql, "异常信息 : ", e)
            print(sql, "异常信息 : ", e)
            print(sql, "异常信息 : ", e)
            # 数据库回滚
            con.rollback()
        finally:
            cursor.close()
            # 关闭数据库连接
            con.close()
        return results

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
                # r1 = cursor.execute(args[0])
                # r2 = cursor.fetchall()
                cursor.execute(args[0])
                # results.append(cursor.execute(args[0]))
                # results.append(cursor.fetchall())
                results.extend(cursor.fetchall())
            elif len(args) == 2:
                # results.append(cursor.executemany(args[0], args[1]))
                # results.append(cursor.fetchall())
                cursor.executemany(args[0], args[1])
                results.extend(cursor.fetchall())
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