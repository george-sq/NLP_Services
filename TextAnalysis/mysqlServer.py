# -*- coding: utf-8 -*-
"""
    @File   : mysqlServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/16 13:46
    @Todo   : 
"""

import logging
import pymysql

logger = logging.getLogger(__name__)


class MysqlServer(object):
    """关于Mysql数据库的操作的服务类"""

    def __init__(self, **kwargs):
        """初始化"""
        self.host = kwargs.get("host", None)
        self.user = kwargs.get("user", None)
        self.passwd = kwargs.get("passwd", None)
        self.db = kwargs.get("db", None)
        self.port = 3306
        self.charset = kwargs.get("charset", "utf8mb4")

    def getConnect(self):
        """连接Mysql数据库"""
        # 创建连接
        if self.host and self.user and self.passwd and self.db:
            logger.info("Connect mysql database [%s] : %s:%s User=%s" % (self.db, self.host, self.port, self.user))
            try:
                con = pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd,
                                      db=self.db, charset=self.charset)
                return con
            except Exception as e:
                logger.error("Can not connect mysql database [%s] : %s:%s User=%s" %
                             (self.db, self.host, self.port, self.user))
                logger.error("Connect mysql database Error : %s" % e)
                pass
        else:
            logger.error("Can not connect mysql database [%s] : %s:%s User=%s" %
                         (self.db, self.host, self.port, self.user))

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

    def executeSql(self, **kwargs):
        """
        :param kwargs: sql, params
        :return: results = [受影响的行数, [返回结果的行内容,]]
        """
        # 获取数据库连接
        con = self.getConnect()
        if con is not None:
            # 使用cursor()方法获取操作游标
            cursor = con.cursor()
            results = []
            sql = kwargs.get("sql", None)
            params = kwargs.get("args", None)
            try:
                if sql and params:
                    logger.info("Execute SQL : %s" % sql)
                    logger.info("SQL Params : %s ......" % params[:10])
                    results.append(cursor.executemany(sql, params))
                    results.append(cursor.fetchall())
                    logger.info("Fetch All : have %s records" % results[0])
                elif sql and params is None:  # 批量操作
                    logger.info("Execute SQL : %s" % sql)
                    results.append(cursor.execute(sql))
                    results.extend(cursor.fetchall())
                    logger.info("Fetch All : have %s records" % results[0])
                else:
                    logger.error("输入参数错误！！！")
                    raise ValueError
                con.commit()
            except Exception as e:
                # 输出异常信息
                logger.error("Mysql Error : %s" % e)
                # 数据库回滚
                con.rollback()
            finally:
                cursor.close()
                # 关闭数据库连接
                con.close()
            return results
        else:
            logger.error("Can not connect mysql database [%s] : %s:%s User=%s" %
                         (self.db, self.host, self.port, self.user))


class Application(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            logger.info("App方法获得的参数 : %s" % args)
        mysql = MysqlServer(host="10.0.0.247", db="db_pamodata", user="pamo", passwd="pamo")
        sql = "SELECT * FROM tb_ajinfo ORDER BY tid LIMIT 10"
        retVal = mysql.executeSql(sql=sql)
        return str(retVal[0])


app = Application()


def main():
    pass


if __name__ == '__main__':
    main()
