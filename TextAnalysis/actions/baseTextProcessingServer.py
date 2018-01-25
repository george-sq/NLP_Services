# -*- coding: utf-8 -*-
"""
    @File   : baseTextProcessingServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/19 10:05
    @Todo   : 
"""

from mysqlServer import MysqlServer
import keyWordRecognitionServer as kws
import logging

logger = logging.getLogger(__name__)


def main():
    # 数据库连接
    mysql = MysqlServer(host="10.0.0.247", db="db_pamodata", user="pamo", passwd="pamo")
    # 查询结果
    # sql = "SELECT * FROM corpus_rawtxts WHERE txtLabel<>'电信诈骗' ORDER BY txtId LIMIT 70,5"
    sql = "SELECT * FROM corpus_rawtxts ORDER BY txtId"
    queryResult = mysql.executeSql(sql=sql)
    queryResult = [[record[0], record[2], record[3].replace(" ", "")] for record in queryResult[1:]]
    # 数字处理
    corpus = []
    for record in queryResult:
        tid, tlabel, txt = record
        # print(txt.replace("\u3000", "").replace("\n", ""))
        txts = [w for w in txt.replace("\n", "").replace("\u3000", "")]
        for i in range(len(txts)):
            # print(txts[i])
            al = ["①", "②", "③", "④", "⑤", "⑥", "⑴", "⑵", "⑶", "⑷", "⑸", "⑹", "⑺"]
            if txts[i] not in al and txts[i].isdigit():
                txts[i] = str(int(txts[i]))
        t = "".join(txts)
        corpus.append([t, tid])
        print(tid)
        print("**********" * 20)

    # 更新语料库
    update_sql = "UPDATE corpus_rawtxts SET txtContent=%s WHERE txtId=%s"
    r = mysql.executeSql(sql=update_sql, args=corpus)
    print(r)

    # 切分标注
    # queryResult = [(record[0], record[2], record[3]) for record in queryResult[1:]]
    # retVal = []
    # for tid, l, txt in queryResult:
    #     retVal.append(kws.fullMatch([tid, txt.replace(" ", "")]))
    #
    # print(type(retVal))
    # print(len(retVal))
    # for i in retVal:
    #     for a in i:
    #         if isinstance(a, list):
    #             for s in a:
    #                 print(s)
    #         else:
    #             print(a)
    #     print("##########" * 20)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
