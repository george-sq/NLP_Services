# -*- coding: utf-8 -*-
"""
    @File   : services_actions.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/8 15:47
    @Todo   : 
"""

import json
import time
import services_similarity4tfidf as sim
import services_database as dbs
import logging

logger = logging.getLogger(__name__)

# 功能执行的返回状态代码字典
RESULT_CODES = {"f": 0, "s": 1}


def useMysql(sql):
    retVal = []
    if len(sql) > 0:
        q = dbs.MysqlServer().executeSql(sql)
        retVal.extend(q[1:])
    return retVal


def show_ctime(request_data):
    """测试0"""
    request_params, request_json = request_data
    result = {"Root Response": {"RESULT_CODES": 0, "RESULT": None}}
    rsp = result.get("Root Response")
    rsp["RESULT"] = str(time.ctime())
    if -1 != request_json:
        rsp["RESULT_CODES"] = RESULT_CODES["s"]
    result = json.dumps(result, ensure_ascii=False)

    return result


def getAnjianSimilarity(request_data):
    """案件相似度"""
    request_params, request_json = request_data
    result = {"Action Response": {"RESULT_CODES": 0, "RESULT": None}}
    if isinstance(request_json, str) and len(request_json) > 0:
        jsonData = dict(json.loads(request_json))
        # 相似度分析
        path = "./Out/"
        mname = "model_tfidfSimilarity_anjian.pickle"
        logger.info("[ 服务子进程 ] 客户端请求的JSON数据 : %s" % str(jsonData))
        sim_result = sim.getTfidfSimilarty(jsonData.get("aj", "").upper(), path=path, mname=mname)
        logger.info("[ 服务子进程 ] TFIDF相似性分析结果 : 相似案件集合 > %s" % sim_result)
        sql = "SELECT * FROM tb_ajinfo WHERE tid=%s"
        logger.info("[ 服务子进程 ] 执行SQL : %s" % sql)
        qr = [dbs.MysqlServer().executeSql(sql % s)[-1][:2] for s in sim_result]
        for index, record in enumerate(qr):
            logger.info("[ 服务子进程 ] #L%d tid=%s txt=%s" % (index + 1, record[0], record[1]))
        rsp = result.get("Action Response")
        if len(qr) > 0:
            rsp["RESULT_CODES"] = RESULT_CODES["s"]
            rsp["RESULT"] = dict(qr)
        result = json.dumps(result, ensure_ascii=False)

    return result


def getAtmSimilarity(request_data):
    """atm相似度"""
    request_params, request_json = request_data
    result = {"Action Response": {"RESULT_CODES": 0, "RESULT": None}}
    if isinstance(request_json, str) and len(request_json) > 0:
        jsonData = dict(json.loads(request_json))
        # 相似度分析
        path = "./Out/"
        mname = "model_tfidfSimilarity_atm.pickle"
        logger.info("[ 服务子进程 ] 客户端请求的JSON数据 : %s" % str(jsonData))
        sim_result = sim.getTfidfSimilarty(jsonData.get("atm", "").upper(), path=path, mname=mname)
        logger.info("[ 服务子进程 ] TFIDF相似性分析结果 : 相似ATM集合 > %s" % sim_result)
        sql = "SELECT * FROM tb_atminfos WHERE id=%s"
        logger.info("[ 服务子进程 ] 执行SQL : %s" % sql)
        qr = [dbs.MysqlServer().executeSql(sql % s)[-1][:2] for s in sim_result]
        for index, record in enumerate(qr):
            logger.info("[ 服务子进程 ] #L%d id=%s atm=%s" % (index + 1, record[0], record[1]))
        rsp = result.get("Action Response")
        if len(qr) > 0:
            rsp["RESULT_CODES"] = RESULT_CODES["s"]
            rsp["RESULT"] = dict(qr)
        result = json.dumps(result, ensure_ascii=False)

    return result


def main():
    pass


if __name__ == '__main__':
    main()
