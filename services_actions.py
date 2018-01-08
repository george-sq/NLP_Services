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
import logging

logger = logging.getLogger(__name__)

# 创建一个handler，用于写入日志文件
logfile = './Logs/log_actionServices.log'
fileLogger = logging.FileHandler(filename=logfile, encoding="utf-8")
fileLogger.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
stdoutLogger = logging.StreamHandler()
stdoutLogger.setLevel(logging.INFO)  # 输出到console的log等级的开关

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d(%A) %H:%M:%S", handlers=[fileLogger, stdoutLogger])


def show_ctime(request_data):
    """测试0"""
    type(request_data)
    return json.dumps({"NO_ACTION Response": {"404": str(time.ctime())}})


def tst_sucessResponse(request_data):
    """测试1"""
    type(request_data)
    return json.dumps({"Sucess Response": {"200": str(time.ctime())}})


def getAnjianSimilarity(request_data):
    """案件相似度"""
    request_params, request_json = request_data
    # print("request_params :", request_params)
    # print("request_params type:", type(request_params))
    # print("request_body :", request_body)
    # print("request_body type:", type(request_body))
    result = ""
    if isinstance(request_json, str) and len(request_json) > 0:
        jsonData = json.loads(request_json)
        jsonData = dict(jsonData)
        # print("jsonData", jsonData)

        # 相似度分析
        path = "./Out/"
        mname = "model_tfidfSimilarity_anjian.pickle"
        sql = "SELECT * FROM tb_ajinfo WHERE tid=%s"
        sim_result = sim.tfidfSimilartyProcess(jsonData.get("txt", "").upper(), sql=sql, path=path, mname=mname)
        result = json.dumps({"Status": 200, "atm": dict(sim_result)}, ensure_ascii=False)

    return result


def getAtmSimilarity(request_data):
    """atm相似度"""
    request_params, request_json = request_data
    # print("request_params :", request_params)
    # print("request_params type:", type(request_params))
    # print("request_body :", request_body)
    # print("request_body type:", type(request_body))
    result = ""
    if isinstance(request_json, str) and len(request_json) > 0:
        jsonData = json.loads(request_json)
        jsonData = dict(jsonData)

        # 相似度分析
        path = "./Out/"
        mname = "model_tfidfSimilarity_atm.pickle"
        sql = "SELECT * FROM tb_atminfos WHERE id=%s"
        sim_result = sim.tfidfSimilartyProcess(jsonData.get("atm", "").upper(), sql=sql, path=path, mname=mname)
        result = json.dumps({"Status": 200, "atm": dict(sim_result)}, ensure_ascii=False)

    return result


def main():
    pass


if __name__ == '__main__':
    main()
