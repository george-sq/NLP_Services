# -*- coding: utf-8 -*-
"""
    @File   : services_regularExpression.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/12/18 14:07
    @Todo   : 
"""

import logging
import re

logger = logging.getLogger(__name__)

url_regx = r"((https?|ftp|file)://(www\.)?|www\.)[a-zA-Z0-9+&@#/%=~_|$?!:,.-]*[a-zA-Z0-9+&@#/%=~_|$]"
unitMoney_regx = r"[0-9]+(?:\s*)(?P<unit>元|人民币|美元|韩元|日元|美金|欧元|英镑)"
idcard_regx = r"(([1-9]\d{5})((18|19|2\d)\d{2}[0-1]\d[0-3]\d)(\d{3})[\dxX])|" \
              r"[1-9]\d{5}\d{2}((0[1-9])|(10|11|12))(([0-2][1-9])|10|20|30|31)\d{2}[0-9Xx]"
phoneNumber_regx = r""
bankcardId_regx = r""
email_regx = r""


def main():
    pass


if __name__ == '__main__':
    # 创建一个handler，用于写入日志文件
    logfile = "./Logs/log_regExp.log"
    fileLogger = logging.FileHandler(filename=logfile, encoding="utf-8")
    fileLogger.setLevel(logging.NOTSET)

    # 再创建一个handler，用于输出到控制台
    stdoutLogger = logging.StreamHandler()
    stdoutLogger.setLevel(logging.INFO)  # 输出到console的log等级的开关

    logging.basicConfig(level=logging.NOTSET,
                        format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S", handlers=[fileLogger, stdoutLogger])
    main()
