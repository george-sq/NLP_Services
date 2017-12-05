# -*- coding: utf-8 -*-
"""
    @File   : services_userLogger.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/12/5 10:00
    @Todo   : 
"""

import logging

userLogger = logging.getLogger("userLogger")
userLogger.setLevel(logging.DEBUG)

# 第二步，创建一个handler，用于写入日志文件
logfile = './Out/log.txt'
fileLogger = logging.FileHandler(logfile)
fileLogger.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

# 第三步，再创建一个handler，用于输出到控制台
stdoutLogger = logging.StreamHandler()
stdoutLogger.setLevel(logging.INFO)  # 输出到console的log等级的开关

# 第四步，定义handler的输出格式
formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(filename)s[line:%(lineno)s] | %(message)s",
                              datefmt="%Y-%m-%d(%A) %H:%M:%S")
fileLogger.setFormatter(formatter)
stdoutLogger.setFormatter(formatter)

userLogger.addHandler(fileLogger)
userLogger.addHandler(stdoutLogger)


def main():
    userLogger.debug('this is a logger debug message')
    userLogger.info('this is a logger info message')
    userLogger.warning('this is a logger warning message')
    userLogger.error('this is a logger error message')
    userLogger.critical('this is a logger critical message')
    pass


if __name__ == '__main__':
    main()
