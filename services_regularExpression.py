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

url_regExp = re.compile(r"((?:(?:https?|ftp|file)://(?:www\.)?|www\.)[a-zA-Z0-9+&@#/%=~_|$?!:,.-]*"
                        r"[a-zA-Z0-9+&@#/%=~_|$])")

money_regExp = re.compile(r"((?:(?:\d|one|two|three|four|five|six|seven|eight|nine|ten|一|二|两|三|四|五|六|七|八|九|十|零"
                          r"|兆|亿|万|千|百|拾|玖|捌|柒|陆|伍|肆|叁|贰|壹)+)(?:\s*(?:hundred|thousand|Million|Billion)?\s*)"
                          r"(?:元|人民币|rmb|美元|美金|dollars?|韩元|日元|欧元|英镑))",
                          re.IGNORECASE)

idcard_regExp = re.compile(r"(?<!\d)((?:(?:[1-9]\d{5})(?:(?:18|19|2\d)\d{2}[0-1]\d[0-3]\d)(?:\d{3})[\dxX])|"
                           r"[1-9]\d{5}\d{2}(?:(?:0[1-9])|(?:10|11|12))(?:(?:[0-2][1-9])|10|20|30|31)\d{2}[0-9Xx])"
                           r"(?!\d)")

phoneNumber_regExp = re.compile(r"((?<!\d)(?:\(?\+?(?:[0-9]{1,4} ?)?(?:\)|\) |-| - )*)(?:[1-9]\d{7,10})(?!\d))")

bankCard_regExp = re.compile(r"((?<![0-9_+=-])([\d]{6})([\d]{6,12})[\d ]?(?!\d))")

email_regExp = re.compile(r"((?:(?:[a-z0-9+.']+)|(?:\"\w+\\ [a-z0-9']+\"))@"
                          r"(?:(?:[a-z0-9]+|\[)+(?:\.(?!\.+)))+(?:(?:[a-z0-9]+|\])+)?)", re.IGNORECASE)


def urlMatch(inStr):
    # 测试url的正则表达式
    rUrl = url_regExp.findall(inStr)
    print(">>findall() :", rUrl)
    # 获取匹配结果
    # 统计匹配结果
    # 构建索引集合
    l = len(rUrl)
    indexs = [0]
    for i in range(l):
        url = rUrl[i]
        if 0 == len(indexs):
            idx = inStr.find(url)
            print(idx, len(url))
        else:
            start = indexs[-1] + len(rUrl[i - 1])
            idx = inStr.find(url, start)
        indexs.append(idx)
    return url_regExp.findall(inStr)


def emailMatch(inStr):
    # 测试电子邮件地址的正则表达式
    rEmail = email_regExp.search(inStr)
    print(">>group() :", rEmail.group())
    print(">>groups() :", rEmail.groups())
    print(">>findall() :", email_regExp.findall(inStr))
    return email_regExp.findall(inStr)


def moneyMatch(inStr):
    # 测试货币金额的正则表达式
    rMoney = money_regExp.search(inStr)
    print(">>group() :", rMoney.group())
    print(">>groups() :", rMoney.groups())
    print(">>findall() :", money_regExp.findall(inStr))
    return money_regExp.findall(inStr)


def idcardMatch(inStr):
    # 测试身份证的正则表达式
    rIdcard = idcard_regExp.search(inStr)
    print(">>group() :", rIdcard.group())
    print(">>groups() :", rIdcard.groups())
    print(">>findall() :", idcard_regExp.findall(inStr))
    return idcard_regExp.findall(inStr)


def phoneMatch(inStr):
    # 测试电话的正则表达式
    rPhone = phoneNumber_regExp.search(inStr)
    print(">>group() :", rPhone.group())
    print(">>groups() :", rPhone.groups())
    print(">>findall() :", phoneNumber_regExp.findall(inStr))
    return phoneNumber_regExp.findall(inStr)


def bankCardMatch(inStr):
    # 测试银行卡的正则表达式
    rBankCard = bankCard_regExp.search(inStr)
    print(">>group() :", rBankCard.group())
    print(">>groups() :", rBankCard.groups())
    print(">>findall() :", bankCard_regExp.findall(inStr))
    return bankCard_regExp.findall(inStr)


def main():
    txt = """
            软件费用: 20元/月，50人民币/季度，160美元/年，220美金/2年，1000欧元/5年，2000英镑/10年，2万日元/月，八十万 韩元/年
            在线演示
            http://jiebademo.ap01.aws.af.cm/
            网站代码：https://github.com/fxsjy/jiebademo
            全自动安装：easy_install jieba 或者 pip install jieba / pip3 install jieba
            半自动安装：先下载 http://pypi.python.org/pypi/jieba/ ，解压后运行 python setup.py install
            手动安装：将 jieba 目录放置于当前目录或者 site-packages 目录
            通过 import jieba 来引用
            范例：
            自定义词典：https://github.com/fxsjy/jieba/blob/master/test/userdict.txt
            用法示例：https://github.com/fxsjy/jieba/blob/master/test/test_userdict.py
            联系我
            我的博客: http://blog.csdn.net/qibin0506
            我的邮箱: qibin0506@gmail.com
            联系电话: 13507453457
            身份证号: 431202198906283548/43120219890628357x/43120219890628327X

            银行卡号：6228480402564890018（19位）,农业银行.金穗通宝卡（个人普卡），中国银联卡
            举例：625965087177209（不含校验码的15位银行卡号）
            查询的银行卡号： 6212262102012020709 （19位）
            http://jiebademo.ap01.aws.af.cm/
            """

    # 测试url的正则表达式
    print(">>测试url的正则表达式")
    rUrl = urlMatch(txt)

    print()
    print("**********" * 15)

    # 测试电子邮件地址的正则表达式
    print(">>测试电子邮件地址的正则表达式")
    rEmail = emailMatch(txt)

    print()
    print("**********" * 15)

    # 测试货币金额的正则表达式
    print(">>测试货币金额的正则表达式")
    rMoney = moneyMatch(txt)

    print()
    print("**********" * 15)

    # 测试身份证的正则表达式
    print(">>测试身份证的正则表达式")
    rIdcard = idcardMatch(txt)

    print()
    print("**********" * 15)

    # 测试电话的正则表达式
    print(">>测试电话的正则表达式")
    rPhone = phoneMatch(txt)

    print()
    print("**********" * 15)

    # 测试银行卡的正则表达式
    print(">>测试银行卡的正则表达式")
    rBankCard = bankCardMatch(txt)

    print()
    print("**********" * 15)


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
