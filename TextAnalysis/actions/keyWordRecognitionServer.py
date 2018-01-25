# -*- coding: utf-8 -*-
"""
    @File   : keyWordRecognitionServer.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/1/19 13:25
    @Todo   : 
"""

import logging
import re
import jieba

jieba.setLogLevel(logging.INFO)
jieba.set_dictionary("../../Dicts/dict_jieba_check.txt")
with open("../../Dicts/newWords.txt", "r", encoding="utf-8") as nw:
    newWords = nw.readlines()
    for line in newWords:
        w, t = line.split(" ")
        jieba.add_word(word=w, tag=t.strip())
        print("%s(%s) Freq : %s" % (w, t.strip(), str(jieba.suggest_freq("茶坞"))))
jieba.enable_parallel(4)
from jieba import posseg

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

phoneNumber_regExp = re.compile(r"(?<!\d)(?:([(+（]{0,2})?(?: ?[0-9]{2,4} ?)(?:[)-）] ?)?)?(?:1[3-9]\d{9})(?!\d)")

bankCard_regExp = re.compile(r"((?<![0-9_+=-])(?:[\d]{6})(?:[\d]{6,12})[\d ]?(?!\d))")

email_regExp = re.compile(r"((?:(?:[a-z0-9+.']+)|(?:\"\w+\\ [a-z0-9']+\"))@"
                          r"(?:(?:[a-z0-9]+|\[)+(?:\.(?!\.+)))+(?:(?:[a-z0-9]+|\])+)?)", re.IGNORECASE)

time_regExp = re.compile(r"(?:(?:上午|中午|下午|凌晨|早上|晚上|午夜|半夜)?(?:(?:(?:[0-1]\d|2[0-3]|(?<!\d)\d(?!\d))(?:点钟?|时))"
                         r"(?:(?:过?(?:(?:(?<!\d)\d(?!\d))|(?:(?<!\d)[0-5]\d(?!\d)))分)(?:过?(?:(?:(?<!\d)\d(?:\.\d+)?"
                         r"(?!\d))|(?:(?<!\d)[0-5]\d(?:\.\d+)?(?!\d)))秒)?|(?:一刻钟|一刻|半|多))?|"
                         r"(?:(?:(?:[零一二两三四五六七八九十]|(?:十|一十)[一二三四五六七八九]|二十[一二三四])(?:点钟?|时))"
                         r"(?:(?:(?:过?(?:(?:(?:(?:十|一十|二十|三十|四十|五十)[一二三四五六七八九]?)(?![一二两三四五六七八九十])?)"
                         r"|(?:(?<![一二两三四五六七八九十])零?[一二两三四五六七八九](?![一二两三四五六七八九十])))分)(?:过?"
                         r"(?:(?:(?:(?:十|一十|二十|三十|四十|五十)[一二三四五六七八九]?)(?![一二两三四五六七八九十])?)|"
                         r"(?:(?<![一二两三四五六七八九十])零?[一二两三四五六七八九](?![一二两三四五六七八九])))秒)?)|"
                         r"(?:过?(?:一刻钟|一刻|半|多)|(?:过?(?:(?:十|一十|二十|三十|四十|五十)[一二三四五六七八九]?)"
                         r"(?![一二两三四五六七八九十])?)|(?:(?<![一二两三四五六七八九十])过[一二两三四五六七八九]"
                         r"(?![一二两三四五六七八九]))))?)))|(?:(?<!\d)(?:[0-2]?[0-9][：:]+[0-5]?[0-9])[:：]?"
                         r"(?:[0-5]?[0-9]\.?[0-9]+)?(?:\s*(?:am|pm))?)|"
                         r"(?:(?:(?:一刻|半刻|半|零|一|二|两|三|四|五|六|七|八|九|十|百|千|万|百万|千万|亿|兆|\d)+"
                         r"(?:多个|多|个)?\s*(?:世纪|年|月|季度|日子|日|天|小时|分钟|秒钟|秒|毫秒|钟))|(?:[0-9]+\s*"
                         r"(?:years|year|months|month|days|day|hours|hour|hr\.|Minutes|Minute|second|secs\.|sec\.|"
                         r"Millisecond|msec\.|msel|ms)+))", re.IGNORECASE)

date_regExp = re.compile(r"(?:(?<!\d)(?:3[01]|[12][0-9]|0?[1-9])(?!\d)[ /.-])(?:(?<!\d)(?:1[012]|0?[1-9])(?!\d)[ /.-])"
                         r"(?:(?<!\d)(?:19|20)[0-9]{2}(?!\d))|(?:(?<!\d)(?:1[012]|0?[1-9])(?!\d)[ /.-])(?:(?<!\d)"
                         r"(?:3[01]|[12][0-9]|0?[1-9])(?!\d)[ /.-])(?:(?<!\d)(?:19|20)[0-9]{2}(?!\d))|(?:(?<!\d)"
                         r"(?:(?:19|20)[0-9]{2})(?!\d)[ 年/.-])(?:(?<!\d)(?:1[012]|0?[1-9])(?!\d)[ 月/.-])(?:(?<!\d)"
                         r"(?:3[01]|[12][0-9]|0?[1-9])(?!\d)[日|号]?)|(?:(?<!\d)(?:1[012]|0?[1-9])(?!\d)[月/.-])"
                         r"(?:(?:3[01]|[12][0-9]|0?[1-9])(?!\d)[日|号]?)")

num_regExp = re.compile(r"(?<!\d)(?:\d|一|二|两|三|四|五|六|七|八|九|十|百|千|万|百万|千万|亿|兆)+(?!\d)")

regExpSets = {"url": url_regExp, "email": email_regExp, "money": money_regExp, "num": num_regExp, "date": date_regExp,
              "idcard": idcard_regExp, "phnum": phoneNumber_regExp, "bkcard": bankCard_regExp, "time": time_regExp}


def getKeyWords(inList, regExpK=None):
    """
    :param inList: [[txt],]
    :param regExpK: 正则表达式规则字典集合的关键字索引
    :return: [[keyWord, label],]
    """
    retVal = []
    regExp = regExpSets.get(regExpK, None)
    for i in range(len(inList)):
        sub = inList[i]
        results = []
        if 1 == len(sub):
            content = sub[0].strip()
            if 0 != len(content):
                if isinstance(regExp, type(re.compile(""))):
                    resultSet = regExp.findall(content)
                    # 根据正则表达式的匹配结果处理输入inStr
                    if len(resultSet) > 0:
                        post = content
                        for res in resultSet:
                            idx = post.find(res)
                            if idx is not None:
                                pre = post[:idx].strip()
                                if len(pre) > 0:
                                    results.append([pre])
                                results.append([res, regExpK])
                                idx += len(res)
                                post = post[idx:].strip()
                        endPart = post.strip()
                        if len(endPart) > 0:
                            results.append([endPart])
                else:
                    # 分词处理
                    rPos = [[item, pos] for item, pos in posseg.lcut(content, HMM=False) if len(item.strip()) > 0]
                    results.extend(rPos)
            else:
                logger.warning("处理内容的长度错误 len = 0")
                print("处理内容的长度错误 len = 0")
                print("sub : %s" % sub)
        if len(results) > 0:
            retVal.extend(results)
        elif len(sub[0].strip()) > 0:
            retVal.append(sub)

    return retVal


def fullMatch(record):
    """
    :param record: [tid, txt]
    :return: (tid, [(item, label),])
    """
    tid = record[0]
    # print(tid)
    inStr = record[1]
    # print(inStr)

    # url处理
    step1 = getKeyWords([[inStr]], regExpK="url")

    # email处理
    step2 = getKeyWords(step1, regExpK="email")

    # money处理
    step3 = getKeyWords(step2, regExpK="money")

    # idcard处理
    step4 = getKeyWords(step3, regExpK="idcard")

    # bankcard处理
    step5 = getKeyWords(step4, regExpK="bkcard")

    # date处理
    step6 = getKeyWords(step5, regExpK="date")

    # time处理
    step7 = getKeyWords(step6, regExpK="time")

    # phone处理
    step8 = getKeyWords(step7, regExpK="phnum")

    # num处理
    # step9 = getKeyWords(step8, regExpK="num")

    # 未标注内容的分词处理
    step10 = getKeyWords(step8)

    # 修改时间词汇标记
    # for i in range(len(step10)):
    #     if "t" == step10[i][-1] or "tg" == step10[i][-1]:
    #         step10[i][-1] = "time"

    return tid, step10


def main():
    txt = """
            软件费用: 20元/月，50人民币/季度，160美元/年，220美金/2年，1000欧元/5年，2000英镑/10年，2万日元/月，八十万 韩元/年
            在线演示 QQ上买手机被骗一千，在此地汇款，请妥处
            事发地11点多接到诈骗电话，支付宝汇款10万元，请核实
            2:15 12:36
            12:36:45.3654 12:36 am  12:36 AM 12:36pm 12:36PM1 12:36am啊 12:36AMa 
            12:35:34 12:35:34 pm 12:35:34 PM 12:35:34 am 12:35:34 AM 12:35:34am 12:35:34AM 12:35:34pm 12:35:34PM
            13 pm	16: 
            12：35：34 13点24分  13点8分	13点07分 13点07分46秒	13点07分46秒 
            06时 一点 三点一刻 两点半 十二点多 凌晨 四点钟 5点一刻钟	23时
            11时38分 11时28分54秒 11时28分54 
            13点46分     1点多	下午3点 上午5点 凌晨1点 六点三十三分 七点零九分
                        1999/12/31            1999.12.31            1999 12 31            2048年10月3日
            2048年10月6日 2048年10月6号
            http://jiebademo.ap01.aws.af.cm/
            网站代码：https://github.com/fxsjy/jiebademo
            全自动安装：easy_install jieba 或者 pip install jieba / pip3 install jieba
            半自动安装：先下载 http://pypi.python.org/pypi/jieba/ ，解压后运行 python setup.py install
            手动安装：将 jieba 目录放置于当前目录或者 site-packages 目录
            通过 import jieba 来引用
            范例：
            自定义词典：https://github.com/fxsjy/jieba/blob/master/test/userdict.txt
            用法示例：https://github.com/fxsjy/jieba/blob/master/test/test_userdict.py
            联系方式
            我的博客: http://blog.csdn.net/qibin0506
            我的邮箱: qibin0506@gmail.com
            联系电话: 13507453457
            身份证号: 431202198906283548/43120219890628357x/43120219890628327X

            银行卡号：6228480402564890018（19位）,农业银行.金穗通宝卡（个人普卡），中国银联卡
            举例：625965087177209（不含校验码的15位银行卡号）
            查询的银行卡号： 6212262102012020709 （19位）
            http://jiebademo.ap01.aws.af.cm/
            """

    result = fullMatch((0, txt))
    for s in result[1]:
        print(s)


if __name__ == '__main__':
    logging.basicConfig(level=logging.NOTSET,
                        format="%(asctime)s | %(levelname)s | %(filename)s(line:%(lineno)s) | %(message)s",
                        datefmt="%Y-%m-%d(%A) %H:%M:%S")
    main()
