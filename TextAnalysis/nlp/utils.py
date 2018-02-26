# -*- coding: utf-8 -*-
"""
    @File   : utils.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2018/2/26 10:12
    @Todo   : 
"""

from sklearn.feature_extraction.text import HashingVectorizer


def convertfullwidth(content):
    """ 将中文全角字符转换成半角字符
    :param content: 原始字符串
    :return: 半角字符串
    """
    new_str = ""
    for each_char in content:
        char_code = ord(each_char)
        if 12288 == char_code:
            char_code = 32
        elif 65281 <= char_code <= 65374:
            char_code -= 65248
        new_str += chr(char_code)
    return new_str


def txt2sparsematrix(txts):
    """ 将文本转化成稀疏矩阵
    :param txts: 文本列表
    :return:
    """
    vector_model = HashingVectorizer(non_negative=True)
    sparsematrix = vector_model.fit_transform(txts)
    return sparsematrix


def main():
    txt = "中国电影《红海行动》 very good。!. 12345abcde＋－＊／＝？"
    print("Raw : %s" % txt)
    new_txt = convertfullwidth(txt)
    print("Converted : %s" % new_txt)
    sm = txt2sparsematrix([new_txt])
    print(sm)


if __name__ == '__main__':
    main()
