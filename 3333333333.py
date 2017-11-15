# -*- coding: utf-8 -*-
"""
    @File   : 3333333333.py.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/10/12 9:27
    @Todo   : 
"""

from multiprocessing import Pool
from collections import Iterable
from collections import Iterator


def f(x):
    return x * x


if __name__ == '__main__':
    r = None
    with Pool(5) as p:
        res = p.map(f, (i for i in range(10)))
        print(isinstance(res, Iterator))
        print(isinstance(res, Iterable))
        print(res)

        al = range(10)
        print(type(al))
        resi = p.imap(f, al)
        r = resi
        print(isinstance(resi, Iterator))
        print(isinstance(resi, Iterable))
        print(resi)
        print(r)
        # while True:
        #     print(resi.__next__)
        #     print(next(resi))
        for n in r:
            print(n)


