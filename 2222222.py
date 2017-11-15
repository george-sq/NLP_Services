# -*- coding: utf-8 -*-
"""
    @File   : 2222222.py.py
    @Author : NLP_QiangShen (275171387@qq.com)
    @Time   : 2017/10/10 16:04
    @Todo   : 
"""

from multiprocessing import Process, Pool
from multiprocessing.dummy import Pool as DummyPool
import time


def log_time(methond_name):
    def decorator(f):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            res = f(*args, **kwargs)
            end_time = time.time()
            print('%s cost %ss' % (methond_name, (end_time - start_time)))
            return res
        return wrapper
    return decorator


def fib(n):
    if n <= 2:
        return 1
    return fib(n - 1) + fib(n - 2)


@log_time('single_process')
def single_process():
    fib(33)
    fib(33)


@log_time('multi_process')
def multi_process():
    jobs = []
    for _ in range(2):
        p = Process(target=fib, args=(33, ))
        p.start()
        jobs.append(p)
    for j in jobs:
        j.join()


@log_time('pool_process')
def pool_process():
    pool = Pool(2)
    pool.map(fib, [33] * 2)


@log_time('dummy_pool')
def dummy_pool():
    pool = DummyPool(2)
    pool.map(fib, [33] * 2)


if __name__ == '__main__':
    single_process()
    multi_process()
    pool_process()
    dummy_pool()

