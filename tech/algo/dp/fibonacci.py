#!/usr/bin/python
# -*- coding: utf-8 -*-

"""斐波那契数列，爬楼梯问题"""


# 尾递归
def fib_tail(self, n, a, b):
    if n < 2:
        return a
    return self.fib(n - 1, b, a + b)


# 带缓存的递归
def fib_tmp(self, n):
    if n < 3:
        return n - 1, n
    tup = self.fib_tmp(n - 1)
    return tup[1], tup[0] + tup[1]


if __name__ == '__main__':
    print(fib_tmp(3))
