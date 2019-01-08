# coding:utf8

import random as rd
import math


def sqrt(t, alpha=0.001, eps=0.0001):
    """
    Calculate the sqrt of the t
    :param t: value
    :return: sqrt(t)
    """
    x = rd.random()
    dy = x ** 2 - t
    i = 0
    while True:
        x = x - alpha * dy
        dy = x ** 2 - t
        if abs(x ** 2 - t) < eps:
            break
        i += 1
    print('gd iter times:', i)
    return x


def sqrt_newton(t, alpha=1., eps=0.0001):
    """
    Calculate the sqrt of the t with newton
    :param t: value
    :return: sqrt(t)
    :param t:
    :param alpha:
    :param eps:
    :return:
    """
    x = rd.random()
    i = 0
    while True:
        dy = x ** 2 - t
        ddy = 2 * x
        x = x - alpha * dy / ddy

        if abs(x ** 2 - t) < eps:
            break
        i += 1
    print('newton iter times:', i)
    return x


def sqrt_nt(t, alpha=1., eps=0.0001):
    x = rd.random()
    i = 0
    while True:
        y = x ** 2 - t
        dy = 2 * x
        x = x - y / dy
        print(x, dy)
        if abs(x ** 2 - t) < eps:
            break
        i += 1

    return x


if __name__ == '__main__':
    x = 3
    # sq = sqrt(x)
    # sqn = sqrt_newton(x)
    # print('gradient descend:', sq)
    # print('newton function:', sqn)
    # print('system function:', math.sqrt(x))
    # print('gd error:', sq - math.sqrt(x))
    # print('nt error:', sqn - math.sqrt(x))
    print(sqrt_nt(3))