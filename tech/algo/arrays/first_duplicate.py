#!/usr/bin/python
# -*- coding: utf-8 -*-


def first_duplicate(a):
    """
    find first duplicate char of a array
    :param a:
    :return:
    """
    for i in xrange(len(a)):
        if a[abs(a[i]) - 1] < 0:
            return abs(a[i])
        else:
            a[abs(a[i]) - 1] = - a[abs(a[i]) - 1]
    return -1


if __name__ == '__main__':
    arr = [2, 3, 3, 1, 5, 2]
    print first_duplicate(arr)
