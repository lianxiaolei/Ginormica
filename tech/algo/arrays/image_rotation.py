#!/usr/bin/python
# -*- coding: utf-8 -*-


def image_rotation(a):
    n = len(a)
    for i in range(n - 1):
        for j in range(i + 1, n):
            tmp = a[i, j]
            a[i, j] = a[j, i]
            a[j, i] = tmp
    for i in range(n):
        for j in range(n / 2):
            tmp = a[i, j]
            a[i, j] = a[i, n - 1 - j]
            a[i, n - 1 - j] = tmp
    return a


rotate_image_awsome = lambda a: zip(*a[::-1])


if __name__ == '__main__':
    import numpy as np
    a = np.linspace(1, 16, 16).reshape(4, 4)
    print 'rotation--'
    print image_rotation(a)