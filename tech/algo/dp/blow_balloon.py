#!/usr/bin/python
# -*- coding: utf-8 -*-


def blow_balloon(arr, dp, visit, left, right):
    """
    blow balloon: section dp
    :param arr: balloon value list
    :param dp: blow value
    :param visit: is visited?
    :param left: left index
    :param right: right index
    :return: ultimate score
    """
    if visit[left][right] == 1:
        return dp[left][right]

    res = 0
    for i in range(left, right + 1):
        midval = arr[left - 1] * arr[i] * arr[right + 1]
        leftval = blow_balloon(arr, dp, visit, left, i - 1)  # 左夹心
        rightval = blow_balloon(arr, dp, visit, i + 1, right)  # 右夹心

        res = max(res, leftval + midval + rightval)

    visit[left][right] = 1
    dp[left][right] = res
    return dp[left][right]


if __name__ == '__main__':
    a = [4, 1, 5, 10]
    length = len(a)
    arr = [1] + a + [1]
    dp = []
    visit = []
    for i in range(length + 2):
        dp.append([])
        visit.append([])
        for j in range(length + 2):
            dp[i].append(0)
            visit[i].append(0)

    left = 1
    right = length
    print(blow_balloon(arr, dp, visit, left, right))
