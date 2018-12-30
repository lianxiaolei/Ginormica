# coding:utf8

import numpy as np


def lcs_dp(s0, s1, dp, ii, jj):
    """
    动态规划
    :param s0:
    :param s1:
    :param dp:
    :param ii:
    :param jj:
    :return:
    """
    if len(s0) == 1 and len(s1) == 1:
        return 1 if s0 == s1 else 0
    for i in range(ii, len(s0) - 1):
        for j in range(jj, len(s1) - 1):
            if s0[i] == s1[j]:
                lcs_dp(s0[i + 1:], s1[j + 1:], dp, i, j)
                dp[i, j] = max(dp[i + 1, j], dp[i, j + 1]) + 1
    return np.max(dp)


def search(s0, s1):
    dp = np.zeros((100, 100), dtype=np.int8)

    if len(s0) == 1:
        if s0 in s1:
            return True
    if len(s1) == 1:
        if s1 in s0:
            return True
    return lcs_dp(s0, s1, dp, 0, 0)


def stupid(s0, s1):
    """
    弱智级
    :param s0:
    :param s1:
    :return:
    """
    smax = s0 if len(s0) >= len(s1) else s1  # 长串
    smin = s1 if len(s1) <= len(s0) else s0  # 短串
    # substr = ''
    maxlen = 0
    templen = 0
    for i in range(len(smax) - len(smin) + 1):
        ii = i
        for j in range(len(smin)):
            if smax[ii] == smin[j]:
                # substr += smax[ii]
                templen += 1
                ii += 1
            else:
                maxlen = maxlen if maxlen > templen else templen
                templen = 0
                # substr = ''
                ii = i
        if maxlen < templen:
            maxlen = templen
    return maxlen


if __name__ == '__main__':
    # print('The longest common subsequence is:', stupid('abcddffgiem', 'cddejjghem'))
    # print('The longest common subsequence is:', stupid('bedaacbade', 'dccaeedbeb'))
    print('The longest common subsequence is:', search('bedaacbade', 'dccaeedbeb'))
    print('The longest common subsequence is:', search('abcddffgiem', 'cddejjghem'))
