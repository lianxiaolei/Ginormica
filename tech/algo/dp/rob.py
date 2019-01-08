# coding:utf8


def dp(l):
    if len(l) == 1: return 0, l[0]
    if len(l) == 2: return l[0], l[1]
    if len(l) == 3: return l[1], l[0] + l[2]
    if len(l) == 4: return l[0] + l[2], max(l[0] + l[3], l[1] + l[3])
    # maxval = 0
    # maxval += max(rob(l[:-1]), rob(l[:-2]) + l[-1])
    val = dp(l[:-1])
    return val[1], val[0] + l[-1]


def rob(l):
    if len(l) == 0: return 0
    if len(l) == 1: return l[0]
    if len(l) == 2: return max(l[0], l[1])
    if len(l) == 3: return max(l[1], l[0] + l[2])
    if len(l) == 4: return max(l[0] + l[2], max(l[0] + l[3], l[1] + l[3]))
    val = dp(l)

    return val[0] if val[0] > val[1] else val[1]


if __name__ == '__main__':
    print('max value', rob([1, 2, 3, 1]))
    print('max value', rob([2, 7, 9, 3, 1]))
    print('max value', rob([2, 1, 1, 2]))
    print('max value', rob([6, 3, 10, 8, 2, 10, 3, 5, 10, 5, 3]))
    # print('max value', rob(
    #     [183, 219, 57, 193, 94, 233, 202, 154, 65, 240, 97, 234, 100, 249, 186, 66, 90, 238, 168, 128, 177, 235, 50, 81,
    #      185, 165, 217, 207, 88, 80, 112, 78, 135, 62, 228, 247, 211]))
