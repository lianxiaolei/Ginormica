# coding: utf-8


def step(s, p):
    if len(s) < p[0]:
        return 0
    if len(s) <= p[0]:
        return s[p[0] - 1]
    if len(s) <= p[1]:
        return max(s[p[0] - 1], s[p[1] - 1])
    if len(s) <= p[2]:
        return max(s[p[0] - 1], s[p[1] - 1], s[p[2] - 1])

    lis = []
    print '*' * 30
    for i in xrange(p[2] - 1, len(s)):
        mx0 = step(s[0: i + 2 - p[0]], p)
        mx1 = step(s[0: i + 2 - p[1]], p)
        mx2  = step(s[0: i + 2 - p[2]], p)
        print '-' * 10
        print i, i + 2 - p[0], i + 2 - p[1], i + 2 - p[2]
        print mx0, mx1, mx2
        print '-' * 10
        lis.append(max(mx0, mx1, mx2) + s[i])
    if not lis:
        return 0
    return max(lis)

if __name__ == '__main__':
    staircase = [11, 22, 44, 5, 12, 34, 55, 45, 23, 64]
    possible_steps = [3, 4, 5]
    print step(staircase, possible_steps)
