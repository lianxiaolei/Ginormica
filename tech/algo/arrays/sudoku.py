#!/usr/bin/python
# -*- coding: utf-8 -*-


def sudoku2(grid):
    """
    sudoku check（自己实现的辣鸡版）
    :param grid:
    :return:
    """
    row = {}
    col = {}
    mat = {}
    for i in xrange(len(grid)):
        row.setdefault(i, [])
        for j in xrange(len(grid[i])):
            if grid[i][j] == '.':
                continue
            if not row[i].count(grid[i][j]) == 0:
                return False
            row[i].append(grid[i][j])
            # col
            col.setdefault(j, [])
            if not col[j].count(grid[i][j]) == 0:
                return False
            col[j].append(grid[i][j])
            # mat
            r = i / 3
            c = j / 3
            mat.setdefault('%s%s' % (r, c), [])
            if not mat.get('%s%s' % (r, c)).count(grid[i][j]) == 0:
                return False
            else:
                mat['%s%s' % (r, c)].append(grid[i][j])
    return True


def sudoku2_awesome(grid):
    """
    吊的一批
    :param grid:
    :return:
    """

    def unique(G):
        G = [x for x in G if x != '.']
        return len(set(G)) == len(G)

    def groups(A):
        B = zip(*A)
        for v in xrange(9):
            yield A[v]
            yield B[v]
            yield [A[v / 3 * 3 + r][v % 3 * 3 + c]
                   for r in xrange(3) for c in xrange(3)]

    return all(unique(grp) for grp in groups(grid))


if __name__ == '__main__':
    grid = [[".", "9", ".", ".", "4", ".", ".", ".", "."],
            ["1", ".", ".", ".", ".", ".", "6", ".", "."],
            [".", ".", "3", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", "7", ".", ".", ".", ".", "."],
            ["3", ".", ".", ".", "5", ".", ".", ".", "."],
            [".", ".", "7", ".", ".", "4", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", "7", ".", ".", ".", "."]]
    print 'result', sudoku2(grid)
