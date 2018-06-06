# coding: utf-8

def bin_search(k, arr):
    """
    二分查找指定元素所在的index
    :param k: 指定元素
    :param arr: 数组
    :return: index
    """
    print arr
    if not arr: return -1
    if len(arr) == 1 and not k == arr[0]: return -1

    idx = -1
    s = 0  # start index
    e = len(arr)  # end index
    mid = len(arr) / 2  # mid index

    if k < arr[mid]:
        idx = bin_search(k, arr[s: mid])
    elif k > arr[mid]:
        tmp = bin_search(k, arr[mid: e])
        if tmp == -1: return -1
        idx = mid + tmp
    elif k == arr[mid]:
        tmp = bin_search(k, arr[s: mid])
        if tmp == -1:
            return mid
        else:
            return tmp
    return idx


if __name__ == '__main__':
    arr = [3, 4, 5, 8, 8, 8, 8, 10, 13, 14]
    k = 8
    print bin_search(k, arr)
