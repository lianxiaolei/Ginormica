#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys

reload(sys)
sys.setdefaultencoding('utf8')


def create_data_set():
    """
    创建数据集
    :return:
    """
    dataSet = [['younger', 'no', 'no', '一般', 'reject'],
               ['younger', 'no', 'no', '好', 'reject'],
               ['younger', 'yes', 'no', '好', 'accept'],
               ['younger', 'yes', 'yes', '一般', 'accept'],
               ['younger', 'no', 'no', '一般', 'reject'],
               ['adult', 'no', 'no', '一般', 'reject'],
               ['adult', 'no', 'no', '好', 'reject'],
               ['adult', 'yes', 'yes', '好', 'accept'],
               ['adult', 'no', 'yes', '非常好', 'accept'],
               ['adult', 'no', 'yes', '非常好', 'accept'],
               ['elder', 'no', 'yes', '非常好', 'accept'],
               ['elder', 'no', 'yes', '好', 'accept'],
               ['elder', 'yes', 'no', '好', 'accept'],
               ['elder', 'yes', 'no', '非常好', 'accept'],
               ['elder', 'no', 'no', '一般', 'reject'],
               ]
    labels = ['age', 'have job', 'have house', 'credit position']
    # 返回数据集和每个维度的名称
    return dataSet, labels


def load_data_set():
    '''
    加载数据集
    :return:
    '''
    # data_set = np.recfromcsv('../../../data/watermelon3.csv')
    data = pd.read_csv('../../../data/watermelon3.csv',
                       header=0)
    labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    matrix_data = data.as_matrix()[:, 1:]
    matrix_data = np.delete(matrix_data, -3, axis=1)
    matrix_data = np.delete(matrix_data, -2, axis=1)
    return matrix_data, labels


def majority_cnt(class_list):
    """
    返回出现次数最多的分类名称 当只有一个特征的时候使用
    :param class_list: 类列表
    :return: 出现次数最多的类名称
    """
    class_count = {}
    for key in class_list:
        class_count.setdefault(key, 0)
        class_count[key] += 1
    class_count = sorted(class_count.iteritems(), key=lambda item: item[1], reverse=True)
    return class_count[0][0]


def split_data_set(data_set, feature, feature_value):
    """
    按特征值切分数据集
    :param data_set:
    :param feature:
    :param feature_value:
    :return:
    """
    splited_set = list()
    for data in data_set:
        if data[feature] == feature_value:
            item = data[:feature]
            item.extend(data[feature + 1:])
            splited_set.append(item)
    return splited_set


def calc_shannon_ent(data_set):
    """
    计算香农熵
    :param data_set:
    :return:
    """
    num_data = len(data_set)
    count_value = {}
    for item in data_set:
        count_value.setdefault(item[-1], 0.0)
        count_value[item[-1]] += 1
    shannon_ent = 0.0
    for feat_value, num in count_value.iteritems():
        prob = num / num_data
        shannon_ent = shannon_ent - prob * np.log2(prob)
    return shannon_ent


def calc_conditional_shannon_ent(data_set, feature, feature_values):
    """
    计算条件熵
    :param data_set:
    :param feature:
    :param feature_values: 唯一特征值集合
    :return:
    """
    ce = 0.0
    for feature_value in feature_values:
        sub_data_set = split_data_set(data_set, feature, feature_value)
        prob = float(len(sub_data_set)) / len(data_set)
        ce += prob * calc_shannon_ent(sub_data_set)
    return ce


def calc_info_gain(data_set, base_ent, feature):
    """
    计算信息增益
    :param data_set:
    :param base_ent:
    :param feature:
    :return:
    """
    feature_list = [item[feature] for item in data_set]
    unique_feature = set(feature_list)
    condition_ent = calc_conditional_shannon_ent(data_set, feature, unique_feature)
    info_gain = base_ent - condition_ent
    return info_gain


def calc_info_gain_rate(data_set, base_ent, feature):
    """
    计算信息增益率
    :param data_set:
    :param base_ent:
    :param feature:
    :return:
    """
    return calc_info_gain(data_set, base_ent, feature) / base_ent


def choose_best_feature_id3(data_set):
    """
    依据IDE选择最优特征值
    :param data_set:
    :return:
    """
    features = len(data_set[0]) - 1
    base_ent = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for feature in range(features):
        info_gain = calc_info_gain(data_set, base_ent, feature)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
    return best_feature


def create_tree(data_set, labels, chooseBestFeatureToSplitFunc=choose_best_feature_id3):
    """
    递归创建决策树
    :param data_set:
    :param labels:
    :param chooseBestFeatureToSplitFunc:
    :return:
    """
    print len(data_set[0]), len(labels)
    class_list = [item[-1] for item in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feature = chooseBestFeatureToSplitFunc(data_set)

    best_feature_label = labels[best_feature]
    tree = {best_feature_label: {}}

    del labels[best_feature]
    feature_values = [item[best_feature] for item in data_set]
    unique_feature_values = set(feature_values)
    for feature_value in unique_feature_values:
        sub_labels = labels[:]
        tree[best_feature_label][feature_value] = \
            create_tree(split_data_set(data_set, best_feature, feature_value), sub_labels)
    return tree


def cut(tree):
    for key, value in tree.iteritems():
        if isinstance(value, dict):
            sub_tree = value
            cut(sub_tree)


if __name__ == '__main__':
    # data_set, labels = create_data_set()
    # mytree = create_tree(data_set, labels)
    # print mytree
    # import treePlotter
    # from pylab import *
    #
    # # mpl.rcParams['font.sans-serif'] = ['Hei']  # 指定默认字体
    # # mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    #
    # treePlotter.createPlot(mytree)

    data_set, labels = load_data_set()
    mytree = create_tree(data_set.tolist(), labels)
    import json
    print json.dumps(mytree, encoding='utf8')

    # import treePlotter
    # from pylab import *
    #
    # mpl.rcParams['font.sans-serif'] = ['Hei']  # 指定默认字体
    # mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    #
    # treePlotter.createPlot(mytree)

    from