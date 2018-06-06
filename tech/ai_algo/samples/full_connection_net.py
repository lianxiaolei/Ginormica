#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import scipy.io as scio


def load_data_set():
    """
    加载数据集
    :return:
    """
    data_set = scio.loadmat('yaleB_face_dataset.mat')
    return data_set['unlabeled_data'], data_set['trainData'], data_set['train_labels'], \
        data_set['testData'], data_set['test_labels']


def sigmoid(in_x):
    """
    激活函数
    :param in_x:
    :return:
    """
    return 1.0 / (1 + np.exp(-in_x))


def feed_forward(w, a, x):
    """
    前向传播
    :param w:
    :param a:
    :param x:
    :return:
    """
    w = np.array(w)  # feature * number
    temp_array = np.concatenate((a, x), axis=0)
    z = w.dot(temp_array)
    a_next = sigmoid(z)
    return a_next, z


def back_propagation(w, z, delta_next):
    """
    反向传播
    :param w:
    :param z:
    :param delta_next:
    :return:
    """
    f = lambda s: 1 / (1 + np.exp(-s))
    df = lambda s: f(s) * 1 - f(s)
    delta = df(z) * np.dot(w.T, delta_next)
    return delta


def normalization(source_data, dataset_size):
    """
    归一化数据集
    :param unlabeledData:
    :param dataset_size:
    :return:
    """
    result_data = np.zeros(source_data.shape)  # 即将归一化的数据集，首先初始化为0
    # 利用z-score归一化方法归一数据，按列来归一化，先除以最大值，再进行z-score归一化
    for i in range(dataset_size):
        tmp = source_data[:, i] / 255
        result_data[:, i] = (tmp - np.mean(tmp)) / np.std(tmp)
    return result_data


def auto_encoder_train(unlabeled_data):
    """
    自编码器训练
    :return:
    """
    dataset_size = 80  # 我们所准备无标签的人脸图片数据数量

    unlabeled_data = normalization(unlabeled_data, dataset_size)  # 归一化数据集

    '''初始化参数'''
    alpha = 0.5  # 学习步长
    max_epoch = 300  # 自编码器训练总次数
    mini_batch = 10  # 最小批训练时，每次使用10个样本同时进行训练
    height = 48  # 人脸数据图片的高度
    width = 42  # 人脸数据图片的宽度
    img_size = height * width
    hidden_node = 60  # 网络隐藏层节点数
    layer_struct = [
        [img_size, 1],
        [0, hidden_node],
        [0, img_size]
    ]
    layer_num = 3

    '''随机初始化权值矩阵，大小为下一层的层数*本层内外结点数之和'''
    w = []
    for l in range(layer_num - 1):
        w.append(np.random.randn(layer_struct[l + 1][1], sum(layer_struct[l])))

    X = []
    X.append(np.array(unlabeled_data))
    for l in range(1, layer_num):
        X.append(np.zeros((0, dataset_size)))

    delta = []
    for l in range(layer_num):
        delta.append([])

    count = 0
    for ite in range(max_epoch):
        index = list(range(dataset_size))
        rd.shuffle(index)

        a = []
        z = []
        z.append([])
        for i in range(int(np.ceil(float(dataset_size) / mini_batch))):
            #  初始化所有层外部结点
            x = []
            for l in range(layer_num):
                x.append(X[l][:, index[i * mini_batch: min((i + 1) * mini_batch, dataset_size)]])
            #  内部结点第一层初值
            a.append(np.zeros((layer_struct[0][1], min((i + 1) * mini_batch, dataset_size) - i * mini_batch)))

            y = unlabeled_data[:, index[i * mini_batch: min((i + 1) * mini_batch, dataset_size)]]

            #  前向传播
            for l in range(layer_num - 1):
                a_next, z_now = feed_forward(w[l], a[l], x[l])
                z.append(z_now)
                a.append(a_next)

            # 输出层误差
            delta[layer_num - 1] = (a[layer_num - 1] - y) * a[layer_num - 1] * (1 - a[layer_num - 1])

            #  反向传播
            for l in range(layer_num - 2, 0, -1):
                delta[l] = back_propagation(w[l], z[l], delta[l + 1])

            for l in range(layer_num - 1):
                dw = np.dot(delta[l + 1], np.concatenate((a[l], x[l]), axis=0).T) / mini_batch
                w[l] = w[l] - alpha * dw
        count += 1

    return w[0]


def nn_train(train_data, train_labels):
    """
    有监督训练
    :param w:
    :param train_data:
    :param train_labels:
    :return:
    """
    err = []
    acc = []

    hidden_node = 60
    dataset_size = 56

    max_epoch = 200
    mini_batch = 14
    alpha = 0.5
    height = 48  # 人脸数据图片的高度
    width = 42  # 人脸数据图片的宽度
    img_size = height * width
    #  [外部结点, 内部结点]
    layer_struct = [
        [img_size, 1],
        [0, hidden_node],
        [0, 4]
    ]
    layer_num = 3

    w = []
    for l in range(layer_num - 1):
        w.append(np.random.randn(layer_struct[l + 1][1], sum(layer_struct[l])))

    delta = []
    for l in range(layer_num):
        delta.append([])

    train_data = normalization(train_data, dataset_size)

    delta = []
    for l in range(layer_num):
        delta.append([])

    X = []
    X.append(np.array(train_data))
    for l in range(1, layer_num):
        X.append(np.zeros((0, dataset_size)))

    for ite in range(max_epoch):
        index = list(range(dataset_size))
        rd.shuffle(index)

        for i in range(int(np.ceil(dataset_size / mini_batch))):
            a = []
            z = []
            z.append([])
            a.append(np.zeros((1, min((i + 1) * mini_batch, dataset_size) - i * mini_batch)))

            x = []
            for l in range(layer_num):
                x.append(X[l][:, index[i * mini_batch: min((i + 1) * mini_batch, dataset_size)]])

            for l in range(layer_num - 1):
                a_next, z_now = feed_forward(w[l], a[l], x[l])
                a.append(a_next)
                z.append(z_now)

            y = train_labels[:, i * mini_batch: min((i + 1) * mini_batch, dataset_size)]

            delta[layer_num - 1] = (a[layer_num - 1] - y) * a[layer_num - 1] * (1 - a[layer_num - 1])

            err.append(np.mean(np.abs(a[layer_num - 1] - y)))
            print err[-1]
            for l in range(layer_num - 2, 0, -1):
                delta[l] = back_propagation(w[l], z[l], delta[l + 1])

            for l in range(layer_num - 1):
                dw = np.dot(delta[l + 1], np.concatenate((a[l], x[l]), axis=0).T) / mini_batch
                w[l] = w[l] - alpha * dw

if __name__ == '__main__':
    unlabeled_data, train_data, train_labels, test_data, test_labels = load_data_set()
    # w = auto_encoder_train(unlabeled_data)
    nn_train(train_data, train_labels)
    import pymc as pm
    pm.MCMC()
