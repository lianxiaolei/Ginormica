#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import random as rd


def sigmoid(in_x):
    return 1.0 / (1 + np.exp(-in_x))


def gradient_asscent(x, y, alpha=0.001, cycles=400):
    data_matrix = np.mat(x)
    label_matrix = np.mat(y)
    m, n = np.shape(data_matrix)
    weight_matrix = np.random.random((n, 1))
    for i in range(0, cycles):
        sigmoid_val = sigmoid(data_matrix * weight_matrix)

        error = label_matrix - sigmoid_val
        print 'error', np.abs(error).mean()

        gradient = alpha * data_matrix.T * error

        weight_matrix += gradient
    return weight_matrix


def stochastic_dg(x, y, alpha=0.001, cycles=400, min_batch=12):
    data_matrix = np.mat(x)
    label_matrix = np.mat(y)
    m, n = np.shape(data_matrix)
    weight_matrix = np.ones((n, 1))
    for iter in range(0, cycles):
        tmp_indexs = range(m)
        rd.shuffle(tmp_indexs)
        for i in range(int(np.ceil(m / min_batch)) + 1):
            sigmoid_val = sigmoid(
                data_matrix[i * min_batch: min((i + 1) * min_batch, len(data_matrix)), :] * weight_matrix)

            error = label_matrix[i * min_batch: min((i + 1) * min_batch, len(data_matrix)), :] - sigmoid_val
            print 'error', np.abs(error).mean()

            gradient = alpha * data_matrix[i * min_batch: min((i + 1) * min_batch, len(data_matrix)), :].T * error

            weight_matrix += gradient
    return weight_matrix


if __name__ == '__main__':
    import tech.lian.python.utils.load_data_set as ld

    features, labels = ld.load_data_set_with_bias('testSet.txt')
    weights = stochastic_dg(features, labels)
    print weights
