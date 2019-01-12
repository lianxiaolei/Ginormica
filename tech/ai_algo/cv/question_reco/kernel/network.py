# coding:utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from abc import ABCMeta, abstractmethod

import numpy as np


class CRNN():
    """

    """
    def __init__(self):
        pass

    def _init_variable(self, shape, name=None):
        return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

    def _change_size(self, input_shape, channel=None, auto_change_channel=True):
        input_shape[2] = input_shape[2] // 2
        if auto_change_channel:
            input_shape[3] = input_shape[3] * 2
        if not channel
            input_shape[3] = channel
        return input_shape

    def image2head(self, x):
        for i in range(3):
            x = tf.nn.conv2d(x, eval('self.w%s0' % i), [1, 2, 2, 2], padding='same', name=)
            tf.nn.relu(x)

            x = tf.nn.conv2d(x, eval('self.w%s1' % i), [1, 2, 2, 2], padding='same', name=)
            tf.nn.relu(x)

            x = tf.nn.max_pool(x, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='valid', name=)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)
        return x

    def head2tail(self, x, out):
        x = slim.fully_connected(x, )

    def ctc_loss(self, x, label):
        return tf.nn.ctc_loss(label, x, sequence_length=len(label))

    def build_model(self, input_shape, output_shape, keep_prob=1.0, lr=1e-2, epoch=1e1, mode='train'):
        X = tf.placeholder(shape=input_shape)
        y = tf.placeholder(shape=output_shape)

        self.keep_prob = keep_prob

        self.w00 = self._init_variable(self._change_size(input_shape, channel=32,
                                                        auto_change_channel=False), name='conv_w00')
        self.w01 = self._init_variable(self.w00.shape, name='conv_w01')

        self.w10 = self._init_variable(self._change_size(input_shape), name='conv_w10')
        self.w11 = self._init_variable(self.w10.shape, name='conv_w11')

        self.w20 = self._init_variable(self._change_size(input_shape), name='conv_w20')
        self.w21 = self._init_variable(self._change_size(input_shape), name='conv_w21')

        head = self.image2head(X)



