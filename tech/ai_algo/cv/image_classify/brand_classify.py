# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
from keras.models import Model
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import os
import skimage
from tqdm import *
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.preprocessing import image


def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(Concatenate(axis=0)(outputs))

        return Model(model.inputs, merged)


def get_img_with_fname(fname, size=None):
    """

    :param fname:
    :param size:
    :return:
    """
    img = cv2.imread(fname)
    if size:
        img = cv2.resize(img, size)
    return img


def read_data(flag, path='../../../../assets/brand_images'):
    #     train = open(os.path.join(path, 'train.txt'))
    #     text = open(os.path.join(path, 'test.txt'))
    content = open(os.path.join(path, '%s.txt' % flag))

    imgs = []
    labels = []

    lines = content.readlines()
    for i in tqdm(lines):
        fname, y = i.replace('\n', '').split(' ')
        y = int(y)
        #         print(os.path.join(path, 'train', fname))
        x = get_img_with_fname(os.path.join(path, flag, fname), size=(600, 300))

        imgs.append(x)
        labels.append(y)
    return np.array(imgs), np.array(labels)


X, y = read_data('train')

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, random_state=0)
print(trainX.shape, testX.shape)

datagen = image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 0.,
    width_shift_range = 0.,
    height_shift_range = 0.,
    shear_range = 0.,
    zoom_range = 0.,
    channel_shift_range = 0.,
    fill_mode = 'nearest',
    cval = 0.0,
    horizontal_flip = False,
    vertical_flip = False,
    rescale = None,
    preprocessing_function = None,
    data_format = K.image_data_format(),
)

i = 0
XX = None
yy = None
for batch in datagen.flow(trainX, trainy, batch_size=len(trainX)):
    print(batch[0].shape, batch[1].shape)
    print(batch[1])
    if not type(XX) == np.ndarray:
        XX = batch[0]
        yy = batch[1]
    else:
        XX = np.concatenate([XX, batch[0]], axis=0)
        yy = np.concatenate([yy, batch[1]], axis=0)

    i += 1
    if i >= 4:
        break
print(XX.shape)
yy_onehot = K.one_hot(yy, num_classes=100)
print(yy_onehot.shape)

weight_decay = 0.0005
nb_epoch=100
batch_size=32
width = 600
height = 300

input_tensor = Input((height, width, 3))


# vgg = VGG16(weights='../../../../models/VGG16_WEIGHTS.h5', include_top=False, input_tensor=input_tensor)
vgg = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
print(vgg.layers[-1])
# print(' '.join())
print(vgg.output)

tensor_shape = vgg.output.shape
print(tensor_shape)

rnn_length = tensor_shape[1].value
rnn_dimen = tensor_shape[2].value * tensor_shape[3].value
units = tensor_shape[3].value

print(rnn_length, rnn_dimen, units)

x = Flatten()(vgg.output)
x = Dense(128, kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.25)(x)
x = Dense(100, kernel_initializer='he_normal', activation='softmax')(x)
print('now x\'s shape:', x.shape)

base_model = Model(input=input_tensor, output=x)

base_model.compile(loss='mean_squared_error', optimizer='adam')

base_model.fit(XX, yy_onehot)

base_model.save('vgg_bottleneck_classify.h5')
base_model.save_weights('vgg_bottleneck_classify_weights.h5')

print('all done')