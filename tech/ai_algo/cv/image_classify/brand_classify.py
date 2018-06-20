# -*- coding: utf-8 -*-

import cv2
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical


class BrandClassify(object):

    def __init__(self, batch_size=2, gene=1, width=600, height=300, num_class=100):
        self.datagen = image.ImageDataGenerator(featurewise_center=False,
                                                samplewise_center=False,
                                                featurewise_std_normalization=False,
                                                samplewise_std_normalization=False,
                                                zca_whitening=False,
                                                rotation_range=0.,
                                                width_shift_range=0.,
                                                height_shift_range=0.,
                                                shear_range=0.,
                                                zoom_range=0.,
                                                channel_shift_range=0.,
                                                fill_mode='nearest',
                                                cval=0.0,
                                                horizontal_flip=False,
                                                vertical_flip=False,
                                                rescale=None,
                                                preprocessing_function=None,
                                                data_format=K.image_data_format(),
                                                )
        self.labels = None
        self.files = None
        self.batch_size = batch_size
        self.gene = gene
        self.width = width
        self.height = height
        self.num_class = num_class
        self.base_model = None
        self.model = None

    def read_data(self, flag, path='../../../../assets/brand_images'):
        """
        read the dataset，dir format must be: ./train/, ./train.txt, ./test/, ./test.txt
        :param flag: 'train' or 'test'
        :param path
        """
        content = open(os.path.join(path, '%s.txt' % flag))

        imgs = []
        labels = []

        lines = content.readlines()
        # tqdm is the progress bar, please install it with "pip install tqdm".
        # if u wan't, you can replace tqdm(lines) with lines.
        for i in lines:
            fname, y = i.replace('\n', '').split(' ')
            y = int(y)

            x = os.path.join(path, flag, fname)

            imgs.append(x)
            labels.append(y)
        return np.array(imgs), np.array(labels)

    def data_gen(self, batch_size=128, gene=4):
        """
        generate the data with the size batch_size
        :param batch_size:
        :param gene:
        :return:
        """

        while True:

            index = np.random.choice(len(self.labels), batch_size, replace=False)
            label_list = self.labels[index]
            file_list = self.files[index]

            X_ = []
            for fname in file_list:
                img = cv2.imread(fname)
                img = cv2.resize(img, (self.width, self.height))
                X_.append(img.tolist())
            X_ = np.array(X_)
            y_ = np.array(label_list)

            i = 0
            X = None
            y = None

            for batch in self.datagen.flow(X_, y_, batch_size=batch_size):
                if not type(X) == np.ndarray:
                    X = batch[0]
                    y = batch[1]
                else:
                    X = np.concatenate([X, batch[0]], axis=0)
                    y = np.concatenate([y, batch[1]], axis=0)

                i += 1
                if i >= gene:
                    break
            import tensorflow as tf

            # y_onehot = K.one_hot(np.array(y), num_classes=100)
            y_onehot = to_categorical(np.array(y), num_classes=100)
            # print('input shape:', type(np.array(X)), np.array(X).shape, type(y_onehot), y_onehot.shape)

            yield (np.array(X), y_onehot)
            # yield np.array(X), y_onehot, np.ones(batch_size * gene)

    def model_struct(self):
        input_tensor = Input((self.height, self.width, 3))
        # vgg = VGG16(weights='../../../../models/VGG16_WEIGHTS.h5', include_top=False, input_tensor=input_tensor)
        # print('the last vgg layer is ', vgg.layers[-1])
        # print('the last vgg layer output is ', vgg.output)

        # tensor_shape = vgg.output.shape
        # print(tensor_shape)

        # rnn_length = tensor_shape[1].value
        # rnn_dimen = tensor_shape[2].value * tensor_shape[3].value
        # units = tensor_shape[3].value
        # print(rnn_length, rnn_dimen, units)
        x = input_tensor
        for i in range(3):
            x = Conv2D(32 * 2 ** i, (3, 3), kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(32 * 2 ** 2, (3, 3), kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPool2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(128, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(100, kernel_initializer='he_normal', activation='softmax')(x)
        # print('now x\'s shape:', x.shape)

        # self.base_model = Model(input=input_tensor, output=x)
        labels = Input(name='labels', shape=[self.num_class], dtype='float32')
        self.model = Model(inputs=input_tensor, outputs=x)

    def train(self):
        self.files, self.labels = self.read_data('train')
        # print('读取数据', len(self.files), len(self.labels))
        # print(next(self.data_gen(1, 1)))
        self.model_struct()
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        # print('model:', self.model)
        self.model.fit_generator(
            self.data_gen(self.batch_size, self.gene),
            steps_per_epoch=100,
            epochs=20,
            validation_data=self.data_gen(20, 1),
            validation_steps=10
        )
        self.base_model.save('brand_classify_vgg16.h5')
        self.base_model.save_weights('brand_classify_vgg16_weights.h5')


if __name__ == '__main__':
    bc = BrandClassify()
    bc.train()
    print('done')